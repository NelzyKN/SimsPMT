#include "PhotonTriggerML.h"

#include <fwk/CentralConfig.h>
#include <utl/ErrorLogger.h>
#include <evt/Event.h>

#include <evt/ShowerSimData.h>
#include <sevt/SEvent.h>
#include <sevt/Station.h>
#include <sevt/PMT.h>
#include <sevt/PMTSimData.h>
#include <sevt/StationConstants.h>
#include <sdet/Station.h>
#include <sdet/PMTConstants.h>
#include <det/Detector.h>
#include <sdet/SDetector.h>
#include <utl/CoordinateSystemPtr.h>
#include <utl/TimeDistribution.h>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TGraph.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TPaveText.h>
#include <TLegend.h>
#include <TDirectory.h>
#include <TKey.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <csignal>
#include <iomanip>
#include <ctime>
#include <random>
#include <deque>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace det;
using namespace utl;

// ============================================================================
// Static globals / helpers (no header changes needed)
// ============================================================================

static const int    kTraceLen   = 2048;   // SD FADC length (fixed)
static const double kNsPerBin   = 25.0;   // 25 ns/bin
static const double kAdcPerVem  = 180.0;  // simple VEM scaling

// Small helper for end‑of‑run feature diagnostics (no header change)
struct RunningStats {
  double sum[17]   = {0};
  double sumsq[17] = {0};
  long   n         = 0;
  void add(const std::vector<double>& v) {
    for (size_t i=0;i<v.size() && i<17;i++){ sum[i]+=v[i]; sumsq[i]+=v[i]*v[i]; }
    n++;
  }
};
static RunningStats gPhotonStats, gHadronStats;

// Static instance pointer for signal handler
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

// Static ML results map
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Guard to avoid touching external ROOT files from a signal handler
static volatile sig_atomic_t gCalledFromSignal = 0;

// ---------------------------------------------------------------------------
// Signal handler
// ---------------------------------------------------------------------------
void PhotonTriggerMLSignalHandler(int sig)
{
  if (sig == SIGINT || sig == SIGTSTP) {
    std::cout << "\n\nInterrupt received. Saving PhotonTriggerML data...\n" << std::endl;
    gCalledFromSignal = 1;
    if (PhotonTriggerML::fInstance) {
      try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); }
      catch (...) { /* swallow */ }
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
    _Exit(0);
  }
}

// ============================================================================
// Neural network (unchanged shape; cleaned warnings)
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork()
: fInputSize(0), fHidden1Size(0), fHidden2Size(0),
  fTimeStep(0), fDropoutRate(0.1),
  fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int h1, int h2)
{
  fInputSize = input_size; fHidden1Size=h1; fHidden2Size=h2;

  std::cout<<"Initializing Physics NN: "<<input_size<<" -> "<<h1<<" -> "<<h2<<" -> 1\n";

  std::mt19937 gen(12345);
  std::uniform_real_distribution<> dist(-0.5,0.5);

  fWeights1.assign(h1, std::vector<double>(input_size));
  for (int i=0;i<h1;i++) for (int j=0;j<input_size;j++)
    fWeights1[i][j] = dist(gen)/std::sqrt((double)input_size);

  fWeights2.assign(h2, std::vector<double>(h1));
  for (int i=0;i<h2;i++) for (int j=0;j<h1;j++)
    fWeights2[i][j] = dist(gen)/std::sqrt((double)h1);

  fWeights3.assign(1, std::vector<double>(h2));
  for (int j=0;j<h2;j++) fWeights3[0][j] = dist(gen)/std::sqrt((double)h2);

  fBias1.assign(h1,0.0); fBias2.assign(h2,0.0); fBias3=0.0;

  fMomentum1_w1.assign(h1,std::vector<double>(input_size,0));
  fMomentum1_w2.assign(h2,std::vector<double>(h1,0));
  fMomentum1_w3.assign(1,std::vector<double>(h2,0));
  fMomentum2_w1.assign(h1,std::vector<double>(input_size,0));
  fMomentum2_w2.assign(h2,std::vector<double>(h1,0));
  fMomentum2_w3.assign(1,std::vector<double>(h2,0));
  fMomentum1_b1.assign(h1,0); fMomentum2_b1.assign(h1,0);
  fMomentum1_b2.assign(h2,0); fMomentum2_b2.assign(h2,0);
  fMomentum1_b3=0; fMomentum2_b3=0;

  fTimeStep = 0;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
  if ((int)x.size()!=fInputSize) return 0.5;

  std::vector<double> h1(fHidden1Size);
  for (int i=0;i<fHidden1Size;i++){
    double s=fBias1[i];
    for (int j=0;j<fInputSize;j++) s+=fWeights1[i][j]*x[j];
    h1[i]=1.0/(1.0+std::exp(-s));
    if (training && (rand()/double(RAND_MAX))<fDropoutRate) h1[i]=0.0;
  }

  std::vector<double> h2(fHidden2Size);
  for (int i=0;i<fHidden2Size;i++){
    double s=fBias2[i];
    for (int j=0;j<fHidden1Size;j++) s+=fWeights2[i][j]*h1[j];
    h2[i]=1.0/(1.0+std::exp(-s));
    if (training && (rand()/double(RAND_MAX))<fDropoutRate) h2[i]=0.0;
  }

  double out=fBias3;
  for (int j=0;j<fHidden2Size;j++) out+=fWeights3[0][j]*h2[j];
  return 1.0/(1.0+std::exp(-out));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y,
                                             double lr)
{
  if (X.empty() || X.size()!=y.size()) return -1.0;

  const int B = (int)X.size();
  const int n1=fHidden1Size, n2=fHidden2Size;

  // softer class weighting
  int nPhoton = std::count(y.begin(), y.end(), 1);
  const double wPhoton = (nPhoton>0)?1.5:1.0;
  const double wHadron = 1.0;

  std::vector<std::vector<double>> gw1(n1, std::vector<double>(fInputSize,0));
  std::vector<std::vector<double>> gw2(n2, std::vector<double>(n1,0));
  std::vector<std::vector<double>> gw3(1 , std::vector<double>(n2,0));
  std::vector<double> gb1(n1,0), gb2(n2,0); double gb3=0;
  double totalLoss=0.0;

  for (int s=0;s<B;s++){
    const auto& x = X[s]; const int label=y[s];
    const double w = (label==1)?wPhoton:wHadron;

    std::vector<double> h1(n1), h1r(n1);
    for (int i=0;i<n1;i++){
      double sum=fBias1[i];
      for (int j=0;j<fInputSize;j++) sum+=fWeights1[i][j]*x[j];
      h1r[i]=sum; h1[i]=1.0/(1.0+std::exp(-sum));
    }

    std::vector<double> h2(n2), h2r(n2);
    for (int i=0;i<n2;i++){
      double sum=fBias2[i];
      for (int j=0;j<n1;j++) sum+=fWeights2[i][j]*h1[j];
      h2r[i]=sum; h2[i]=1.0/(1.0+std::exp(-sum));
    }

    double outr=fBias3; for (int j=0;j<n2;j++) outr+=fWeights3[0][j]*h2[j];
    double out = 1.0/(1.0+std::exp(-outr));

    totalLoss += -w*( label*std::log(out+1e-7) + (1-label)*std::log(1-out+1e-7) );

    double go = w*(out - label);
    for (int j=0;j<n2;j++) gw3[0][j] += go*h2[j];
    gb3 += go;

    std::vector<double> gh2(n2,0.0);
    for (int j=0;j<n2;j++){
      gh2[j] = fWeights3[0][j]*go * h2[j]*(1.0-h2[j]);
    }
    for (int i=0;i<n2;i++){
      for (int j=0;j<n1;j++) gw2[i][j]+=gh2[i]*h1[j];
      gb2[i]+=gh2[i];
    }

    std::vector<double> gh1(n1,0.0);
    for (int j=0;j<n1;j++){
      for (int i=0;i<n2;i++) gh1[j]+=fWeights2[i][j]*gh2[i];
      gh1[j]*=h1[j]*(1.0-h1[j]);
    }
    for (int i=0;i<n1;i++){
      for (int j=0;j<fInputSize;j++) gw1[i][j]+=gh1[i]*x[j];
      gb1[i]+=gh1[i];
    }
  }

  // SGD + momentum
  fTimeStep++;
  const double mom=0.9;

  for (int i=0;i<fHidden1Size;i++){
    for (int j=0;j<fInputSize;j++){
      const double g = gw1[i][j]/B;
      fMomentum1_w1[i][j] = mom*fMomentum1_w1[i][j] - lr*g;
      fWeights1[i][j] += fMomentum1_w1[i][j];
    }
    const double g = gb1[i]/B;
    fMomentum1_b1[i] = mom*fMomentum1_b1[i] - lr*g;
    fBias1[i] += fMomentum1_b1[i];
  }
  for (int i=0;i<fHidden2Size;i++){
    for (int j=0;j<fHidden1Size;j++){
      const double g = gw2[i][j]/B;
      fMomentum1_w2[i][j] = mom*fMomentum1_w2[i][j] - lr*g;
      fWeights2[i][j] += fMomentum1_w2[i][j];
    }
    const double g = gb2[i]/B;
    fMomentum1_b2[i] = mom*fMomentum1_b2[i] - lr*g;
    fBias2[i] += fMomentum1_b2[i];
  }
  for (int j=0;j<fHidden2Size;j++){
    const double g = gw3[0][j]/B;
    fMomentum1_w3[0][j] = mom*fMomentum1_w3[0][j] - lr*g;
    fWeights3[0][j] += fMomentum1_w3[0][j];
  }
  {
    const double g = gb3/B;
    fMomentum1_b3 = mom*fMomentum1_b3 - lr*g;
    fBias3 += fMomentum1_b3;
  }

  return totalLoss/B;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& fn)
{
  std::ofstream f(fn.c_str());
  if (!f.is_open()) { std::cout<<"Error: cannot save weights to "<<fn<<"\n"; return; }
  f<<fInputSize<<" "<<fHidden1Size<<" "<<fHidden2Size<<"\n";

  for (const auto& row: fWeights1){ for (double w: row) f<<w<<" "; f<<"\n"; }
  for (double b: fBias1) { f<<b<<" "; } f<<"\n";
  for (const auto& row: fWeights2){ for (double w: row) f<<w<<" "; f<<"\n"; }
  for (double b: fBias2) { f<<b<<" "; } f<<"\n";
  for (double w: fWeights3[0]) { f<<w<<" "; } f<<"\n";
  f<<fBias3<<"\n";
  f.close();
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& fn)
{
  std::ifstream f(fn.c_str());
  if (!f.is_open()) { std::cout<<"Warning: cannot open weights "<<fn<<"\n"; return false; }

  f>>fInputSize>>fHidden1Size>>fHidden2Size;
  fWeights1.assign(fHidden1Size,std::vector<double>(fInputSize));
  fWeights2.assign(fHidden2Size,std::vector<double>(fHidden1Size));
  fWeights3.assign(1,std::vector<double>(fHidden2Size));
  fBias1.assign(fHidden1Size,0.0); fBias2.assign(fHidden2Size,0.0);

  for (auto& row: fWeights1) for (double& w: row) f>>w;
  for (double& b: fBias1) f>>b;
  for (auto& row: fWeights2) for (double& w: row) f>>w;
  for (double& b: fBias2) f>>b;
  for (double& w: fWeights3[0]) f>>w;
  f>>fBias3;
  f.close();
  std::cout<<"Loaded weights from "<<fn<<"\n";
  return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights(){ fIsQuantized=true; }

// ============================================================================
// PhotonTriggerML Implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML()
: fNeuralNetwork(std::make_unique<NeuralNetwork>()),
  fIsTraining(true),
  fTrainingEpochs(500),
  fTrainingStep(0),
  fBestValidationLoss(1e9),
  fEpochsSinceImprovement(0),
  fEventCount(0),
  fStationCount(0),
  fPhotonLikeCount(0),
  fHadronLikeCount(0),
  fEnergy(0),
  fCoreX(0), fCoreY(0),
  fPrimaryId(0), fPrimaryType("Unknown"),
  fPhotonScore(0), fConfidence(0),
  fDistance(0), fStationId(0),
  fIsActualPhoton(false),
  fOutputFile(nullptr), fMLTree(nullptr),
  fLogFileName("photon_trigger_ml_physics.log"),
  hPhotonScore(nullptr), hPhotonScorePhotons(nullptr), hPhotonScoreHadrons(nullptr),
  hConfidence(nullptr), hRisetime(nullptr), hAsymmetry(nullptr), hKurtosis(nullptr),
  hScoreVsEnergy(nullptr), hScoreVsDistance(nullptr), gROCCurve(nullptr),
  hConfusionMatrix(nullptr), hTrainingLoss(nullptr), hValidationLoss(nullptr),
  hAccuracyHistory(nullptr),
  fTruePositives(0), fFalsePositives(0), fTrueNegatives(0), fFalseNegatives(0),
  fPhotonThreshold(0.65),
  fEnergyMin(1e18), fEnergyMax(1e19),
  fOutputFileName("photon_trigger_ml_physics.root"),
  fWeightsFileName("photon_trigger_weights_physics.txt"),
  fLoadPretrainedWeights(true)
{
  fInstance=this;
  fFeatureMeans.assign(17,0.0);
  fFeatureStdDevs.assign(17,1.0);
}

PhotonTriggerML::~PhotonTriggerML(){}

VModule::ResultFlag PhotonTriggerML::Init()
{
  INFO("PhotonTriggerML::Init()");
  // open log (classic format kept)
  fLogFile.open(fLogFileName.c_str());
  if (!fLogFile.is_open()){ ERROR("Cannot open log file"); return eFailure; }
  time_t now=time(0);
  fLogFile<<"==========================================\n";
  fLogFile<<"PhotonTriggerML Physics-Based Version Log\n";
  fLogFile<<"Date: "<<ctime(&now);
  fLogFile<<"==========================================\n\n"; fLogFile.flush();

  // network
  fNeuralNetwork->Initialize(17,8,4);
  if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)){
    fIsTraining=false;
  }

  // ROOT outputs
  fOutputFile = new TFile(fOutputFileName.c_str(),"RECREATE");
  if (!fOutputFile || fOutputFile->IsZombie()){ ERROR("Cannot create ROOT output"); return eFailure; }

  fMLTree = new TTree("MLTree","PhotonTriggerML Physics Tree");
  fMLTree->Branch("eventId",&fEventCount,"eventId/I");
  fMLTree->Branch("stationId",&fStationId,"stationId/I");
  fMLTree->Branch("energy",&fEnergy,"energy/D");
  fMLTree->Branch("distance",&fDistance,"distance/D");
  fMLTree->Branch("photonScore",&fPhotonScore,"photonScore/D");
  fMLTree->Branch("confidence",&fConfidence,"confidence/D");
  fMLTree->Branch("primaryId",&fPrimaryId,"primaryId/I");
  fMLTree->Branch("primaryType",&fPrimaryType);
  fMLTree->Branch("isActualPhoton",&fIsActualPhoton,"isActualPhoton/O");

  // Histograms (keep old layout)
  hPhotonScore          = new TH1D("hPhotonScore","ML Photon Score (All);Score;Count",50,0,1);
  hPhotonScorePhotons   = new TH1D("hPhotonScorePhotons","ML Score (True Photons);Score;Count",50,0,1);
  hPhotonScoreHadrons   = new TH1D("hPhotonScoreHadrons","ML Score (True Hadrons);Score;Count",50,0,1);
  hPhotonScorePhotons->SetLineColor(kBlue); hPhotonScoreHadrons->SetLineColor(kRed);
  hConfidence           = new TH1D("hConfidence","ML Confidence;|Score-0.5|;Count",50,0,0.5);
  hRisetime             = new TH1D("hRisetime","Rise Time 10-90%;Time [ns];Count",50,0,1000);
  hAsymmetry            = new TH1D("hAsymmetry","Pulse Asymmetry;(fall-rise)/(fall+rise);Count",50,-1,1);
  hKurtosis             = new TH1D("hKurtosis","Signal Kurtosis;Kurtosis;Count",50,-5,20);
  hScoreVsEnergy        = new TH2D("hScoreVsEnergy","Score vs Energy;Energy [eV];Score",50,1e17,1e20,50,0,1);
  hScoreVsDistance      = new TH2D("hScoreVsDistance","Score vs Distance;Distance [m];Score",50,0,3000,50,0,1);
  hConfusionMatrix      = new TH2D("hConfusionMatrix","Confusion Matrix;Predicted;Actual",2,-0.5,1.5,2,-0.5,1.5);
  hConfusionMatrix->GetXaxis()->SetBinLabel(1,"Hadron");
  hConfusionMatrix->GetXaxis()->SetBinLabel(2,"Photon");
  hConfusionMatrix->GetYaxis()->SetBinLabel(1,"Hadron");
  hConfusionMatrix->GetYaxis()->SetBinLabel(2,"Photon");
  hTrainingLoss         = new TH1D("hTrainingLoss","Training Loss;Batch;Loss",10000,0,10000);
  hValidationLoss       = new TH1D("hValidationLoss","Validation Loss;Batch;Loss",10000,0,10000);
  hAccuracyHistory      = new TH1D("hAccuracyHistory","Accuracy History;Batch;Accuracy [%]",10000,0,10000);

  signal(SIGINT,  PhotonTriggerMLSignalHandler);
  signal(SIGTSTP, PhotonTriggerMLSignalHandler);

  return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
  fEventCount++;
  ClearMLResults();

  // header every 50 events
  if (fEventCount % 50 == 1){
    std::cout<<"\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n";
    std::cout<<"│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│\n";
    std::cout<<"├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n";
  }

  // Shower info
  fEnergy=0; fCoreX=fCoreY=0; fPrimaryId=0; fPrimaryType="Unknown"; fIsActualPhoton=false;
  if (event.HasSimShower()){
    const ShowerSimData& sh = event.GetSimShower();
    fEnergy = sh.GetEnergy(); fPrimaryId=sh.GetPrimaryParticle();
    switch (fPrimaryId){
      case 22:  fPrimaryType="photon"; fIsActualPhoton=true; break;
      case 11: case -11: fPrimaryType="electron"; break;
      case 2212: fPrimaryType="proton"; break;
      case 1000026056: fPrimaryType="iron"; break;
      default: fPrimaryType = (fPrimaryId>1000000000)?"nucleus":"unknown";
    }
    fParticleTypeCounts[fPrimaryType]++;
    if (sh.GetNSimCores()>0){
      const Detector& det=Detector::GetInstance();
      const CoordinateSystemPtr& cs = det.GetSiteCoordinateSystem();
      Point core = sh.GetSimCore(0);
      fCoreX = core.GetX(cs); fCoreY=core.GetY(cs);
    }
    if (fEventCount<=5){
      std::cout<<"\nEvent "<<fEventCount<<": Energy="<<(fEnergy/1e18)<<" EeV, Primary="<<fPrimaryType
               <<" (ID="<<fPrimaryId<<")\n";
    }
  }

  // Stations
  if (event.HasSEvent()){
    const sevt::SEvent& sevent = event.GetSEvent();
    for (sevt::SEvent::ConstStationIterator it=sevent.StationsBegin(); it!=sevent.StationsEnd(); ++it){
      try { ProcessStation(*it); }
      catch (...) { /* be robust across odd stations */ }
    }
  }

  // Update metrics every 10 events
  if (fEventCount % 10 == 0){
    CalculateAndDisplayMetrics();
    hConfusionMatrix->SetBinContent(1,1,fTrueNegatives);
    hConfusionMatrix->SetBinContent(2,1,fFalsePositives);
    hConfusionMatrix->SetBinContent(1,2,fFalseNegatives);
    hConfusionMatrix->SetBinContent(2,2,fTruePositives);
  }

  // Train a bit
  if (fIsTraining && fTrainingFeatures.size()>=20 && (fEventCount%5==0)){
    double val_loss = TrainNetwork(); // also fills hValidationLoss
    int batch = fEventCount/5;
    hTrainingLoss->SetBinContent(batch, val_loss);

    const int correct = fTruePositives+fTrueNegatives;
    const int total   = fTruePositives+fTrueNegatives+fFalsePositives+fFalseNegatives;
    if (total>0){
      const double acc = 100.0*double(correct)/double(total);
      hAccuracyHistory->SetBinContent(batch,acc);
    }
    if (val_loss < fBestValidationLoss){
      fBestValidationLoss=val_loss; fEpochsSinceImprovement=0;
      fNeuralNetwork->SaveWeights("best_"+fWeightsFileName);
    } else { fEpochsSinceImprovement++; }
  }

  return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
  fStationId = station.GetId();

  // geometry
  try{
    const Detector& detector = Detector::GetInstance();
    const sdet::SDetector& sdetector = detector.GetSDetector();
    const sdet::Station& detStation = sdetector.GetStation(fStationId);
    const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
    const double sx = detStation.GetPosition().GetX(siteCS);
    const double sy = detStation.GetPosition().GetY(siteCS);
    fDistance = std::sqrt( (sx-fCoreX)*(sx-fCoreX) + (sy-fCoreY)*(sy-fCoreY) );
  } catch (...) {
    fDistance=-1; return;
  }

  const int firstPMT = sdet::Station::GetFirstPMTId();

  // loop PMTs
  for (int p=0;p<3;p++){
    const int pmtId = firstPMT + p;
    if (!station.HasPMT(pmtId)) continue;

    const sevt::PMT& pmt = station.GetPMT(pmtId);

    std::vector<double> trace;
    bool ok=false;
    try { ok = ExtractTraceData(pmt, trace); }
    catch (...) { ok=false; }
    if (!ok || (int)trace.size()!=kTraceLen) continue;

    const double vmax = *std::max_element(trace.begin(), trace.end());
    const double vmin = *std::min_element(trace.begin(), trace.end());
    if (vmax - vmin < 10.0) continue; // tiny activity

    // features
    fFeatures = ExtractEnhancedFeatures(trace);
    // fill quick feature plots
    if (hRisetime)  hRisetime->Fill(fFeatures.risetime_10_90);
    if (hAsymmetry) hAsymmetry->Fill(fFeatures.asymmetry);
    if (hKurtosis)  hKurtosis->Fill(fFeatures.kurtosis);

    UpdateFeatureStatistics(fFeatures);

    // normalize (and physics emphasis)
    std::vector<double> x = NormalizeFeatures(fFeatures);
    if (x.size()>=17){
      x[1]*=2.0;  // risetime_10_90
      x[3]*=2.0;  // pulse_width
      x[7]*=1.5;  // peak_charge_ratio
      x[16]*=1.5; // secondary_peak_ratio
    }

    // predict + simple spike penalty
    double ml = fNeuralNetwork->Predict(x,false);
    double penalty=0.0;
    if (fFeatures.pulse_width<120.0 && fFeatures.peak_charge_ratio>0.35) penalty-=0.20;
    if (fFeatures.num_peaks<=2  && fFeatures.secondary_peak_ratio>0.60)  penalty-=0.10;

    fPhotonScore = std::max(0.0, std::min(1.0, ml + penalty));
    fConfidence  = std::fabs(fPhotonScore-0.5);

    // store for training/validation (modest oversampling)
    const bool isVal = (fStationCount%10==0);
    if (fIsTraining){
      if (!isVal){
        if (fIsActualPhoton){
          for (int c=0;c<2;c++){
            std::vector<double> v=x;
            static std::mt19937 g(1234567u);
            std::normal_distribution<> n(0.0,0.02);
            for (auto& e: v) e+=n(g);
            fTrainingFeatures.push_back(v);
            fTrainingLabels.push_back(1);
          }
        } else {
          fTrainingFeatures.push_back(x);
          fTrainingLabels.push_back(0);
        }
      } else {
        fValidationFeatures.push_back(x);
        fValidationLabels.push_back(fIsActualPhoton?1:0);
      }
    }

    fStationCount++;
    const bool isPhotonPred = (fPhotonScore > fPhotonThreshold);

    // ML result map
    MLResult res;
    res.photonScore=fPhotonScore; res.identifiedAsPhoton=isPhotonPred;
    res.isActualPhoton=fIsActualPhoton; res.vemCharge=fFeatures.total_charge;
    res.features=fFeatures; res.primaryType=fPrimaryType; res.confidence=fConfidence;
    fMLResultsMap[fStationId]=res;

    // confusion counts
    if (isPhotonPred){
      fPhotonLikeCount++;
      if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
    } else {
      fHadronLikeCount++;
      if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
    }

    // ROOT hists
    if (hPhotonScore) hPhotonScore->Fill(fPhotonScore);
    if (hConfidence)  hConfidence->Fill(fConfidence);
    if (fIsActualPhoton) { if (hPhotonScorePhotons) hPhotonScorePhotons->Fill(fPhotonScore); }
    else                { if (hPhotonScoreHadrons)  hPhotonScoreHadrons->Fill(fPhotonScore); }
    if (hScoreVsEnergy)   hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
    if (hScoreVsDistance) hScoreVsDistance->Fill(fDistance, fPhotonScore);

    if (fMLTree) fMLTree->Fill();

    // --- TSV diagnostics (no header modifications) -------------------------
    static std::ofstream sTSV;
    static bool header=false;
    if (!sTSV.is_open()){ sTSV.open("photon_trigger_scores.tsv"); }
    if (sTSV.is_open()){
      if (!header){
        sTSV<<"eventId\tstationId\tpmtId\tenergyEeV\tdistanceM\ttrueLabel\tpredLabel\tscore\t"
               "rise10_90\twidth\tpeakAmp\tcharge\tpeakChargeRatio\tsecPeakRatio\tasym\n";
        header=true;
      }
      sTSV<<fEventCount<<"\t"<<fStationId<<"\t"<<pmtId<<"\t"<<(fEnergy/1e18)<<"\t"<<fDistance<<"\t"
          <<(fIsActualPhoton?1:0)<<"\t"<<(isPhotonPred?1:0)<<"\t"<<std::fixed<<std::setprecision(4)<<fPhotonScore<<"\t"
          <<fFeatures.risetime_10_90<<"\t"<<fFeatures.pulse_width<<"\t"<<fFeatures.peak_amplitude<<"\t"
          <<fFeatures.total_charge<<"\t"<<fFeatures.peak_charge_ratio<<"\t"
          <<fFeatures.secondary_peak_ratio<<"\t"<<fFeatures.asymmetry<<"\n";
      sTSV.flush();
    }
    // accumulate simple feature stats (means/RMS) per true class
    {
      std::vector<double> v = {
        fFeatures.risetime_10_50, fFeatures.risetime_10_90, fFeatures.falltime_90_10,
        fFeatures.pulse_width, fFeatures.asymmetry, fFeatures.peak_amplitude,
        fFeatures.total_charge, fFeatures.peak_charge_ratio, fFeatures.smoothness,
        fFeatures.kurtosis, fFeatures.skewness, fFeatures.early_fraction,
        fFeatures.late_fraction, fFeatures.time_spread, fFeatures.high_freq_content,
        (double)fFeatures.num_peaks, fFeatures.secondary_peak_ratio
      };
      if (fIsActualPhoton) gPhotonStats.add(v); else gHadronStats.add(v);
    }
  } // PMT loop
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& out)
{
  out.clear();

  // 1) Direct FADC trace
  if (pmt.HasFADCTrace()){
    try{
      const auto& tr = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain); // utl::Trace<int>
      out.reserve(kTraceLen);
      for (int i=0;i<kTraceLen;i++) out.push_back(tr[i]); // operator[] is available
      return true;
    } catch (...) { /* fall through */ }
  }

  // 2) Simulation data trace
  if (pmt.HasSimData()){
    try{
      const sevt::PMTSimData& sd = pmt.GetSimData();
      if (sd.HasFADCTrace(sevt::StationConstants::eTotal)){
        const auto& tr = sd.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                         sevt::StationConstants::eTotal); // utl::TimeDistribution<int>
        out.reserve(kTraceLen);
        for (int i=0;i<kTraceLen;i++) out.push_back(tr[i]); // operator[] works; no .size()
        return true;
      }
    } catch (...) { /* fall through */ }
  }

  return false;
}

PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
  EnhancedFeatures F; // all zeros by default
  const int N = (int)trace.size();
  if (N<=0) return F;

  // baseline from first 100 bins
  double base=0.0; const int B=std::min(100,N);
  for (int i=0;i<B;i++) base+=trace[i];
  base/=std::max(1,B);

  std::vector<double> sig(N,0.0);
  int pbin=0; double pval=0.0, tsum=0.0;
  for (int i=0;i<N;i++){
    double v=trace[i]-base; if (v<0) v=0;
    sig[i]=v; tsum+=v; if (v>pval){ pval=v; pbin=i; }
  }
  if (pval<5.0 || tsum<10.0) return F;

  F.peak_amplitude   = pval / kAdcPerVem;
  F.total_charge     = tsum / kAdcPerVem;
  F.peak_charge_ratio= F.peak_amplitude / (F.total_charge+1e-3);

  // rise/fall at 10/50/90 %
  const double v10=0.10*pval, v50=0.50*pval, v90=0.90*pval;
  int r10=0, r50=0, r90=pbin;
  for (int i=pbin;i>=0;i--){
    if (sig[i]<=v90 && r90==pbin) r90=i;
    if (sig[i]<=v50 && r50==0)    r50=i;
    if (sig[i]<=v10){ r10=i; break; }
  }
  int f90=pbin, f10=N-1;
  for (int i=pbin;i<N;i++){
    if (sig[i]<=v90 && f90==pbin) f90=i;
    if (sig[i]<=v10){ f10=i; break; }
  }
  F.risetime_10_50 = std::abs(r50-r10)*kNsPerBin;
  F.risetime_10_90 = std::abs(r90-r10)*kNsPerBin;
  F.falltime_90_10 = std::abs(f10-f90)*kNsPerBin;

  // width (FWHM)
  const double half = 0.5*pval; int wl=r10, wr=f10;
  for (int i=r10;i<=pbin;i++){ if (sig[i]>=half){ wl=i; break; } }
  for (int i=pbin;i<N;i++){ if (sig[i]<=half){ wr=i; break; } }
  F.pulse_width = std::abs(wr-wl)*kNsPerBin;

  // asymmetry
  const double rise=F.risetime_10_90, fall=F.falltime_90_10;
  F.asymmetry = (fall - rise) / (fall + rise + 1e-3);

  // moments
  double meanT=0.0; for (int i=0;i<N;i++) meanT += i*sig[i];
  meanT /= (tsum+1e-3);
  double var=0.0, sk=0.0, ku=0.0;
  for (int i=0;i<N;i++){
    const double d=i-meanT, w=sig[i]/(tsum+1e-3);
    var += d*d*w; sk += d*d*d*w; ku += d*d*d*d*w;
  }
  const double sd = std::sqrt(var+1e-3);
  F.time_spread = sd*kNsPerBin;
  F.skewness    = sk/std::pow(sd,3.0);
  F.kurtosis    = ku/(var*var+1e-6) - 3.0;

  // early/late fractions (quarters)
  const int Q = N/4;
  double early=0.0, late=0.0;
  for (int i=0;i<Q;i++) early+=sig[i];
  for (int i=3*Q;i<N;i++) late +=sig[i];
  F.early_fraction = early/(tsum+1e-3);
  F.late_fraction  = late /(tsum+1e-3);

  // smoothness / high‑freq
  double ss=0.0; int cnt=0;
  for (int i=1;i<N-1;i++){
    if (sig[i]>0.1*pval){ double s2=sig[i+1]-2*sig[i]+sig[i-1]; ss+=s2*s2; cnt++; }
  }
  F.smoothness = std::sqrt( ss / (cnt+1) );
  double hf_en=0.0;
  for (int i=1;i<N-1;i++){ double d=sig[i+1]-sig[i-1]; hf_en+=d*d; }
  F.high_freq_content = hf_en / (tsum*tsum + 1e-3);

  // peaks (stricter threshold to avoid noise)
  F.num_peaks=0; double sec=0.0;
  const double thr=0.25*pval;
  for (int i=1;i<N-1;i++){
    if (sig[i]>thr && sig[i]>sig[i-1] && sig[i]>sig[i+1]){
      F.num_peaks++;
      if (i!=pbin && sig[i]>sec) sec=sig[i];
    }
  }
  F.secondary_peak_ratio = sec/(pval+1e-3);

  return F;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& f)
{
  std::vector<double> raw = {
    f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
    f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
    f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
    f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
  };

  static int n=0; n++;
  for (size_t i=0;i<raw.size();i++){
    const double d = raw[i]-fFeatureMeans[i];
    fFeatureMeans[i] += d/n;
    const double d2 = raw[i]-fFeatureMeans[i];
    fFeatureStdDevs[i] += d*d2;
  }
  if (n>1){
    for (size_t i=0;i<raw.size();i++){
      fFeatureStdDevs[i] = std::sqrt( std::max(0.0, fFeatureStdDevs[i]/(n-1)) );
      if (fFeatureStdDevs[i]<1e-3) fFeatureStdDevs[i]=1.0;
    }
  }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& f)
{
  std::vector<double> raw = {
    f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
    f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
    f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
    f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
  };
  std::vector<double> mins={0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
  std::vector<double> maxs={500,1000,1000,1000, 1,10,100,1,100,20, 5,1,1,1000,10,10,1};

  std::vector<double> z; z.reserve(raw.size());
  for (size_t i=0;i<raw.size();i++){
    double v = (raw[i]-mins[i])/(maxs[i]-mins[i]+1e-3);
    if (v<0) { v=0; }
    if (v>1) { v=1; }
    z.push_back(v);
  }
  return z;
}

double PhotonTriggerML::TrainNetwork()
{
  if (fTrainingFeatures.empty()) return 1e9;

  std::cout<<"\n  Training with "<<fTrainingFeatures.size()<<" samples...";

  std::vector<std::vector<double>> Bx;
  std::vector<int>                 By;

  const int maxS = std::min(100, (int)fTrainingFeatures.size());
  for (int i=0;i<maxS;i++){ Bx.push_back(fTrainingFeatures[i]); By.push_back(fTrainingLabels[i]); }

  const int nP = std::count(By.begin(), By.end(), 1);
  const int nH = (int)By.size() - nP;
  std::cout<<" (P:"<<nP<<" H:"<<nH<<")";

  const double lr=0.01;
  double total=0.0;
  for (int e=0;e<10;e++) total += fNeuralNetwork->Train(Bx,By,lr);
  const double train_loss = total/10.0;
  std::cout<<" Loss: "<<std::fixed<<std::setprecision(4)<<train_loss;

  // Validation + threshold calibration (F1‑oriented with a touch of Youden J)
  double val_loss=0.0;
  if (!fValidationFeatures.empty()){
    const int N=(int)fValidationFeatures.size();
    std::vector<double> p(N,0.0);
    for (int i=0;i<N;i++) p[i]=fNeuralNetwork->Predict(fValidationFeatures[i],false);

    int correct=0;
    for (int i=0;i<N;i++){
      const int y=fValidationLabels[i]; const double pr=p[i];
      val_loss += - ( y*std::log(pr+1e-7) + (1-y)*std::log(1-pr+1e-7) );
      const int yhat = (pr>fPhotonThreshold)?1:0;
      if (yhat==y) correct++;
    }
    val_loss/=N;
    const double val_acc = 100.0*double(correct)/double(N);
    std::cout<<" Val: "<<val_loss<<" (Acc: "<<val_acc<<"%)";

    // sweep thresholds
    double bestScore=-1.0, bestThr=fPhotonThreshold;
    for (double thr=0.40; thr<=0.90+1e-9; thr+=0.02){
      int TP=0,FP=0,TN=0,FN=0;
      for (int i=0;i<N;i++){
        const int y=fValidationLabels[i];
        const int yhat=(p[i]>thr)?1:0;
        if (yhat==1 && y==1) TP++;
        else if (yhat==1 && y==0) FP++;
        else if (yhat==0 && y==0) TN++;
        else FN++;
      }
      const double prec = (TP+FP>0)? double(TP)/(TP+FP):0.0;
      const double rec  = (TP+FN>0)? double(TP)/(TP+FN):0.0;
      const double F1   = (prec+rec>0)? (2.0*prec*rec/(prec+rec)):0.0;
      const double tpr  = (TP+FN>0)? double(TP)/(TP+FN):0.0;
      const double fpr  = (FP+TN>0)? double(FP)/(FP+TN):0.0;
      const double J    = tpr - fpr;
      const double score= 0.75*F1 + 0.25*J;
      if (score>bestScore){ bestScore=score; bestThr=thr; }
    }
    const double old=fPhotonThreshold;
    fPhotonThreshold = 0.7*fPhotonThreshold + 0.3*bestThr;
    if (fLogFile.is_open()){
      fLogFile<<"ThresholdCalib: "<<old<<" -> "<<fPhotonThreshold
              <<"  (best="<<bestThr<<", mixScore="<<bestScore<<")\n";
      fLogFile.flush();
    }
    const int bin=fEventCount/5;
    if (hValidationLoss) hValidationLoss->SetBinContent(bin,val_loss);
  }

  std::cout<<std::endl;
  // keep memory bounded
  if (fTrainingFeatures.size()>1000){
    fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin()+200);
    fTrainingLabels.erase  (fTrainingLabels.begin(),   fTrainingLabels.begin()+200);
  }
  fTrainingStep++;
  return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
  const int total=fTruePositives+fTrueNegatives+fFalsePositives+fFalseNegatives;
  if (total==0) return;

  const double acc  = 100.0*(fTruePositives+fTrueNegatives)/double(total);
  const double prec = (fTruePositives+fFalsePositives>0)?
                      100.0*fTruePositives/double(fTruePositives+fFalsePositives):0.0;
  const double rec  = (fTruePositives+fFalseNegatives>0)?
                      100.0*fTruePositives/double(fTruePositives+fFalseNegatives):0.0;
  const double f1   = (prec+rec>0)? (2.0*prec*rec/(prec+rec)):0.0;
  const double frac = (fPhotonLikeCount+fHadronLikeCount>0)?
                      100.0*fPhotonLikeCount/double(fPhotonLikeCount+fHadronLikeCount):0.0;

  std::cout<<"│ "<<std::setw(6)<<fEventCount
           <<" │ "<<std::setw(8)<<fStationCount
           <<" │ "<<std::fixed<<std::setprecision(1)<<std::setw(7)<<frac<<"%"
           <<" │ "<<std::setw(7)<<acc<<"%"
           <<" │ "<<std::setw(8)<<prec<<"%"
           <<" │ "<<std::setw(7)<<rec<<"%"
           <<" │ "<<std::setw(7)<<f1<<"%│\n";

  if (fLogFile.is_open()){
    fLogFile<<"Event "<<fEventCount<<" - Acc: "<<acc<<"% Prec: "<<prec<<"% Rec: "<<rec<<"% F1: "<<f1<<"%\n";
    fLogFile.flush();
  }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
  std::cout<<"\n==========================================\n";
  std::cout<<"PERFORMANCE METRICS\n";
  std::cout<<"==========================================\n";

  const int total=fTruePositives+fTrueNegatives+fFalsePositives+fFalseNegatives;
  if (total==0){ std::cout<<"No predictions made yet!\n"; return; }

  const double acc  = 100.0*(fTruePositives+fTrueNegatives)/double(total);
  const double prec = (fTruePositives+fFalsePositives>0)?
                      100.0*fTruePositives/double(fTruePositives+fFalsePositives):0.0;
  const double rec  = (fTruePositives+fFalseNegatives>0)?
                      100.0*fTruePositives/double(fTruePositives+fFalseNegatives):0.0;
  const double f1   = (prec+rec>0)? (2.0*prec*rec/(prec+rec)):0.0;

  std::cout<<"Accuracy:  "<<std::fixed<<std::setprecision(1)<<acc<<"%\n";
  std::cout<<"Precision: "<<prec<<"%\n";
  std::cout<<"Recall:    "<<rec<<"%\n";
  std::cout<<"F1-Score:  "<<f1<<"%\n\n";

  std::cout<<"CONFUSION MATRIX:\n";
  std::cout<<"                Predicted\n";
  std::cout<<"             Hadron   Photon\n";
  std::cout<<"Actual Hadron  "<<std::setw(6)<<fTrueNegatives<<"   "<<std::setw(6)<<fFalsePositives<<"\n";
  std::cout<<"       Photon  "<<std::setw(6)<<fFalseNegatives<<"   "<<std::setw(6)<<fTruePositives<<"\n\n";

  std::cout<<"Total Stations: "<<fStationCount<<"\n";
  std::cout<<"Photon-like: "<<fPhotonLikeCount<<" ("<< ( (fPhotonLikeCount+fHadronLikeCount>0)?
            100.0*fPhotonLikeCount/double(fPhotonLikeCount+fHadronLikeCount):0.0 ) <<"%)\n";
  std::cout<<"Hadron-like: "<<fHadronLikeCount<<" ("<< ( (fPhotonLikeCount+fHadronLikeCount>0)?
            100.0*fHadronLikeCount/double(fPhotonLikeCount+fHadronLikeCount):0.0 ) <<"%)\n";

  std::cout<<"\nPARTICLE TYPE BREAKDOWN:\n";
  for (const auto& pr: fParticleTypeCounts) std::cout<<"  "<<pr.first<<": "<<pr.second<<" events\n";
  std::cout<<"==========================================\n";

  if (fLogFile.is_open()){
    fLogFile<<"\nFinal Performance Metrics:\n";
    fLogFile<<"Accuracy: "<<acc<<"%\n";
    fLogFile<<"Precision: "<<prec<<"%\n";
    fLogFile<<"Recall: "<<rec<<"%\n";
    fLogFile<<"F1-Score: "<<f1<<"%\n";
    fLogFile.flush();
  }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
  std::cout<<"\n==========================================\n";
  std::cout<<"PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n";
  std::cout<<"==========================================\n";
  std::cout<<"Events processed: "<<fEventCount<<"\n";
  std::cout<<"Stations analyzed: "<<fStationCount<<"\n";

  CalculatePerformanceMetrics();

  // persist weights
  fNeuralNetwork->SaveWeights(fWeightsFileName);

  // ROOT outputs
  if (fOutputFile){
    fOutputFile->cd();
    if (fMLTree) fMLTree->Write();
    if (hPhotonScore)          hPhotonScore->Write();
    if (hPhotonScorePhotons)   hPhotonScorePhotons->Write();
    if (hPhotonScoreHadrons)   hPhotonScoreHadrons->Write();
    if (hConfidence)           hConfidence->Write();
    if (hRisetime)             hRisetime->Write();
    if (hAsymmetry)            hAsymmetry->Write();
    if (hKurtosis)             hKurtosis->Write();
    if (hScoreVsEnergy)        hScoreVsEnergy->Write();
    if (hScoreVsDistance)      hScoreVsDistance->Write();
    if (hConfusionMatrix)      hConfusionMatrix->Write();
    if (hTrainingLoss)         hTrainingLoss->Write();
    if (hValidationLoss)       hValidationLoss->Write();
    if (hAccuracyHistory)      hAccuracyHistory->Write();
    fOutputFile->Close(); delete fOutputFile; fOutputFile=nullptr;
  }

  // small summary ROOT file
  {
    TFile fsum("photon_trigger_summary.root","RECREATE");
    if (!fsum.IsZombie()){
      if (hConfusionMatrix) hConfusionMatrix->Write("ConfusionMatrix");
      if (hPhotonScore)     hPhotonScore->Write("PhotonScore_All");
      if (hPhotonScorePhotons) hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
      if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
      if (hConfidence)      hConfidence->Write("Confidence");
      fsum.Close();
    }
  }
  // human-readable summary
  {
    std::ofstream txt("photon_trigger_summary.txt");
    if (txt.is_open()){
      const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
      const int tot=TP+FP+TN+FN;
      const double acc  = (tot>0)? 100.0*(TP+TN)/double(tot):0.0;
      const double prec = (TP+FP>0)? 100.0*TP/double(TP+FP):0.0;
      const double rec  = (TP+FN>0)? 100.0*TP/double(TP+FN):0.0;
      const double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec):0.0;
      txt<<"PhotonTriggerML Summary\n";
      txt<<"Threshold: "<<fPhotonThreshold<<"\n";
      txt<<"TP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN<<"\n";
      txt<<std::fixed<<std::setprecision(2);
      txt<<"Accuracy="<<acc<<"%  Precision="<<prec<<"%  Recall="<<rec<<"%  F1="<<f1<<"%\n";
      txt.close();
    }
  }
  // feature statistics (means & RMS for each class)
  {
    std::ofstream ft("photon_trigger_feature_stats.tsv");
    if (ft.is_open()){
      const char* names[17] = {
        "rise10_50","rise10_90","fall90_10","width","asymmetry","peak_amp",
        "charge","peak_charge_ratio","smoothness","kurtosis","skewness",
        "early_frac","late_frac","time_spread","hf_content","n_peaks","sec_peak_ratio"
      };
      ft<<"class\tfeature\tmean\trms\tN\n";
      auto dump=[&](const char* cls, const RunningStats& S){
        if (S.n<=0) return;
        for (int i=0;i<17;i++){
          const double mean = S.sum[i]/double(S.n);
          const double rms  = std::sqrt( std::max(0.0, S.sumsq[i]/double(S.n) - mean*mean) );
          ft<<cls<<"\t"<<names[i]<<"\t"<<mean<<"\t"<<rms<<"\t"<<S.n<<"\n";
        }
      };
      dump("photon", gPhotonStats);
      dump("hadron", gHadronStats);
      ft.close();
    }
  }

  // annotate existing PMT trace file in place (skip under signal)
  if (!gCalledFromSignal){
    auto strip_ml = [](const std::string& t)->std::string{
      size_t p=t.find(" [ML:"); return (p==std::string::npos)?t:t.substr(0,p);
    };
    auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool{
      if (!h) return false;
      const int n=h->GetNbinsX(); if (n<=0) return false;
      std::vector<double> tr(n); for (int i=1;i<=n;i++) tr[i-1]=h->GetBinContent(i);
      EnhancedFeatures F = ExtractEnhancedFeatures(tr);
      std::vector<double> X = NormalizeFeatures(F);
      if (X.size()>=17){ X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5; }
      double ml=fNeuralNetwork->Predict(X,false);
      double pen=0.0;
      if (F.pulse_width<120.0 && F.peak_charge_ratio>0.35) pen-=0.20;
      if (F.num_peaks<=2  && F.secondary_peak_ratio>0.60)  pen-=0.10;
      score = std::max(0.0,std::min(1.0, ml+pen));
      isPhoton = (score>fPhotonThreshold);
      return true;
    };
    auto annotate_dir = [&](TDirectory* dir, auto&& self)->void{
      if (!dir) return;
      TIter next(dir->GetListOfKeys()); TKey* key;
      while ((key=(TKey*)next())){
        TObject* obj = dir->Get(key->GetName()); if (!obj) continue;
        if (obj->InheritsFrom(TDirectory::Class())){
          self((TDirectory*)obj, self);
        } else if (obj->InheritsFrom(TH1::Class())){
          TH1* h=(TH1*)obj; double sc=0.0; bool ph=false;
          if (score_from_hist(h,sc,ph)){
            const std::string base=strip_ml(h->GetTitle()?h->GetTitle():"");
            std::ostringstream t; t<<base<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc
                                   <<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
            h->SetTitle(t.str().c_str()); dir->cd();
            h->Write(h->GetName(), TObject::kOverwrite);
          }
        } else if (obj->InheritsFrom(TCanvas::Class())){
          TCanvas* c=(TCanvas*)obj; TH1* h=nullptr;
          if (TList* prim=c->GetListOfPrimitives()){
            TIter nx(prim); while (TObject* po=nx()){ if (po->InheritsFrom(TH1::Class())){ h=(TH1*)po; break; } }
          }
          if (h){
            double sc=0.0; bool ph=false;
            if (score_from_hist(h,sc,ph)){
              const std::string baseC=strip_ml(c->GetTitle()?c->GetTitle():"");
              const std::string baseH=strip_ml(h->GetTitle()?h->GetTitle():"");
              std::ostringstream tc,th;
              tc<<baseC<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
              th<<baseH<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
              c->SetTitle(tc.str().c_str()); h->SetTitle(th.str().c_str());
              c->Modified(); c->Update(); dir->cd();
              c->Write(c->GetName(), TObject::kOverwrite);
            }
          }
        }
        delete obj;
      }
    };

    const char* cands[]={"pmt_traces_1EeV.root","pmt_Traces_1EeV.root","pmt_traces_1eev.root","pmt_Traces_1eev.root"};
    TFile* f=nullptr;
    for (const char* nm: cands){
      f=TFile::Open(nm,"UPDATE");
      if (f && !f->IsZombie()) break;
      if (f){ delete f; f=nullptr; }
    }
    if (f && !f->IsZombie()){
      if (!f->TestBit(TFile::kRecovered)){
        annotate_dir(f, annotate_dir);
        f->Write("", TObject::kOverwrite);
        std::cout<<"Annotated ML tags in "<<f->GetName()<<"\n";
      } else {
        std::cout<<"Recovered trace file detected; skip annotation.\n";
      }
      f->Close(); delete f;
    } else {
      std::cout<<"Trace file not found for ML annotation (skipping).\n";
    }
  }

  if (fLogFile.is_open()) fLogFile.close();
  std::cout<<"==========================================\n";
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
  INFO("PhotonTriggerML::Finish()");
  SaveAndDisplaySummary();
  return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& result)
{
  auto it=fMLResultsMap.find(stationId);
  if (it!=fMLResultsMap.end()){ result=it->second; return true; }
  return false;
}

void PhotonTriggerML::ClearMLResults(){ fMLResultsMap.clear(); }

