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

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
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

using namespace std;
using namespace fwk;
using namespace evt;
using namespace det;

// ============================================================================
// Globals for signal-safe summary
// ============================================================================
PhotonTriggerML* PhotonTriggerML::fInstance = 0;
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;
static volatile sig_atomic_t gCalledFromSignal = 0;

static void PhotonTriggerMLSignalHandler(int sig)
{
  if (sig==SIGINT || sig==SIGTSTP) {
    std::cout << "\n\n[PhotonTriggerML] Caught interrupt. Writing summary...\n";
    gCalledFromSignal = 1;
    if (PhotonTriggerML::fInstance) {
      try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); }
      catch (...) { /* swallow to guarantee termination */ }
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
    _Exit(0);
  }
}

// ============================================================================
// Tiny feed-forward NN (unchanged architecture, safer I/O formatting)
// ============================================================================
PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
  fInputSize(0), fHidden1Size(0), fHidden2Size(0),
  fTimeStep(0), fDropoutRate(0.10),
  fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int in, int h1, int h2)
{
  fInputSize = in; fHidden1Size = h1; fHidden2Size = h2;

  std::mt19937 gen(12345);
  std::uniform_real_distribution<> d(-0.5, 0.5);

  fWeights1.assign(h1, std::vector<double>(in, 0));
  for (int i=0;i<h1;++i) for (int j=0;j<in;++j) fWeights1[i][j] = d(gen)/std::sqrt(in);

  fWeights2.assign(h2, std::vector<double>(h1, 0));
  for (int i=0;i<h2;++i) for (int j=0;j<h1;++j) fWeights2[i][j] = d(gen)/std::sqrt(h1);

  fWeights3.assign(1, std::vector<double>(h2, 0));
  for (int j=0;j<h2;++j) fWeights3[0][j] = d(gen)/std::sqrt(h2);

  fBias1.assign(h1, 0.0);
  fBias2.assign(h2, 0.0);
  fBias3 = 0.0;

  fMomentum1_w1.assign(h1, std::vector<double>(in, 0));
  fMomentum1_w2.assign(h2, std::vector<double>(h1, 0));
  fMomentum1_w3.assign(1, std::vector<double>(h2, 0));
  fMomentum1_b1.assign(h1, 0.0);
  fMomentum1_b2.assign(h2, 0.0);
  fMomentum1_b3 = 0.0;

  fTimeStep = 0;
  std::cout << "[PhotonTriggerML] NN " << in << " -> " << h1 << " -> " << h2 << " -> 1\n";
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
  if ((int)x.size()!=fInputSize) return 0.5;

  std::vector<double> h1(fHidden1Size,0.0);
  for (int i=0;i<fHidden1Size;++i) {
    double s = fBias1[i];
    for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
    h1[i] = 1.0/(1.0+std::exp(-s));
    if (training && (rand()/double(RAND_MAX))<fDropoutRate) h1[i]=0.0;
  }

  std::vector<double> h2(fHidden2Size,0.0);
  for (int i=0;i<fHidden2Size;++i) {
    double s = fBias2[i];
    for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
    h2[i] = 1.0/(1.0+std::exp(-s));
    if (training && (rand()/double(RAND_MAX))<fDropoutRate) h2[i]=0.0;
  }

  double o = fBias3;
  for (int j=0;j<fHidden2Size;++j) o += fWeights3[0][j]*h2[j];
  return 1.0/(1.0+std::exp(-o));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y,
                                             double lr)
{
  if (X.empty() || X.size()!=y.size()) return -1.0;
  const int B = (int)X.size();

  int nP = std::count(y.begin(), y.end(), 1);
  const double wP = (nP>0)? 1.5 : 1.0;
  const double wH = 1.0;

  std::vector<std::vector<double>> gw1(fHidden1Size, std::vector<double>(fInputSize,0));
  std::vector<std::vector<double>> gw2(fHidden2Size, std::vector<double>(fHidden1Size,0));
  std::vector<std::vector<double>> gw3(1, std::vector<double>(fHidden2Size,0));
  std::vector<double> gb1(fHidden1Size,0), gb2(fHidden2Size,0);
  double gb3=0.0, total=0.0;

  for (int n=0;n<B;++n) {
    const auto& x = X[n];
    const int    t = y[n];
    const double w = (t==1? wP : wH);

    // forward
    std::vector<double> h1r(fHidden1Size), h2r(fHidden2Size), h1(fHidden1Size), h2(fHidden2Size);
    for (int i=0;i<fHidden1Size;++i) { double s=fBias1[i]; for (int j=0;j<fInputSize;++j) s+=fWeights1[i][j]*x[j]; h1r[i]=s; h1[i]=1.0/(1.0+std::exp(-s)); }
    for (int i=0;i<fHidden2Size;++i) { double s=fBias2[i]; for (int j=0;j<fHidden1Size;++j) s+=fWeights2[i][j]*h1[j]; h2r[i]=s; h2[i]=1.0/(1.0+std::exp(-s)); }
    double o=fBias3; for (int j=0;j<fHidden2Size;++j) o+=fWeights3[0][j]*h2[j];
    const double p = 1.0/(1.0+std::exp(-o));

    total += -w*( t*std::log(p+1e-7) + (1-t)*std::log(1-p+1e-7) );

    // backward
    const double go = w*(p - t);
    for (int j=0;j<fHidden2Size;++j) gw3[0][j] += go*h2[j];
    gb3 += go;

    std::vector<double> gh2(fHidden2Size,0);
    for (int j=0;j<fHidden2Size;++j) gh2[j] = fWeights3[0][j]*go * h2[j]*(1-h2[j]);
    for (int i=0;i<fHidden2Size;++i){ for (int j=0;j<fHidden1Size;++j) gw2[i][j]+= gh2[i]*h1[j]; gb2[i]+=gh2[i]; }

    std::vector<double> gh1(fHidden1Size,0);
    for (int j=0;j<fHidden1Size;++j){ double s=0; for (int i=0;i<fHidden2Size;++i) s+=fWeights2[i][j]*gh2[i]; gh1[j] = s * h1[j]*(1-h1[j]); }
    for (int i=0;i<fHidden1Size;++i){ for (int j=0;j<fInputSize;++j) gw1[i][j]+= gh1[i]*x[j]; gb1[i]+=gh1[i]; }
  }

  const double mom=0.9;
  for (int i=0;i<fHidden1Size;++i){
    for (int j=0;j<fInputSize;++j){ const double g = gw1[i][j]/B; fMomentum1_w1[i][j] = mom*fMomentum1_w1[i][j] - lr*g; fWeights1[i][j] += fMomentum1_w1[i][j]; }
    { const double g = gb1[i]/B; fMomentum1_b1[i] = mom*fMomentum1_b1[i] - lr*g; fBias1[i]+=fMomentum1_b1[i]; }
  }
  for (int i=0;i<fHidden2Size;++i){
    for (int j=0;j<fHidden1Size;++j){ const double g = gw2[i][j]/B; fMomentum1_w2[i][j] = mom*fMomentum1_w2[i][j] - lr*g; fWeights2[i][j] += fMomentum1_w2[i][j]; }
    { const double g = gb2[i]/B; fMomentum1_b2[i] = mom*fMomentum1_b2[i] - lr*g; fBias2[i]+=fMomentum1_b2[i]; }
  }
  for (int j=0;j<fHidden2Size;++j){ const double g = gw3[0][j]/B; fMomentum1_w3[0][j] = mom*fMomentum1_w3[0][j] - lr*g; fWeights3[0][j] += fMomentum1_w3[0][j]; }
  { const double g = gb3/B; fMomentum1_b3 = mom*fMomentum1_b3 - lr*g; fBias3 += fMomentum1_b3; }

  return total/B;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& fn)
{
  std::ofstream f(fn.c_str());
  if (!f.is_open()) { std::cout<<"[PhotonTriggerML] Cannot write "<<fn<<"\n"; return; }

  f << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";

  for (const auto& row: fWeights1) { for (double w: row) f<<w<<" "; f<<"\n"; }
  { for (double b: fBias1) f<<b<<" "; f<<"\n"; }

  for (const auto& row: fWeights2) { for (double w: row) f<<w<<" "; f<<"\n"; }
  { for (double b: fBias2) f<<b<<" "; f<<"\n"; }

  { for (double w: fWeights3[0]) f<<w<<" "; f<<"\n"; }
  f << fBias3 << "\n";
  f.close();
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& fn)
{
  std::ifstream f(fn.c_str());
  if (!f.is_open()) { std::cout<<"[PhotonTriggerML] No weight file "<<fn<<" (start fresh)\n"; return false; }

  f >> fInputSize >> fHidden1Size >> fHidden2Size;
  fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize,0));
  fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size,0));
  fWeights3.assign(1, std::vector<double>(fHidden2Size,0));
  fBias1.assign(fHidden1Size,0); fBias2.assign(fHidden2Size,0);

  for (auto& row: fWeights1) for (double& w: row) f>>w;
  for (double& b: fBias1) f>>b;
  for (auto& row: fWeights2) for (double& w: row) f>>w;
  for (double& b: fBias2) f>>b;
  for (double& w: fWeights3[0]) f>>w;
  f >> fBias3;
  f.close();
  std::cout << "[PhotonTriggerML] Loaded weights from " << fn << "\n";
  return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights() { fIsQuantized=true; }

// ============================================================================
// Extra diagnostics accumulators (no header change, file-scope statics)
// ============================================================================
namespace {
  struct Accum {
    double sum[17], sum2[17], mn[17], mx[17]; long long n;
    Accum(){ reset(); }
    void reset(){ n=0; for (int i=0;i<17;++i){ sum[i]=sum2[i]=0; mn[i]=+1e300; mx[i]=-1e300; } }
    void push(const std::vector<double>& v){
      n++; for (int i=0;i<17;++i){ const double x=v[i]; sum[i]+=x; sum2[i]+=x*x; if (x<mn[i]) mn[i]=x; if (x>mx[i]) mx[i]=x; }
    }
  };
  static Accum gPhoton, gHadron;
  static std::vector<double> gScorePhoton, gScoreHadron;
}

static inline std::vector<double> FeaturesRawVector(const PhotonTriggerML::EnhancedFeatures& F)
{
  std::vector<double> raw;
  raw.reserve(17);
  raw.push_back(F.risetime_10_50);
  raw.push_back(F.risetime_10_90);
  raw.push_back(F.falltime_90_10);
  raw.push_back(F.pulse_width);
  raw.push_back(F.asymmetry);
  raw.push_back(F.peak_amplitude);
  raw.push_back(F.total_charge);
  raw.push_back(F.peak_charge_ratio);
  raw.push_back(F.smoothness);
  raw.push_back(F.kurtosis);
  raw.push_back(F.skewness);
  raw.push_back(F.early_fraction);
  raw.push_back(F.late_fraction);
  raw.push_back(F.time_spread);
  raw.push_back(F.high_freq_content);
  raw.push_back((double)F.num_peaks);
  raw.push_back(F.secondary_peak_ratio);
  return raw;
}

// ============================================================================
// Module lifecycle
// ============================================================================
PhotonTriggerML::PhotonTriggerML() :
  fNeuralNetwork(new NeuralNetwork()),
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
  hScoreVsEnergy(nullptr), hScoreVsDistance(nullptr),
  gROCCurve(nullptr), hConfusionMatrix(nullptr),
  hTrainingLoss(nullptr), hValidationLoss(nullptr), hAccuracyHistory(nullptr),
  fTruePositives(0), fFalsePositives(0), fTrueNegatives(0), fFalseNegatives(0),
  fPhotonThreshold(0.65),
  fEnergyMin(1e18), fEnergyMax(1e19),
  fOutputFileName("photon_trigger_ml_physics.root"),
  fWeightsFileName("photon_trigger_weights_physics.txt"),
  fLoadPretrainedWeights(true)
{
  fInstance = this;
  fFeatureMeans.assign(17, 0.0);
  fFeatureStdDevs.assign(17, 1.0);
}

PhotonTriggerML::~PhotonTriggerML(){}

VModule::ResultFlag PhotonTriggerML::Init()
{
  INFO("PhotonTriggerML::Init()");

  // log
  fLogFile.open(fLogFileName.c_str());
  if (!fLogFile.is_open()) { ERROR("Cannot open log file"); return eFailure; }
  time_t now = time(0);
  fLogFile << "==========================================\n";
  fLogFile << "PhotonTriggerML Physics-Based Version Log\n";
  fLogFile << "Date: " << ctime(&now);
  fLogFile << "==========================================\n\n";
  fLogFile.flush();

  // NN
  fNeuralNetwork->Initialize(17, 8, 4);
  if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
    fIsTraining = false;
  }

  // ROOT out
  fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
  if (!fOutputFile || fOutputFile->IsZombie()) { ERROR("Cannot create ROOT output"); return eFailure; }

  fMLTree = new TTree("MLTree","PhotonTriggerML");
  fMLTree->Branch("eventId",&fEventCount,"eventId/I");
  fMLTree->Branch("stationId",&fStationId,"stationId/I");
  fMLTree->Branch("energy",&fEnergy,"energy/D");
  fMLTree->Branch("distance",&fDistance,"distance/D");
  fMLTree->Branch("photonScore",&fPhotonScore,"photonScore/D");
  fMLTree->Branch("confidence",&fConfidence,"confidence/D");
  fMLTree->Branch("primaryId",&fPrimaryId,"primaryId/I");
  fMLTree->Branch("primaryType",&fPrimaryType);
  fMLTree->Branch("isActualPhoton",&fIsActualPhoton,"isActualPhoton/O");

  // histograms
  hPhotonScore = new TH1D("hPhotonScore","Score(all);score;count",50,0,1);
  hPhotonScorePhotons = new TH1D("hPhotonScorePhotons","Score(true photons);score;count",50,0,1);
  hPhotonScorePhotons->SetLineColor(kBlue);
  hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons","Score(true hadrons);score;count",50,0,1);
  hPhotonScoreHadrons->SetLineColor(kRed);
  hConfidence = new TH1D("hConfidence","|score-0.5|;conf;count",50,0,0.5);
  hRisetime   = new TH1D("hRise1090","Rise(10-90%)[ns];ns;count",60,0,1500);
  hAsymmetry  = new TH1D("hAsymm","(fall-rise)/(fall+rise);val;count",60,-1,1);
  hKurtosis   = new TH1D("hKurt","kurtosis;val;count",60,-5,20);
  hScoreVsEnergy   = new TH2D("hScoreVsE","score vs E;E[eV];score",50,1e17,1e20,50,0,1);
  hScoreVsDistance = new TH2D("hScoreVsR","score vs R;R[m];score",50,0,3000,50,0,1);

  hConfusionMatrix = new TH2D("hCM","Confusion;Predicted;Actual",2,-0.5,1.5,2,-0.5,1.5);
  hConfusionMatrix->GetXaxis()->SetBinLabel(1,"Hadron");
  hConfusionMatrix->GetXaxis()->SetBinLabel(2,"Photon");
  hConfusionMatrix->GetYaxis()->SetBinLabel(1,"Hadron");
  hConfusionMatrix->GetYaxis()->SetBinLabel(2,"Photon");

  hTrainingLoss   = new TH1D("hTrainLoss","Train loss;step;loss",10000,0,10000);
  hValidationLoss = new TH1D("hValLoss","Val loss;step;loss",10000,0,10000);
  hAccuracyHistory= new TH1D("hAcc","Accuracy[%];step;acc",10000,0,10000);

  std::signal(SIGINT,  PhotonTriggerMLSignalHandler);
  std::signal(SIGTSTP, PhotonTriggerMLSignalHandler);

  // diagnostics buffers
  gPhoton.reset(); gHadron.reset();
  gScorePhoton.clear(); gScoreHadron.clear();

  return eSuccess;
}

// ============================================================================
// Event processing
// ============================================================================
VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
  fEventCount++;
  ClearMLResults();

  // shower info
  fEnergy=0; fCoreX=0; fCoreY=0; fPrimaryId=0; fPrimaryType="Unknown"; fIsActualPhoton=false;

  if (event.HasSimShower()) {
    const ShowerSimData& sh = event.GetSimShower();
    fEnergy = sh.GetEnergy();
    fPrimaryId = sh.GetPrimaryParticle();
    switch (fPrimaryId) {
      case 22:            fPrimaryType="photon"; fIsActualPhoton=true; break;
      case 2212:          fPrimaryType="proton"; break;
      case 11: case -11:  fPrimaryType="electron"; break;
      case 1000026056:    fPrimaryType="iron"; break;
      default:            fPrimaryType = (fPrimaryId>1000000000) ? "nucleus" : "unknown";
    }
    if (sh.GetNSimCores()>0) {
      const Detector& det = Detector::GetInstance();
      const utl::CoordinateSystemPtr& site = det.GetSiteCoordinateSystem();
      utl::Point core = sh.GetSimCore(0);
      fCoreX = core.GetX(site); fCoreY = core.GetY(site);
    }
    if (fEventCount<=5) {
      std::cout<<"Event "<<fEventCount<<": Energy="<<(fEnergy/1e18)<<" EeV, Primary="<<fPrimaryType
               <<" (ID="<<fPrimaryId<<")\n";
    }
  }

  // stations
  if (event.HasSEvent()) {
    const sevt::SEvent& se = event.GetSEvent();
    for (sevt::SEvent::ConstStationIterator it=se.StationsBegin(); it!=se.StationsEnd(); ++it) {
      ProcessStation(*it);
    }
  }

  // periodic metrics & confusion
  if (fEventCount%10==0) {
    CalculateAndDisplayMetrics();
    hConfusionMatrix->SetBinContent(1,1, fTrueNegatives);
    hConfusionMatrix->SetBinContent(2,1, fFalsePositives);
    hConfusionMatrix->SetBinContent(1,2, fFalseNegatives);
    hConfusionMatrix->SetBinContent(2,2, fTruePositives);
  }

  // on-the-fly training with validation threshold calibration
  if (fIsTraining && fTrainingFeatures.size()>=20 && fEventCount%5==0) {
    const double vloss = TrainNetwork();
    const int step = fEventCount/5;
    hTrainingLoss->SetBinContent(step, vloss);

    const int correct = fTruePositives + fTrueNegatives;
    const int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total>0) hAccuracyHistory->SetBinContent(step, 100.0*correct/total);

    if (vloss < fBestValidationLoss) {
      fBestValidationLoss = vloss; fEpochsSinceImprovement=0;
      fNeuralNetwork->SaveWeights(std::string("best_")+fWeightsFileName);
    } else {
      fEpochsSinceImprovement++;
    }
  }

  return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& st)
{
  fStationId = st.GetId();

  // distance to core
  try {
    const Detector& det = Detector::GetInstance();
    const sdet::SDetector& sdetector = det.GetSDetector();
    const sdet::Station& dst = sdetector.GetStation(fStationId);
    const utl::CoordinateSystemPtr& site = det.GetSiteCoordinateSystem();
    const double sx = dst.GetPosition().GetX(site);
    const double sy = dst.GetPosition().GetY(site);
    fDistance = std::sqrt((sx-fCoreX)*(sx-fCoreX) + (sy-fCoreY)*(sy-fCoreY));
  } catch (...) { fDistance=-1.0; return; }

  const int firstPMT = sdet::Station::GetFirstPMTId();
  for (int p=0;p<3;++p) {
    const int pid = firstPMT + p;
    if (!st.HasPMT(pid)) continue;
    const sevt::PMT& pmt = st.GetPMT(pid);

    std::vector<double> trace;
    if (!ExtractTraceData(pmt, trace)) continue;
    if ((int)trace.size()!=2048) continue;

    const auto mm = std::minmax_element(trace.begin(), trace.end());
    if ((*mm.second - *mm.first) < 10.0) continue;

    fFeatures = ExtractEnhancedFeatures(trace);

    // bookkeeping hists & stats
    hRisetime->Fill(fFeatures.risetime_10_90);
    hAsymmetry->Fill(fFeatures.asymmetry);
    hKurtosis->Fill(fFeatures.kurtosis);
    UpdateFeatureStatistics(fFeatures);

    // normalize + physic emphasis
    std::vector<double> x = NormalizeFeatures(fFeatures);
    if (x.size()>=17){ x[1]*=2.0; x[3]*=2.0; x[7]*=1.5; x[16]*=1.5; }

    double score = fNeuralNetwork->Predict(x, false);

    // milder spike penalty
    double penalty = 0.0;
    if (fFeatures.pulse_width < 120.0 && fFeatures.peak_charge_ratio>0.35) penalty -= 0.10;
    if (fFeatures.num_peaks<=2 && fFeatures.secondary_peak_ratio>0.60)    penalty -= 0.05;

    fPhotonScore = std::max(0.0, std::min(1.0, score+penalty));
    fConfidence  = std::fabs(fPhotonScore - 0.5);

    // training buffers
    const bool isValidation = (fStationCount%10==0);
    if (fIsTraining) {
      if (!isValidation) {
        if (fIsActualPhoton) {
          static std::mt19937 g(1234567u);
          std::normal_distribution<> n(0,0.02);
          for (int k=0;k<2;++k){ std::vector<double> v=x; for (double& z: v) z+=n(g); fTrainingFeatures.push_back(v); fTrainingLabels.push_back(1); }
        } else { fTrainingFeatures.push_back(x); fTrainingLabels.push_back(0); }
      } else {
        fValidationFeatures.push_back(x);
        fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
      }
    }

    fStationCount++;
    const bool predPhoton = (fPhotonScore > fPhotonThreshold);

    // per-station result
    MLResult r; r.photonScore=fPhotonScore; r.identifiedAsPhoton=predPhoton;
    r.isActualPhoton=fIsActualPhoton; r.vemCharge=fFeatures.total_charge;
    r.features=fFeatures; r.primaryType=fPrimaryType; r.confidence=fConfidence;
    fMLResultsMap[fStationId]=r;

    // confusion counters
    if (predPhoton) {
      fPhotonLikeCount++;
      if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
    } else {
      fHadronLikeCount++;
      if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
    }

    // hists
    hPhotonScore->Fill(fPhotonScore);
    if (fIsActualPhoton) { hPhotonScorePhotons->Fill(fPhotonScore); gScorePhoton.push_back(fPhotonScore); }
    else                { hPhotonScoreHadrons->Fill(fPhotonScore); gScoreHadron.push_back(fPhotonScore); }
    hConfidence->Fill(fConfidence);
    hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
    hScoreVsDistance->Fill(fDistance, fPhotonScore);
    fMLTree->Fill();

    // diagnostics TSV (scores)
    {
      std::ofstream tsv("photon_trigger_scores.tsv", std::ios::app);
      if (tsv.is_open()) {
        tsv<<fEventCount<<"\t"<<fStationId<<"\t"<<(fIsActualPhoton?1:0)<<"\t"
           <<(predPhoton?1:0)<<"\t"<<std::fixed<<std::setprecision(6)<<fPhotonScore
           <<"\t"<<fFeatures.risetime_10_90<<"\t"<<fFeatures.pulse_width
           <<"\t"<<fFeatures.peak_charge_ratio<<"\t"<<fFeatures.secondary_peak_ratio
           <<"\t"<<fEnergy<<"\t"<<fDistance<<"\n";
      }
    }

    // diagnostics TSV (misclassified only)
    if ((predPhoton && !fIsActualPhoton) || (!predPhoton && fIsActualPhoton)) {
      std::ofstream m("photon_trigger_misclassified.tsv", std::ios::app);
      if (m.is_open()) {
        m<<fEventCount<<"\t"<<fStationId<<"\t"<<(fIsActualPhoton?1:0)<<"\t"
         <<(predPhoton?1:0)<<"\t"<<std::fixed<<std::setprecision(6)<<fPhotonScore
         <<"\t"<<fFeatures.risetime_10_90<<"\t"<<fFeatures.pulse_width
         <<"\t"<<fFeatures.peak_charge_ratio<<"\t"<<fFeatures.secondary_peak_ratio
         <<"\t"<<fEnergy<<"\t"<<fDistance<<"\n";
      }
    }

    // per-class parameter accumulators (actual labels)
    {
      const std::vector<double> raw = FeaturesRawVector(fFeatures);
      if (fIsActualPhoton) gPhoton.push(raw); else gHadron.push(raw);
    }
  } // PMT loop
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& out)
{
  out.clear();

  // Prefer real FADC trace
  if (pmt.HasFADCTrace()) {
    try {
      const auto& tr = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
      for (int i=0;i<2048;++i) out.push_back(tr[i]);
      return true;
    } catch (...) {}
  }

  // Fall back to simulation store
  if (pmt.HasSimData()) {
    try {
      const sevt::PMTSimData& sim = pmt.GetSimData();
      if (sim.HasFADCTrace(sevt::StationConstants::eTotal)) {
        const auto& tr = sim.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                          sevt::StationConstants::eTotal);
        for (int i=0;i<2048;++i) out.push_back(tr[i]);
        return true;
      }
    } catch (...) {}
  }

  return false;
}

// ============================================================================
// Feature extraction and normalization
// ============================================================================
PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
  EnhancedFeatures f; // zeros by default
  const int N = (int)trace.size();
  if (N<=0) return f;

  const double ADC_PER_VEM = 180.0;
  const double NS_PER_BIN  = 25.0;

  // baseline from first 100 bins (guard N)
  const int B = std::min(100, N);
  double base=0; for (int i=0;i<B;++i) base += trace[i]; base/=std::max(1,B);

  std::vector<double> sig(N,0.0);
  int pk=0; double pv=0, sum=0;
  for (int i=0;i<N;++i){
    double s = trace[i]-base; if (s<0) s=0;
    sig[i]=s; sum+=s;
    if (s>pv){ pv=s; pk=i; }
  }
  if (pv<5.0 || sum<10.0) return f;

  f.peak_amplitude   = pv/ADC_PER_VEM;
  f.total_charge     = sum/ADC_PER_VEM;
  f.peak_charge_ratio= f.peak_amplitude / (f.total_charge+1e-3);

  const double t10 = 0.10*pv, t50=0.50*pv, t90=0.90*pv;
  int r10=0, r50=0, r90=pk;
  for (int i=pk;i>=0;--i){
    if (sig[i]<=t90 && r90==pk) r90=i;
    if (sig[i]<=t50 && r50==0)  r50=i;
    if (sig[i]<=t10){ r10=i; break; }
  }
  int f90=pk, f10=N-1;
  for (int i=pk;i<N;++i){
    if (sig[i]<=t90 && f90==pk) f90=i;
    if (sig[i]<=t10){ f10=i; break; }
  }

  f.risetime_10_50  = std::abs(r50-r10)*NS_PER_BIN;
  f.risetime_10_90  = std::abs(r90-r10)*NS_PER_BIN;
  f.falltime_90_10  = std::abs(f10-f90)*NS_PER_BIN;

  // FWHM
  const double half = 0.5*pv;
  int hL=r10, hR=f10;
  for (int i=r10;i<=pk;++i){ if (sig[i]>=half){ hL=i; break; } }
  for (int i=pk;i<N;++i){ if (sig[i]<=half){ hR=i; break; } }
  f.pulse_width = std::abs(hR-hL)*NS_PER_BIN;

  const double rise=f.risetime_10_90, fall=f.falltime_90_10;
  f.asymmetry = (fall - rise) / (fall + rise + 1e-3);

  // moments
  double m=0; for (int i=0;i<N;++i) m += i*sig[i]; m/= (sum+1e-3);
  double var=0, sk=0, ku=0;
  for (int i=0;i<N;++i){
    const double d=i-m, w=sig[i]/(sum+1e-3);
    var+=d*d*w; sk+=d*d*d*w; ku+=d*d*d*d*w;
  }
  const double sd = std::sqrt(var+1e-3);
  f.time_spread = sd*NS_PER_BIN;
  f.skewness = sk/std::max(1e-3, sd*sd*sd);
  f.kurtosis = ku/std::max(1e-3, var*var) - 3.0;

  // early/late
  const int Q = N/4;
  double early=0, late=0;
  for (int i=0;i<Q;++i) early+=sig[i];
  for (int i=3*Q;i<N;++i) late+=sig[i];
  f.early_fraction = early/(sum+1e-3);
  f.late_fraction  = late /(sum+1e-3);

  // smoothness (2nd deriv RMS on >10% peak)
  double acc=0; int cnt=0;
  for (int i=1;i<N-1;++i){
    if (sig[i] > 0.1*pv){
      const double s2 = sig[i+1] - 2*sig[i] + sig[i-1];
      acc += s2*s2; cnt++;
    }
  }
  f.smoothness = std::sqrt( acc/std::max(1,cnt) );

  // high-frequency content
  double hfacc=0;
  for (int i=1;i<N-1;++i){ const double d = sig[i+1]-sig[i-1]; hfacc += d*d; }
  f.high_freq_content = hfacc / std::max(1e-3, sum*sum);

  // peaks (stricter threshold to reduce fake multi-peak)
  f.num_peaks=0; double sec=0;
  const double pth = 0.25*pv;
  for (int i=1;i<N-1;++i){
    if (sig[i]>pth && sig[i]>sig[i-1] && sig[i]>sig[i+1]){
      f.num_peaks++;
      if (i!=pk && sig[i]>sec) sec=sig[i];
    }
  }
  f.secondary_peak_ratio = sec/(pv+1e-3);
  return f;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& F)
{
  std::vector<double> raw = {
    F.risetime_10_50, F.risetime_10_90, F.falltime_90_10, F.pulse_width, F.asymmetry,
    F.peak_amplitude, F.total_charge, F.peak_charge_ratio, F.smoothness, F.kurtosis,
    F.skewness, F.early_fraction, F.late_fraction, F.time_spread, F.high_freq_content,
    double(F.num_peaks), F.secondary_peak_ratio
  };

  static int n=0; ++n;
  for (size_t i=0;i<raw.size();++i){
    double delta = raw[i]-fFeatureMeans[i];
    fFeatureMeans[i] += delta/n;
    double delta2 = raw[i]-fFeatureMeans[i];
    fFeatureStdDevs[i] += delta*delta2;
  }
  if (n>1) {
    for (size_t i=0;i<raw.size();++i){
      fFeatureStdDevs[i] = std::sqrt( fFeatureStdDevs[i]/(n-1) );
      if (fFeatureStdDevs[i]<1e-3) fFeatureStdDevs[i]=1.0;
    }
  }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& F)
{
  std::vector<double> raw = {
    F.risetime_10_50, F.risetime_10_90, F.falltime_90_10, F.pulse_width, F.asymmetry,
    F.peak_amplitude, F.total_charge, F.peak_charge_ratio, F.smoothness, F.kurtosis,
    F.skewness, F.early_fraction, F.late_fraction, F.time_spread, F.high_freq_content,
    double(F.num_peaks), F.secondary_peak_ratio
  };
  std::vector<double> mn = {0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
  std::vector<double> mx = {500,1000,1000,1000,1,10,100,1,100,20,5,1,1,1000,10,10,1};

  std::vector<double> z; z.reserve(raw.size());
  for (size_t i=0;i<raw.size();++i){
    double v = (raw[i]-mn[i])/(mx[i]-mn[i]+1e-3);
    v = std::max(0.0, std::min(1.0, v));   // clamp to [0,1] (avoids misleading-indentation warnings)
    z.push_back(v);
  }
  return z;
}

// ============================================================================
// Training + validation with robust threshold calibration
// ============================================================================
double PhotonTriggerML::TrainNetwork()
{
  if (fTrainingFeatures.empty()) return 1e9;
  std::cout<<"\n  Training with "<<fTrainingFeatures.size()<<" samples...";

  // batch: cap for speed
  const int M = std::min<int>(100, fTrainingFeatures.size());
  std::vector<std::vector<double>> Bx; Bx.reserve(M);
  std::vector<int>                By; By.reserve(M);
  for (int i=0;i<M;++i){ Bx.push_back(fTrainingFeatures[i]); By.push_back(fTrainingLabels[i]); }

  const int P = std::count(By.begin(), By.end(), 1);
  const int H = (int)By.size() - P;
  std::cout<<" (P:"<<P<<" H:"<<H<<")";

  const double lr = 0.01;
  double total=0;
  for (int e=0;e<10;++e) total += fNeuralNetwork->Train(Bx, By, lr);
  const double train_loss = total/10.0;
  std::cout<<" Loss: "<<std::fixed<<std::setprecision(4)<<train_loss;

  // validation + threshold sweep
  double vloss = 0.0;
  if (!fValidationFeatures.empty()) {
    const int N = (int)fValidationFeatures.size();
    std::vector<double> preds(N,0.0);
    for (int i=0;i<N;++i) preds[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

    int correct=0;
    for (int i=0;i<N;++i){
      const int y = fValidationLabels[i];
      const double p = preds[i];
      vloss -= y*std::log(p+1e-7) + (1-y)*std::log(1-p+1e-7);
      const int yhat = (p>fPhotonThreshold)?1:0;
      if (yhat==y) correct++;
    }
    vloss/=N;
    const double vacc = 100.0*correct/N;
    std::cout<<" Val: "<<vloss<<" (Acc: "<<vacc<<"%)";

    // Primary sweep: F1 in [0.35,0.85] to avoid extremes
    double bestF1=-1.0, bestThr=fPhotonThreshold;
    int    bestTP=0;
    auto sweepEval = [&](double thr, int& TP, int& FP, int& TN, int& FN){
      TP=FP=TN=FN=0;
      for (int i=0;i<N;++i){
        const int y = fValidationLabels[i];
        const int yhat = (preds[i]>thr)?1:0;
        if (yhat==1 && y==1) TP++; else if (yhat==1 && y==0) FP++;
        else if (yhat==0 && y==0) TN++; else FN++;
      }
      const double prec = (TP+FP>0)? double(TP)/(TP+FP) : 0.0;
      const double rec  = (TP+FN>0)? double(TP)/(TP+FN) : 0.0;
      const double F1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;
      return F1;
    };

    double bestPrec=0.0, bestPrecThr=fPhotonThreshold;
    for (double thr=0.35; thr<=0.85+1e-9; thr+=0.01){
      int TP,FP,TN,FN; const double F1 = sweepEval(thr,TP,FP,TN,FN);
      if (F1>bestF1){ bestF1=F1; bestThr=thr; bestTP=TP; }
      const double prec = (TP+FP>0)? double(TP)/(TP+FP) : 0.0;
      if (TP>0 && prec>bestPrec){ bestPrec=prec; bestPrecThr=thr; }
    }

    // Fallbacks if sweep yields no TP at any threshold
    bool usedFallback = false;
    if (bestTP==0) {
      double mx = 0.0; for (double p: preds) if (p>mx) mx=p;
      double thrFallback = std::min(0.85, std::max(0.35, mx-1e-4));
      int TP,FP,TN,FN; (void)sweepEval(thrFallback,TP,FP,TN,FN);
      if (TP>0) { bestThr=thrFallback; bestTP=TP; usedFallback=true; }
      else if (bestPrec>0.0) { bestThr=bestPrecThr; usedFallback=true; }
    }

    const double old = fPhotonThreshold;
    fPhotonThreshold = 0.8*fPhotonThreshold + 0.2*bestThr; // gentle move

    if (fLogFile.is_open()) {
      fLogFile << "[Calibrator] N="<<N<<"  bestF1="<<bestF1<<" at thr="<<bestThr
               << " | usedFallback="<<(usedFallback?"yes":"no")
               << " | updated Thr: "<<old<<" -> "<<fPhotonThreshold<<"\n";
      fLogFile.flush();
    }

    const int step = fEventCount/5;
    hValidationLoss->SetBinContent(step, vloss);
  }

  std::cout<<"\n";

  // FIFO training window
  if (fTrainingFeatures.size()>1000){
    fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin()+200);
    fTrainingLabels.erase(fTrainingLabels.begin(), fTrainingLabels.begin()+200);
  }
  fTrainingStep++;
  return vloss;
}

// ============================================================================
// Metrics, summary, and rich diagnostics dumps
// ============================================================================
void PhotonTriggerML::CalculateAndDisplayMetrics()
{
  const int tot = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
  if (tot==0) return;

  const double acc = 100.0*(fTruePositives+fTrueNegatives)/tot;
  const double prec= (fTruePositives+fFalsePositives>0)? 100.0*fTruePositives/(fTruePositives+fFalsePositives) : 0.0;
  const double rec = (fTruePositives+fFalseNegatives>0)? 100.0*fTruePositives/(fTruePositives+fFalseNegatives) : 0.0;
  const double f1  = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;

  const double frac = (fPhotonLikeCount+fHadronLikeCount>0)?
                      100.0*fPhotonLikeCount/(fPhotonLikeCount+fHadronLikeCount) : 0.0;

  std::cout << "│ " << std::setw(6) << fEventCount
            << " │ " << std::setw(8) << fStationCount
            << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << frac << "%"
            << " │ " << std::setw(7) << acc << "%"
            << " │ " << std::setw(8) << prec << "%"
            << " │ " << std::setw(7) << rec << "%"
            << " │ " << std::setw(7) << f1 << "%│" << std::endl;

  if (fLogFile.is_open()) {
    fLogFile << "Event " << fEventCount
             << " - Acc: " << acc
             << "% Prec: " << prec
             << "% Rec: " << rec
             << "% F1: " << f1 << "%\n";
    fLogFile.flush();
  }
}

static inline double quantile(std::vector<double> v, double q)
{
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(v.begin(), v.end());
  double pos = q*(v.size()-1);
  size_t i = (size_t)std::floor(pos);
  size_t j = std::min(i+1, v.size()-1);
  double frac = pos - i;
  return (1.0-frac)*v[i] + frac*v[j];
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
  std::cout << "\n==========================================\n";
  std::cout << "PERFORMANCE METRICS\n";
  std::cout << "==========================================\n";

  const int tot = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
  if (tot==0) { std::cout<<"No predictions made yet!\n"; return; }

  const double acc = 100.0*(fTruePositives+fTrueNegatives)/tot;
  const double prec= (fTruePositives+fFalsePositives>0)? 100.0*fTruePositives/(fTruePositives+fFalsePositives) : 0.0;
  const double rec = (fTruePositives+fFalseNegatives>0)? 100.0*fTruePositives/(fTruePositives+fFalseNegatives) : 0.0;
  const double f1  = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;

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
  std::cout<<"Photon-like: "<<fPhotonLikeCount<<" ("<< 100.0*fPhotonLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount) <<"%)\n";
  std::cout<<"Hadron-like: "<<fHadronLikeCount<<" ("<< 100.0*fHadronLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount) <<"%)\n";

  if (fLogFile.is_open()) {
    fLogFile << "\nFinal Performance Metrics:\n";
    fLogFile << "Accuracy: " << acc << "%\n";
    fLogFile << "Precision: " << prec << "%\n";
    fLogFile << "Recall: " << rec << "%\n";
    fLogFile << "F1-Score: " << f1 << "%\n";
    fLogFile.flush();
  }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
  std::cout << "\n==========================================\n";
  std::cout << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n";
  std::cout << "==========================================\n";
  std::cout << "Events processed: " << fEventCount << "\n";
  std::cout << "Stations analyzed: " << fStationCount << "\n";

  CalculatePerformanceMetrics();

  // persist weights
  fNeuralNetwork->SaveWeights(fWeightsFileName);

  // ROOT output
  if (fOutputFile) {
    fOutputFile->cd();
    if (fMLTree) fMLTree->Write();
    if (hPhotonScore) hPhotonScore->Write();
    if (hPhotonScorePhotons) hPhotonScorePhotons->Write();
    if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write();
    if (hConfidence) hConfidence->Write();
    if (hRisetime) hRisetime->Write();
    if (hAsymmetry) hAsymmetry->Write();
    if (hKurtosis) hKurtosis->Write();
    if (hScoreVsEnergy) hScoreVsEnergy->Write();
    if (hScoreVsDistance) hScoreVsDistance->Write();
    if (hConfusionMatrix) hConfusionMatrix->Write();
    if (hTrainingLoss) hTrainingLoss->Write();
    if (hValidationLoss) hValidationLoss->Write();
    if (hAccuracyHistory) hAccuracyHistory->Write();
    fOutputFile->Close();
    delete fOutputFile; fOutputFile=nullptr;
  }

  // small summary files
  {
    TFile s("photon_trigger_summary.root","RECREATE");
    if (!s.IsZombie()) {
      if (hConfusionMatrix) hConfusionMatrix->Write("ConfusionMatrix");
      if (hPhotonScore)     hPhotonScore->Write("PhotonScore_All");
      if (hPhotonScorePhotons) hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
      if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
      if (hConfidence)      hConfidence->Write("Confidence");
      s.Close();
    }
  }
  {
    std::ofstream txt("photon_trigger_summary.txt");
    if (txt.is_open()) {
      const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
      const int tot=TP+FP+TN+FN;
      const double acc  = (tot>0)? 100.0*(TP+TN)/tot : 0.0;
      const double prec = (TP+FP>0)? 100.0*TP/(TP+FP) : 0.0;
      const double rec  = (TP+FN>0)? 100.0*TP/(TP+FN) : 0.0;
      const double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;
      txt<<"PhotonTriggerML Summary\n";
      txt<<"Threshold: "<<fPhotonThreshold<<"\n";
      txt<<"TP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN<<"\n";
      txt<<std::fixed<<std::setprecision(2);
      txt<<"Accuracy="<<acc<<"%  Precision="<<prec<<"%  Recall="<<rec<<"%  F1="<<f1<<"%\n";
      txt.close();
    }
  }

  // NEW: per-feature stats by actual class
  {
    const char* names[17] = {"rise10_50","rise10_90","fall90_10","fwhm","asym",
                             "peak_amp","total_q","peak_q_ratio","smooth","kurt",
                             "skew","early_frac","late_frac","time_spread","hf_content",
                             "num_peaks","sec_peak_ratio"};

    std::ofstream fs("photon_trigger_feature_stats.tsv");
    if (fs.is_open()) {
      fs<<"feature\tphoton_mean\tphoton_std\thadron_mean\thadron_std\tdelta\tcohens_d\n";
      for (int i=0;i<17;++i){
        const double nP = (double)std::max<long long>(1, gPhoton.n);
        const double nH = (double)std::max<long long>(1, gHadron.n);
        const double muP = gPhoton.sum[i]/nP;
        const double muH = gHadron.sum[i]/nH;
        const double vP  = std::max(0.0, gPhoton.sum2[i]/nP - muP*muP);
        const double vH  = std::max(0.0, gHadron.sum2[i]/nH - muH*muH);
        const double sP  = std::sqrt(vP);
        const double sH  = std::sqrt(vH);
        const double delta = muP - muH;
        const double sp    = std::sqrt( ((nP-1)*vP + (nH-1)*vH) / std::max(1.0, (nP+nH-2)) );
        const double d     = (sp>0? delta/sp : 0.0);
        fs<<names[i]<<"\t"<<muP<<"\t"<<sP<<"\t"<<muH<<"\t"<<sH<<"\t"<<delta<<"\t"<<d<<"\n";
      }
      fs.close();
    }

    // quick console highlight: top 5 discriminators by |d|
    struct Item { std::string name; double d; };
    std::vector<Item> items;
    for (int i=0;i<17;++i){
      const double nP = (double)std::max<long long>(1, gPhoton.n);
      const double nH = (double)std::max<long long>(1, gHadron.n);
      const double muP = gPhoton.sum[i]/nP;
      const double muH = gHadron.sum[i]/nH;
      const double vP  = std::max(0.0, gPhoton.sum2[i]/nP - muP*muP);
      const double vH  = std::max(0.0, gHadron.sum2[i]/nH - muH*muH);
      const double sp  = std::sqrt( ((nP-1)*vP + (nH-1)*vH) / std::max(1.0, (nP+nH-2)) );
      double d = 0.0; if (sp>0) d=(muP-muH)/sp;
      items.push_back({names[i], std::fabs(d)});
    }
    std::sort(items.begin(), items.end(), [](const Item&a,const Item&b){return a.d>b.d;});
    std::cout << "Top 5 parameter discriminators by |Cohen d|:\n";
    for (size_t k=0;k<std::min<size_t>(5,items.size());++k) {
      std::cout << "  - " << items[k].name << "  |d|=" << std::setprecision(3) << items[k].d << "\n";
    }
  }

  // NEW: score quantiles by class
  {
    std::ofstream qf("photon_trigger_score_quantiles.txt");
    if (qf.is_open()) {
      auto dump = [&](const std::vector<double>& v, const std::string& name){
        qf<<name<<": N="<<v.size()<<"\n";
        if (!v.empty()) {
          qf<<"  q10="<<quantile(v,0.10)<<"  q25="<<quantile(v,0.25)
            <<"  q50="<<quantile(v,0.50)<<"  q75="<<quantile(v,0.75)
            <<"  q90="<<quantile(v,0.90)<<"\n";
        }
      };
      dump(gScorePhoton, "photons");
      dump(gScoreHadron, "hadrons");
      qf.close();
    }
  }

  // do not touch external trace files if Ctrl+C
  if (!gCalledFromSignal) {
    auto strip = [](const std::string& t)->std::string{
      size_t p=t.find(" [ML:"); return (p==std::string::npos)? t : t.substr(0,p);
    };
    auto rescore = [this](TH1* h, double& s, bool& photon)->bool{
      if (!h) return false;
      const int n=h->GetNbinsX();
      if (n<=0) return false;
      std::vector<double> tr(n); for (int i=1;i<=n;++i) tr[i-1]=h->GetBinContent(i);
      auto F = ExtractEnhancedFeatures(tr); auto X = NormalizeFeatures(F);
      if (X.size()>=17){ X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5; }
      double ml = fNeuralNetwork->Predict(X,false);
      double penalty=0.0;
      if (F.pulse_width<120.0 && F.peak_charge_ratio>0.35) penalty-=0.10;
      if (F.num_peaks<=2 && F.secondary_peak_ratio>0.60)  penalty-=0.05;
      s = std::max(0.0, std::min(1.0, ml+penalty));
      photon = (s>fPhotonThreshold);
      return true;
    };
    auto annotate = [&](TDirectory* d, auto&& self)->void{
      if (!d) return;
      TIter nx(d->GetListOfKeys()); TKey* k;
      while ((k=(TKey*)nx())) {
        TObject* o = d->Get(k->GetName()); if (!o) continue;
        if (o->InheritsFrom(TDirectory::Class())) {
          self((TDirectory*)o, self);
        } else if (o->InheritsFrom(TH1::Class())) {
          TH1* h=(TH1*)o; double sc=0; bool ph=false;
          if (rescore(h,sc,ph)) {
            const std::string base = strip(h->GetTitle()?h->GetTitle():"");
            std::ostringstream t; t<<base<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "
                                   <<(ph?"ML-Photon":"ML-Hadron")<<"]";
            h->SetTitle(t.str().c_str());
            d->cd(); h->Write(h->GetName(), TObject::kOverwrite);
          }
        } else if (o->InheritsFrom(TCanvas::Class())) {
          TCanvas* c=(TCanvas*)o; TH1* h=nullptr;
          if (TList* P=c->GetListOfPrimitives()){ TIter ny(P); while (TObject* q=ny()){ if (q->InheritsFrom(TH1::Class())){ h=(TH1*)q; break; } } }
          if (h){
            double sc=0; bool ph=false;
            if (rescore(h,sc,ph)) {
              const std::string bC=strip(c->GetTitle()?c->GetTitle():"");
              const std::string bH=strip(h->GetTitle()?h->GetTitle():"");
              std::ostringstream tc,th;
              tc<<bC<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
              th<<bH<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
              c->SetTitle(tc.str().c_str()); h->SetTitle(th.str().c_str());
              c->Modified(); c->Update();
              d->cd(); c->Write(c->GetName(), TObject::kOverwrite);
            }
          }
        }
        delete o;
      }
    };

    const char* names[] = {"pmt_traces_1EeV.root","pmt_Traces_1EeV.root","pmt_traces_1eev.root","pmt_Traces_1eev.root"};
    TFile* f=nullptr;
    for (const char* nm: names){
      f=TFile::Open(nm,"UPDATE");
      if (f && !f->IsZombie()) break;
      if (f){ delete f; f=nullptr; }
    }
    if (f && !f->IsZombie()){
      if (!f->TestBit(TFile::kRecovered)) {
        annotate(f, annotate); f->Write("", TObject::kOverwrite);
        std::cout<<"Annotated ML tags in "<<f->GetName()<<"\n";
      } else {
        std::cout<<"Trace file recovered by ROOT; skip annotation.\n";
      }
      f->Close(); delete f;
    } else {
      std::cout<<"Trace file not found for annotation (skip).\n";
    }
  }

  if (fLogFile.is_open()) { fLogFile.flush(); fLogFile.close(); }
  std::cout << "==========================================\n";
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
  INFO("PhotonTriggerML::Finish()");
  SaveAndDisplaySummary();
  return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int id, MLResult& r)
{
  auto it=fMLResultsMap.find(id);
  if (it==fMLResultsMap.end()) return false;
  r=it->second; return true;
}

void PhotonTriggerML::ClearMLResults(){ fMLResultsMap.clear(); }

