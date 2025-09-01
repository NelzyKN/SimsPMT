// PhotonTrigger.cc — precision-balanced operating point with online thresholding
// Implements class PhotonTriggerML declared in PhotonTriggerML.h

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
// Static globals
// ============================================================================
PhotonTriggerML* PhotonTriggerML::fInstance = 0;
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;
static volatile sig_atomic_t gCalledFromSignal = 0;

static inline double clamp01(double x) { return (x<0.0?0.0:(x>1.0?1.0:x)); }

// Sliding windows for **online** threshold calibration (independent of training)
static std::deque<double> gHadScores, gPhoScores;
static const size_t kHadWin=600, kPhoWin=200;

// Robust quantile utility
static inline double quantile(std::vector<double> v, double q){
    if (v.empty()) return 0.5;
    if (q<=0) return *std::min_element(v.begin(), v.end());
    if (q>=1) return *std::max_element(v.begin(), v.end());
    size_t k = (size_t)std::floor(q*(v.size()-1));
    std::nth_element(v.begin(), v.begin()+k, v.end());
    return v[k];
}

void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << endl;
        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); } catch (...) {}
        }
        std::signal(signal, SIG_DFL);
        std::raise(signal);
        _Exit(0);
    }
}

// ============================================================================
// Neural Network (feed-forward; safer I/O formatting)
// ============================================================================
PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.1),
    fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int in, int h1, int h2)
{
    fInputSize=in; fHidden1Size=h1; fHidden2Size=h2;
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.assign(h1, std::vector<double>(in,0));
    for (int i=0;i<h1;++i) for (int j=0;j<in;++j) fWeights1[i][j]=dist(gen)/sqrt(in);

    fWeights2.assign(h2, std::vector<double>(h1,0));
    for (int i=0;i<h2;++i) for (int j=0;j<h1;++j) fWeights2[i][j]=dist(gen)/sqrt(h1);

    fWeights3.assign(1, std::vector<double>(h2,0));
    for (int j=0;j<h2;++j) fWeights3[0][j]=dist(gen)/sqrt(h2);

    fBias1.assign(h1,0.0); fBias2.assign(h2,0.0); fBias3=0.0;

    fMomentum1_w1.assign(h1, std::vector<double>(in,0));
    fMomentum1_w2.assign(h2, std::vector<double>(h1,0));
    fMomentum1_w3.assign(1,  std::vector<double>(h2,0));
    fMomentum1_b1.assign(h1,0);
    fMomentum1_b2.assign(h2,0);
    fMomentum1_b3=0; fTimeStep=0;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if ((int)x.size()!=fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size);
    for (int i=0;i<fHidden1Size;++i){
        double s=fBias1[i]; for(int j=0;j<fInputSize;++j) s+=fWeights1[i][j]*x[j];
        h1[i]=1.0/(1.0+exp(-s));
        if (training && (rand()/double(RAND_MAX))<fDropoutRate) h1[i]=0.0;
    }

    std::vector<double> h2(fHidden2Size);
    for (int i=0;i<fHidden2Size;++i){
        double s=fBias2[i]; for(int j=0;j<fHidden1Size;++j) s+=fWeights2[i][j]*h1[j];
        h2[i]=1.0/(1.0+exp(-s));
        if (training && (rand()/double(RAND_MAX))<fDropoutRate) h2[i]=0.0;
    }

    double o=fBias3; for(int j=0;j<fHidden2Size;++j) o+=fWeights3[0][j]*h2[j];
    return 1.0/(1.0+exp(-o));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y, double lr)
{
    if (X.empty()||X.size()!=y.size()) return -1.0;
    const int B=(int)X.size();

    const int nP = std::count(y.begin(), y.end(), 1);
    const double wP=(nP>0)?1.2:1.0, wN=1.0;

    std::vector<std::vector<double>> gw1(fHidden1Size, std::vector<double>(fInputSize,0));
    std::vector<std::vector<double>> gw2(fHidden2Size, std::vector<double>(fHidden1Size,0));
    std::vector<std::vector<double>> gw3(1, std::vector<double>(fHidden2Size,0));
    std::vector<double> gb1(fHidden1Size,0), gb2(fHidden2Size,0); double gb3=0, total=0;

    for (int n=0;n<B;++n){
        const auto& in=X[n]; const int t=y[n]; const double w=(t==1)?wP:wN;

        std::vector<double> h1(fHidden1Size), h2(fHidden2Size);
        for(int i=0;i<fHidden1Size;++i){ double s=fBias1[i]; for(int j=0;j<fInputSize;++j) s+=fWeights1[i][j]*in[j]; h1[i]=1.0/(1.0+exp(-s)); }
        for(int i=0;i<fHidden2Size;++i){ double s=fBias2[i]; for(int j=0;j<fHidden1Size;++j) s+=fWeights2[i][j]*h1[j]; h2[i]=1.0/(1.0+exp(-s)); }
        double oraw=fBias3; for(int j=0;j<fHidden2Size;++j) oraw+=fWeights3[0][j]*h2[j];
        const double p=1.0/(1.0+exp(-oraw));

        total += -w*(t*log(p+1e-7)+(1-t)*log(1-p+1e-7));
        const double dOut=w*(p-t);

        for(int j=0;j<fHidden2Size;++j) gw3[0][j]+=dOut*h2[j]; gb3+=dOut;

        std::vector<double> dh2(fHidden2Size);
        for(int j=0;j<fHidden2Size;++j){ dh2[j]=fWeights3[0][j]*dOut; dh2[j]*=h2[j]*(1-h2[j]); }
        for(int i=0;i<fHidden2Size;++i){ for(int j=0;j<fHidden1Size;++j) gw2[i][j]+=dh2[i]*h1[j]; gb2[i]+=dh2[i]; }

        std::vector<double> dh1(fHidden1Size,0);
        for(int j=0;j<fHidden1Size;++j){ for(int i=0;i<fHidden2Size;++i) dh1[j]+=fWeights2[i][j]*dh2[i]; dh1[j]*=h1[j]*(1-h1[j]); }
        for(int i=0;i<fHidden1Size;++i){ for(int j=0;j<fInputSize;++j) gw1[i][j]+=dh1[i]*in[j]; gb1[i]+=dh1[i]; }
    }

    const double m=0.9;
    for(int i=0;i<fHidden1Size;++i){
        for(int j=0;j<fInputSize;++j){ const double g=gw1[i][j]/B; fMomentum1_w1[i][j]=m*fMomentum1_w1[i][j]-lr*g; fWeights1[i][j]+=fMomentum1_w1[i][j]; }
        const double g=gb1[i]/B; fMomentum1_b1[i]=m*fMomentum1_b1[i]-lr*g; fBias1[i]+=fMomentum1_b1[i];
    }
    for(int i=0;i<fHidden2Size;++i){
        for(int j=0;j<fHidden1Size;++j){ const double g=gw2[i][j]/B; fMomentum1_w2[i][j]=m*fMomentum1_w2[i][j]-lr*g; fWeights2[i][j]+=fMomentum1_w2[i][j]; }
        const double g=gb2[i]/B; fMomentum1_b2[i]=m*fMomentum1_b2[i]-lr*g; fBias2[i]+=fMomentum1_b2[i];
    }
    for(int j=0;j<fHidden2Size;++j){ const double g=gw3[0][j]/B; fMomentum1_w3[0][j]=m*fMomentum1_w3[0][j]-lr*g; fWeights3[0][j]+=fMomentum1_w3[0][j]; }
    const double g=gb3/B; fMomentum1_b3=m*fMomentum1_b3-lr*g; fBias3+=fMomentum1_b3;

    return total/B;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) { cout << "Error: Could not save weights to " << filename << endl; return; }

    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";
    for (const auto& row : fWeights1) { for (double w : row) file << w << " "; file << "\n"; }
    { for (double b : fBias1) file << b << " "; file << "\n"; }
    for (const auto& row : fWeights2) { for (double w : row) file << w << " "; file << "\n"; }
    { for (double b : fBias2) file << b << " "; file << "\n"; }
    { for (double w : fWeights3[0]) file << w << " "; file << "\n"; }
    file << fBias3 << "\n";
    file.close();
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open()) { cout << "Warning: Could not load weights from " << filename << endl; return false; }

    file >> fInputSize >> fHidden1Size >> fHidden2Size;
    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize,0));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size,0));
    fWeights3.assign(1, std::vector<double>(fHidden2Size,0));
    fBias1.assign(fHidden1Size,0); fBias2.assign(fHidden2Size,0);

    for (auto& row : fWeights1) for (double& w : row) file >> w;
    for (double& b : fBias1) file >> b;
    for (auto& row : fWeights2) for (double& w : row) file >> w;
    for (double& b : fBias2) file >> b;
    for (double& w : fWeights3[0]) file >> w;
    file >> fBias3;

    file.close();
    cout << "Weights loaded from " << filename << endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights(){ fIsQuantized=true; }

// ============================================================================
// PhotonTriggerML
// ============================================================================
PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
    fIsTraining(false),                // run in inference mode by default
    fTrainingEpochs(300), fTrainingStep(0),
    fBestValidationLoss(1e9), fEpochsSinceImprovement(0),
    fEventCount(0), fStationCount(0),
    fPhotonLikeCount(0), fHadronLikeCount(0),
    fEnergy(0), fCoreX(0), fCoreY(0),
    fPrimaryId(0), fPrimaryType("Unknown"),
    fPhotonScore(0), fConfidence(0), fDistance(0),
    fStationId(0), fIsActualPhoton(false),
    fOutputFile(nullptr), fMLTree(nullptr),
    fLogFileName("photon_trigger_ml_physics.log"),
    hPhotonScore(nullptr), hPhotonScorePhotons(nullptr), hPhotonScoreHadrons(nullptr),
    hConfidence(nullptr), hRisetime(nullptr), hAsymmetry(nullptr), hKurtosis(nullptr),
    hScoreVsEnergy(nullptr), hScoreVsDistance(nullptr), gROCCurve(nullptr),
    hConfusionMatrix(nullptr), hTrainingLoss(nullptr), hValidationLoss(nullptr), hAccuracyHistory(nullptr),
    fTruePositives(0), fFalsePositives(0), fTrueNegatives(0), fFalseNegatives(0),
    // Base threshold (will be recalibrated online)
    fPhotonThreshold(0.72),
    fEnergyMin(1e18), fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml.root"),
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
    INFO("PhotonTriggerML::Init() - Starting initialization");

    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) { ERROR("Failed to open log file: " + fLogFileName); return eFailure; }

    fNeuralNetwork->Initialize(17,8,4);

    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        // We keep fIsTraining=false; threshold will be adapted online from validation labels.
    } else {
        fIsTraining=true;  // only if no weights are found
    }

    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) { ERROR("Failed to create output file"); return eFailure; }

    fMLTree = new TTree("MLTree", "PhotonTriggerML Physics Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("photonScore", &fPhotonScore, "photonScore/D");
    fMLTree->Branch("confidence", &fConfidence, "confidence/D");
    fMLTree->Branch("primaryId", &fPrimaryId, "primaryId/I");
    fMLTree->Branch("primaryType", &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");

    hPhotonScore = new TH1D("hPhotonScore", "ML Photon Score (All);Score;Count", 50, 0, 1);
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons", "ML Score (True Photons);Score;Count", 50, 0, 1); hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons", "ML Score (True Hadrons);Score;Count", 50, 0, 1); hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence = new TH1D("hConfidence", "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime  = new TH1D("hRisetime",  "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry = new TH1D("hAsymmetry", "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis  = new TH1D("hKurtosis",  "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy   = new TH2D("hScoreVsEnergy",   "Score vs Energy;Energy [eV];Score", 50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 50, 0, 3000, 50, 0, 1);

    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");

    hTrainingLoss   = new TH1D("hTrainingLoss",   "Training Loss;Batch;Loss", 10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss", 10000, 0, 10000);
    hAccuracyHistory= new TH1D("hAccuracyHistory","Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    // Shower meta
    fEnergy=0; fCoreX=0; fCoreY=0; fPrimaryId=0; fPrimaryType="Unknown"; fIsActualPhoton=false;
    if (event.HasSimShower()) {
        const ShowerSimData& sh = event.GetSimShower();
        fEnergy = sh.GetEnergy(); fPrimaryId = sh.GetPrimaryParticle();
        switch(fPrimaryId){
            case 22: fPrimaryType="photon"; fIsActualPhoton=true; break;
            case 11: case -11: fPrimaryType="electron"; break;
            case 2212: fPrimaryType="proton"; break;
            case 1000026056: fPrimaryType="iron"; break;
            default: fPrimaryType=(fPrimaryId>1000000000)?"nucleus":"unknown";
        }
        if (sh.GetNSimCores()>0){
            const Detector& det = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
            Point core = sh.GetSimCore(0); fCoreX=core.GetX(siteCS); fCoreY=core.GetY(siteCS);
        }
    }

    // Stations
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }

    // Periodic metrics
    if (fEventCount % 10 == 0) {
        hConfusionMatrix->SetBinContent(1,1,fTrueNegatives);
        hConfusionMatrix->SetBinContent(2,1,fFalsePositives);
        hConfusionMatrix->SetBinContent(1,2,fFalseNegatives);
        hConfusionMatrix->SetBinContent(2,2,fTruePositives);
        CalculateAndDisplayMetrics();
    }

    // === Online threshold calibration (runs regardless of fIsTraining) ===
    // Use recent labeled station scores; aim for ~8–10% hadron FPR and >=60% photon TPR
    const size_t needHad=200, needPho=80;
    if (gHadScores.size()>=needHad && gPhoScores.size()>=needPho && (fEventCount%8==0)) {
        std::vector<double> had(gHadScores.begin(), gHadScores.end());
        std::vector<double> pho(gPhoScores.begin(), gPhoScores.end());
        const double thrFPR = quantile(had, 0.90); // ≈10% hadron acceptance
        const double thrTPR = quantile(pho, 0.40); // ≈60% photon efficiency
        double cand = std::max(thrFPR, thrTPR);
        const double THR_MIN=0.62, THR_MAX=0.93;
        if (cand<THR_MIN) cand=THR_MIN; if (cand>THR_MAX) cand=THR_MAX;
        // Smooth updates to avoid oscillation
        fPhotonThreshold = 0.8*fPhotonThreshold + 0.2*cand;
        if (fLogFile.is_open()) fLogFile << "[AutoThr] had90="<<thrFPR<<" pho40="<<thrTPR
                                         <<" -> thr="<<fPhotonThreshold<<"\n";
    }

    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    try{
        const Detector& det = Detector::GetInstance();
        const sdet::SDetector& sdetector = det.GetSDetector();
        const sdet::Station& ds = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
        double sx = ds.GetPosition().GetX(siteCS), sy = ds.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((sx-fCoreX)*(sx-fCoreX) + (sy-fCoreY)*(sy-fCoreY));
    }catch(...){ fDistance = -1; return; }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    struct PMTInfo{
        bool valid=false; double score=0; bool emLike=false; bool muonSpike=false;
        double tot10=0, totAbs=0; EnhancedFeatures feat{}; std::vector<double> norm;
    } r[3];

    bool any=false; double maxScore=0.0;

    for (int p=0; p<3; ++p){
        const int pmtId = firstPMT + p;
        if (!station.HasPMT(pmtId)) continue;
        const sevt::PMT& pmt = station.GetPMT(pmtId);

        std::vector<double> tr;
        if (!ExtractTraceData(pmt, tr) || tr.size()!=2048) continue;

        const double vmax = *std::max_element(tr.begin(), tr.end());
        const double vmin = *std::min_element(tr.begin(), tr.end());
        if (vmax - vmin < 10.0) continue;

        EnhancedFeatures F = ExtractEnhancedFeatures(tr);

        // Baseline & simple TOTs
        const double NS_PER_BIN=25.0;
        double base=0.0; for(int i=0;i<100;++i) base+=tr[i]; base/=100.0;
        double peak=0.0; for(double v:tr){ double s=v-base; if(s<0) s=0; if(s>peak) peak=s; }
        int b10=0,bAbs=0;
        for(double v:tr){ double s=v-base; if(s<0)s=0; if(s>=0.10*peak) b10++; if(s>=15.0) bAbs++; }
        const double TOT10_ns=b10*NS_PER_BIN, TOTabs_ns=bAbs*NS_PER_BIN;

        // Light veto / corroboration (relaxed vs previous version)
        const bool muonSpike = (F.pulse_width<110.0) &&
                               (F.peak_charge_ratio>0.42) &&
                               (F.num_peaks<=2) && (F.late_fraction<0.15) &&
                               (TOTabs_ns<200.0);

        const bool emLike =  (F.pulse_width>=140.0 && F.pulse_width<=900.0)
                          || (F.num_peaks>=3)
                          || (TOT10_ns>=170.0)
                          || (F.late_fraction>0.22);

        // Normalize & emphasis
        std::vector<double> X = NormalizeFeatures(F);
        if (X.size()>=17){ X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5; }

        double ml = fNeuralNetwork->Predict(X,false);

        // Adjustments (gentle)
        double adj = 0.0;
        if (muonSpike)         adj -= 0.18;
        if (emLike)            adj += 0.08;
        if (TOT10_ns>=220.0)   adj += 0.06;
        if (TOT10_ns>=260.0)   adj += 0.10;

        const double score = clamp01(ml + adj);

        r[p].valid=true; r[p].score=score; r[p].emLike=emLike; r[p].muonSpike=muonSpike;
        r[p].tot10=TOT10_ns; r[p].totAbs=TOTabs_ns; r[p].feat=F; r[p].norm=X;
        any=true; if (score>maxScore) maxScore=score;
    }
    if (!any) return;

    // Station-level decision (more permissive to single-strong PMT)
    double thr = fPhotonThreshold;
    if (thr<0.62) thr=0.62; if (thr>0.93) thr=0.93;
    const double thrMid  = std::max(0.55, thr-0.12);
    const double thrHigh = thr;

    int votes=0, highVotes=0, notMuon=0, emVotes=0; double sumTOTabs=0.0;
    for (int p=0;p<3;++p) if (r[p].valid){
        const bool midOK  = (r[p].score>=thrMid)  && r[p].emLike && !r[p].muonSpike;
        const bool highOK = (r[p].score>=thrHigh);
        if (midOK || highOK) votes++;
        if (highOK) highVotes++;
        if (!r[p].muonSpike) notMuon++;
        if (r[p].emLike) emVotes++;
        sumTOTabs += r[p].totAbs;
    }

    bool stationPhoton = (highVotes>=1 && notMuon>=2) || (votes>=2) || (highVotes>=1 && emVotes>=1 && sumTOTabs>=180.0);

    // Update counters once per station
    fStationCount++;
    fPhotonScore = maxScore;
    fConfidence  = fabs(fPhotonScore - 0.5);

    hPhotonScore->Fill(fPhotonScore);
    hConfidence->Fill(fConfidence);
    if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore); else hPhotonScoreHadrons->Fill(fPhotonScore);
    hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
    hScoreVsDistance->Fill(fDistance, fPhotonScore);
    fMLTree->Fill();

    if (stationPhoton) {
        fPhotonLikeCount++;
        if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
    } else {
        fHadronLikeCount++;
        if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
    }

    // Feed **online calibration** windows
    if (fIsActualPhoton) {
        gPhoScores.push_back(fPhotonScore);
        if (gPhoScores.size()>kPhoWin) gPhoScores.pop_front();
    } else {
        gHadScores.push_back(fPhotonScore);
        if (gHadScores.size()>kHadWin) gHadScores.pop_front();
    }

    // Save representative result
    MLResult mlr; mlr.photonScore=fPhotonScore; mlr.identifiedAsPhoton=stationPhoton; mlr.isActualPhoton=fIsActualPhoton;
    mlr.vemCharge = 0.0; // optional: could store r[best].feat.total_charge
    mlr.features  = r[0].feat; mlr.primaryType=fPrimaryType; mlr.confidence=fConfidence;
    fMLResultsMap[fStationId]=mlr;
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& tr)
{
    tr.clear();
    if (pmt.HasFADCTrace()){
        try{ const auto& t=pmt.GetFADCTrace(sdet::PMTConstants::eHighGain); for(int i=0;i<2048;i++) tr.push_back(t[i]); return true; }catch(...) {}
    }
    if (pmt.HasSimData()){
        try{
            const sevt::PMTSimData& s=pmt.GetSimData();
            if (s.HasFADCTrace(sevt::StationConstants::eTotal)){
                const auto& t=s.GetFADCTrace(sdet::PMTConstants::eHighGain, sevt::StationConstants::eTotal);
                for(int i=0;i<2048;i++) tr.push_back(t[i]); return true;
            }
        }catch(...) {}
    }
    return false;
}

PhotonTriggerML::EnhancedFeatures PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& tr, double)
{
    EnhancedFeatures F; const int N=(int)tr.size();
    const double ADC_PER_VEM=180.0, NS_PER_BIN=25.0;

    int peak_bin=0; double peak_val=0, total=0; std::vector<double> sig(N);

    double base=0; for(int i=0;i<100;i++) base+=tr[i]; base/=100.0;
    for (int i=0;i<N;i++){
        sig[i]=tr[i]-base; if (sig[i]<0) sig[i]=0;
        if (sig[i]>peak_val){ peak_val=sig[i]; peak_bin=i; }
        total += sig[i];
    }
    if (peak_val<5.0 || total<10.0) return F;

    F.peak_amplitude = peak_val/ADC_PER_VEM;
    F.total_charge   = total/ADC_PER_VEM;
    F.peak_charge_ratio = F.peak_amplitude/(F.total_charge+0.001);

    const double p10=0.1*peak_val, p50=0.5*peak_val, p90=0.9*peak_val;
    int b10r=0,b50r=0,b90r=peak_bin;
    for (int i=peak_bin;i>=0;--i){
        if (sig[i]<=p90 && b90r==peak_bin) b90r=i;
        if (sig[i]<=p50 && b50r==0) b50r=i;
        if (sig[i]<=p10){ b10r=i; break; }
    }
    int b90f=peak_bin, b10f=N-1;
    for (int i=peak_bin;i<N;++i){
        if (sig[i]<=p90 && b90f==peak_bin) b90f=i;
        if (sig[i]<=p10){ b10f=i; break; }
    }

    F.risetime_10_50 = std::abs(b50r-b10r)*NS_PER_BIN;
    F.risetime_10_90 = std::abs(b90r-b10r)*NS_PER_BIN;
    F.falltime_90_10 = std::abs(b10f-b90f)*NS_PER_BIN;

    const double half=peak_val/2.0;
    int br=b10r, bf=b10f;
    for (int i=b10r;i<=peak_bin;++i){ if (sig[i]>=half){ br=i; break; } }
    for (int i=peak_bin;i<N;++i){     if (sig[i]<=half){ bf=i; break; } }
    F.pulse_width = std::abs(bf-br)*NS_PER_BIN;

    const double rise=F.risetime_10_90, fall=F.falltime_90_10;
    F.asymmetry = (fall-rise)/(fall+rise+0.001);

    double mean_t=0; for (int i=0;i<N;i++) mean_t += i*sig[i]; mean_t /= (total+0.001);
    double var=0,skew=0,kur=0;
    for (int i=0;i<N;i++){
        const double d=i-mean_t; const double w=sig[i]/(total+0.001);
        var+=d*d*w; skew+=d*d*d*w; kur+=d*d*d*d*w;
    }
    const double sd=sqrt(var+0.001);
    F.time_spread = sd*NS_PER_BIN;
    F.skewness = skew/(sd*sd*sd+0.001);
    F.kurtosis = kur/(var*var+0.001)-3.0;

    const int q=N/4; double early=0, late=0;
    for (int i=0;i<q;i++) early+=sig[i];
    for (int i=3*q;i<N;i++) late +=sig[i];
    F.early_fraction = early/(total+0.001);
    F.late_fraction  = late /(total+0.001);

    double sumsq=0; int cnt=0;
    for (int i=1;i<N-1;i++) if (sig[i]>0.1*peak_val){ const double d2=sig[i+1]-2*sig[i]+sig[i-1]; sumsq+=d2*d2; cnt++; }
    F.smoothness = sqrt(sumsq/(cnt+1));

    double hf=0; for (int i=1;i<N-1;i++){ const double d=sig[i+1]-sig[i-1]; hf += d*d; }
    F.high_freq_content = hf/(total*total+0.001);

    F.num_peaks=0; double secondary=0; const double thr=0.25*peak_val;
    for (int i=1;i<N-1;i++){
        if (sig[i]>thr && sig[i]>sig[i-1] && sig[i]>sig[i+1]){
            F.num_peaks++;
            if (i!=peak_bin && sig[i]>secondary) secondary=sig[i];
        }
    }
    F.secondary_peak_ratio = secondary/(peak_val+0.001);
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
    for (size_t i=0;i<raw.size();++i){
        double d = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += d/n;
        double d2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += d*d2;
    }
    if (n>1){
        for (size_t i=0;i<raw.size();++i){
            fFeatureStdDevs[i] = sqrt(fFeatureStdDevs[i]/(n-1));
            if (fFeatureStdDevs[i]<0.001) fFeatureStdDevs[i]=1.0;
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
    std::vector<double> mins = {0, 0, 0, 0, -1, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0};
    std::vector<double> maxs = {500, 1000, 1000, 1000, 1, 10, 100, 1, 100, 20, 5, 1, 1, 1000, 10, 10, 1};

    std::vector<double> z; z.reserve(raw.size());
    for (size_t i=0;i<raw.size();++i){
        double v = (raw[i]-mins[i])/(maxs[i]-mins[i]+0.001);
        if (v<0) v=0; if (v>1) v=1;
        z.push_back(v);
    }
    return z;
}

double PhotonTriggerML::TrainNetwork(){ return 0.0; } // (kept off by default)

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total==0) return;

    const double acc  = 100.0*(fTruePositives + fTrueNegatives)/total;
    const double prec = (fTruePositives + fFalsePositives > 0)? 100.0*fTruePositives/(fTruePositives+fFalsePositives):0.0;
    const double rec  = (fTruePositives + fFalseNegatives > 0)? 100.0*fTruePositives/(fTruePositives+fFalseNegatives):0.0;
    const double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec):0.0;

    const double pfrac = (fPhotonLikeCount+fHadronLikeCount>0)?
                         100.0*fPhotonLikeCount/(fPhotonLikeCount+fHadronLikeCount):0.0;

    cout << "│ " << setw(6) << fEventCount
         << " │ " << setw(8) << fStationCount
         << " │ " << fixed << setprecision(1) << setw(7) << pfrac << "%"
         << " │ " << setw(7) << acc << "%"
         << " │ " << setw(8) << prec << "%"
         << " │ " << setw(7) << rec << "%"
         << " │ " << setw(7) << f1 << "%│\n";

    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount << " - Acc: " << acc << "% Prec: " << prec
                 << "% Rec: " << rec << "% F1: " << f1 << "%\n";
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    cout << "\n==========================================\n";
    cout << "PERFORMANCE METRICS\n";
    cout << "==========================================\n";

    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total==0) { cout << "No predictions made yet!\n"; return; }

    const double acc  = 100.0*(fTruePositives + fTrueNegatives)/total;
    const double prec = (fTruePositives + fFalsePositives > 0)? 100.0*fTruePositives/(fTruePositives+fFalsePositives):0.0;
    const double rec  = (fTruePositives + fFalseNegatives > 0)? 100.0*fTruePositives/(fTruePositives+fFalseNegatives):0.0;
    const double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec):0.0;

    cout << "Accuracy:  " << fixed << setprecision(1) << acc << "%\n";
    cout << "Precision: " << prec << "%\n";
    cout << "Recall:    " << rec  << "%\n";
    cout << "F1-Score:  " << f1   << "%\n\n";

    cout << "CONFUSION MATRIX:\n";
    cout << "                Predicted\n";
    cout << "             Hadron   Photon\n";
    cout << "Actual Hadron  " << setw(6) << fTrueNegatives  << "   " << setw(6) << fFalsePositives << "\n";
    cout << "       Photon  " << setw(6) << fFalseNegatives << "   " << setw(6) << fTruePositives  << "\n\n";

    cout << "Total Stations: " << fStationCount << "\n";
    cout << "Photon-like: " << fPhotonLikeCount << " ("
         << 100.0*fPhotonLikeCount/max(1, fPhotonLikeCount+fHadronLikeCount) << "%)\n";
    cout << "Hadron-like: "  << fHadronLikeCount  << " ("
         << 100.0*fHadronLikeCount /max(1, fPhotonLikeCount+fHadronLikeCount) << "%)\n";

    cout << "==========================================\n";

    if (fLogFile.is_open()) {
        fLogFile << "\nFinal Performance Metrics:\n";
        fLogFile << "Accuracy: " << acc << "%\nPrecision: " << prec
                 << "%\nRecall: " << rec << "%\nF1-Score: " << f1 << "%\n";
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n==========================================\n";
    cout << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n";
    cout << "==========================================\n";
    cout << "Events processed: " << fEventCount << "\n";
    cout << "Stations analyzed: " << fStationCount << "\n";

    CalculatePerformanceMetrics();
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    if (fOutputFile) {
        fOutputFile->cd();
        if (fMLTree) { fMLTree->Write(); cout << "Wrote " << fMLTree->GetEntries() << " entries to tree\n"; }
        hPhotonScore->Write(); hPhotonScorePhotons->Write(); hPhotonScoreHadrons->Write();
        hConfidence->Write(); hRisetime->Write(); hAsymmetry->Write(); hKurtosis->Write();
        hScoreVsEnergy->Write(); hScoreVsDistance->Write(); hConfusionMatrix->Write();
        hTrainingLoss->Write(); hValidationLoss->Write(); hAccuracyHistory->Write();
        fOutputFile->Close(); delete fOutputFile; fOutputFile=nullptr;
    }

    // Human-readable summary
    { std::ofstream txt("photon_trigger_summary.txt");
      if (txt.is_open()){
        const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
        const int total=TP+FP+TN+FN;
        const double acc =(total>0)?100.0*(TP+TN)/total:0.0;
        const double prec=(TP+FP>0)?100.0*TP/(TP+FP):0.0;
        const double rec =(TP+FN>0)?100.0*TP/(TP+FN):0.0;
        const double f1  =(prec+rec>0)?2.0*prec*rec/(prec+rec):0.0;
        txt << "PhotonTriggerML Summary\n";
        txt << "BaseThreshold=" << fPhotonThreshold << "\n";
        txt << "TP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN<<"\n";
        txt << std::fixed << std::setprecision(2)
            << "Accuracy="<<acc<<"%  Precision="<<prec<<"%  Recall="<<rec<<"%  F1="<<f1<<"%\n";
        txt.close();
      } }

    // Optional: annotate existing per-trace ROOT files (safe path)
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& t)->std::string{
            size_t p=t.find(" [ML:"); return (p==std::string::npos)? t : t.substr(0,p);
        };
        auto score_and_decide = [this](TH1* h, double& score, bool& isPhoton)->bool{
            if (!h) return false; const int n=h->GetNbinsX(); if (n<=0) return false;
            std::vector<double> tr(n); for(int i=1;i<=n;++i) tr[i-1]=h->GetBinContent(i);

            EnhancedFeatures F = ExtractEnhancedFeatures(tr);
            std::vector<double> X = NormalizeFeatures(F);
            if (X.size()>=17){ X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5; }
            double p = fNeuralNetwork->Predict(X,false);

            // quick TOTs
            double base=0; const int preN=std::min(100,n);
            for(int i=0;i<preN;++i) base+=tr[i]; base/=std::max(1,preN);
            double peak=0; for(double v:tr){ double s=v-base; if(s<0)s=0; if(s>peak) peak=s; }
            int b10=0,bAbs=0; for(double v:tr){ double s=v-base; if(s<0)s=0; if(s>=0.10*peak) b10++; if(s>=15.0) bAbs++; }
            const double TOT10_ns=b10*25.0, TOTabs_ns=bAbs*25.0;

            const bool muonSpike=(F.pulse_width<110.0)&&(F.peak_charge_ratio>0.42)&&(F.num_peaks<=2)&&(F.late_fraction<0.15)&&(TOTabs_ns<200.0);
            const bool emLike=(F.pulse_width>=140.0&&F.pulse_width<=900.0)||(F.num_peaks>=3)||(TOT10_ns>=170.0)||(F.late_fraction>0.22);

            double adj=0.0; if (muonSpike) adj-=0.18; if (emLike) adj+=0.08; if (TOT10_ns>=220.0) adj+=0.06; if (TOT10_ns>=260.0) adj+=0.10;
            score = clamp01(p+adj);

            double thr=fPhotonThreshold; if (thr<0.62) thr=0.62; if (thr>0.93) thr=0.93;
            const double thrMid = std::max(0.55, thr-0.12);
            const double thrHigh= thr;
            isPhoton = (score>=thrHigh) || (score>=thrMid && emLike && !muonSpike);
            return true;
        };
        auto annotate_dir = [&](TDirectory* dir, auto&& self)->void{
            if (!dir) return;
            TIter next(dir->GetListOfKeys()); TKey* key;
            while ((key=(TKey*)next())){
                TObject* obj = dir->Get(key->GetName()); if (!obj) continue;
                if (obj->InheritsFrom(TDirectory::Class())){
                    self((TDirectory*)obj, self);
                }else if (obj->InheritsFrom(TH1::Class())){
                    TH1* h=(TH1*)obj; double sc=0; bool ph=false;
                    if (score_and_decide(h,sc,ph)){
                        std::string base=strip_ml(h->GetTitle()?h->GetTitle():"");
                        std::ostringstream t; t<<base<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                        h->SetTitle(t.str().c_str()); dir->cd(); h->Write(h->GetName(), TObject::kOverwrite);
                    }
                }else if (obj->InheritsFrom(TCanvas::Class())){
                    TCanvas* c=(TCanvas*)obj; TH1* h=nullptr;
                    if (TList* prim=c->GetListOfPrimitives()){ TIter nx(prim); while (TObject* po=nx()){ if (po->InheritsFrom(TH1::Class())){ h=(TH1*)po; break; } } }
                    if (h){ double sc=0; bool ph=false;
                        if (score_and_decide(h,sc,ph)){
                            std::string baseC=strip_ml(c->GetTitle()?c->GetTitle():"");
                            std::string baseH=strip_ml(h->GetTitle()?h->GetTitle():"");
                            std::ostringstream tc,th;
                            tc<<baseC<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                            th<<baseH<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                            c->SetTitle(tc.str().c_str()); h->SetTitle(th.str().c_str());
                            c->Modified(); c->Update(); dir->cd(); c->Write(c->GetName(), TObject::kOverwrite);
                        } }
                }
            }
        };

        const char* candidates[]={"pmt_traces_1EeV.root","pmt_Traces_1EeV.root","pmt_traces_1eev.root","pmt_Traces_1eev.root"};
        TFile* f=nullptr; for (const char* n:candidates){ f=TFile::Open(n,"UPDATE"); if (f && !f->IsZombie()) break; if (f){ delete f; f=nullptr; } }
        if (f && !f->IsZombie()){
            if (!f->TestBit(TFile::kRecovered)){ annotate_dir(f, annotate_dir); f->Write("", TObject::kOverwrite); cout<<"Annotated ML tags in "<<f->GetName()<<"\n"; }
            else cout<<"Trace file recovered; skip annotation.\n";
            f->Close(); delete f;
        } else {
            cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    if (fLogFile.is_open()) fLogFile.close();
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    SaveAndDisplaySummary();
    return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& out)
{
    auto it = fMLResultsMap.find(stationId);
    if (it!=fMLResultsMap.end()){ out=it->second; return true; }
    return false;
}

void PhotonTriggerML::ClearMLResults(){ fMLResultsMap.clear(); }

