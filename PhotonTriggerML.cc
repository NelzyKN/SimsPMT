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
#include <TGraph.h>
#include <TMath.h>
#include <TRandom3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace det;
using namespace utl;

// ============================================================================
// Helpers (local)
// ============================================================================
namespace {
inline double clamp01(double x) { return std::max(0.0, std::min(1.0, x)); }
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double logit(double p) {
    const double eps = 1e-8;
    double q = std::min(1.0 - eps, std::max(eps, p));
    return std::log(q / (1.0 - q));
}
template <class T>
inline void bounded_push(std::deque<T>& dq, const T& v, size_t cap) {
    dq.push_back(v);
    if (dq.size() > cap) dq.pop_front();
}
struct PlattCalib { double alpha = 1.0; double beta = 0.0; };
} // namespace

// ============================================================================
// Static members
// ============================================================================
PhotonTriggerML* PhotonTriggerML::fInstance = 0;
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Flag for signal-safe summary
static volatile sig_atomic_t gInSignal = 0;

static void PhotonTriggerMLSignalHandler(int sig) {
    if (sig == SIGINT || sig == SIGTSTP) {
        std::cout << "\n[PhotonTriggerML] interrupt received → writing summary safely...\n";
        gInSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); } catch (...) {}
        }
        std::signal(sig, SIG_DFL);
        std::raise(sig);
        _Exit(0);
    }
}

// ============================================================================
// Neural network (compact, Adam, no warnings)
// ============================================================================
PhotonTriggerML::NeuralNetwork::NeuralNetwork()
: fInputSize(0), fHidden1Size(0), fHidden2Size(0),
  fBias3(0.0),
  fMomentum1_b3(0.0), fMomentum2_b3(0.0),
  fTimeStep(0), fDropoutRate(0.1) {}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int h1, int h2)
{
    fInputSize   = input_size;
    fHidden1Size = h1;
    fHidden2Size = h2;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> U(-0.5, 0.5);

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.assign(1,            std::vector<double>(fHidden2Size));

    for (int i=0;i<fHidden1Size;++i)
        for (int j=0;j<fInputSize;++j)
            fWeights1[i][j] = U(gen) / std::sqrt(fInputSize);

    for (int i=0;i<fHidden2Size;++i)
        for (int j=0;j<fHidden1Size;++j)
            fWeights2[i][j] = U(gen) / std::sqrt(fHidden1Size);

    for (int j=0;j<fHidden2Size;++j)
        fWeights3[0][j] = U(gen) / std::sqrt(fHidden2Size);

    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);
    fBias3 = 0.0;

    fMomentum1_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fMomentum2_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fMomentum1_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fMomentum2_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fMomentum1_w3.assign(1,            std::vector<double>(fHidden2Size, 0.0));
    fMomentum2_w3.assign(1,            std::vector<double>(fHidden2Size, 0.0));

    fMomentum1_b1.assign(fHidden1Size, 0.0);
    fMomentum2_b1.assign(fHidden1Size, 0.0);
    fMomentum1_b2.assign(fHidden2Size, 0.0);
    fMomentum2_b2.assign(fHidden2Size, 0.0);
    fMomentum1_b3 = 0.0;
    fMomentum2_b3 = 0.0;

    fTimeStep = 0;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& X, bool training)
{
    if ((int)X.size() != fInputSize) return 0.5;

    // h1 = tanh(W1 x + b1)
    std::vector<double> h1(fHidden1Size);
    for (int i=0;i<fHidden1Size;++i) {
        double s = fBias1[i];
        for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*X[j];
        h1[i] = std::tanh(s);
        if (training && (std::rand() / double(RAND_MAX)) < fDropoutRate) h1[i] = 0.0;
    }

    // h2 = tanh(W2 h1 + b2)
    std::vector<double> h2(fHidden2Size);
    for (int i=0;i<fHidden2Size;++i) {
        double s = fBias2[i];
        for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
        h2[i] = std::tanh(s);
        if (training && (std::rand() / double(RAND_MAX)) < fDropoutRate) h2[i] = 0.0;
    }

    // y = sigmoid(W3 h2 + b3)
    double out = fBias3;
    for (int j=0;j<fHidden2Size;++j) out += fWeights3[0][j]*h2[j];
    return sigmoid(out);
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& Xb,
                                             const std::vector<int>& yb,
                                             double lr)
{
    if (Xb.empty() || Xb.size() != yb.size()) return 0.0;

    // Adam constants
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    ++fTimeStep;

    // Accumulators (same shapes)
    std::vector<std::vector<double>> gW1(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    std::vector<std::vector<double>> gW2(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    std::vector<std::vector<double>> gW3(1,            std::vector<double>(fHidden2Size, 0.0));
    std::vector<double> gB1(fHidden1Size, 0.0);
    std::vector<double> gB2(fHidden2Size, 0.0);
    double gB3 = 0.0;

    // Balanced weights (avoid collapse)
    int n = (int)Xb.size();
    int nPos = std::count(yb.begin(), yb.end(), 1);
    int nNeg = n - nPos;
    double wPos = (nPos > 0) ? 0.5 * n / double(nPos) : 0.0;
    double wNeg = (nNeg > 0) ? 0.5 * n / double(nNeg) : 0.0;

    double loss = 0.0;

    for (int s=0;s<n;++s) {
        const auto& x = Xb[s];
        const int y = yb[s];

        // forward
        std::vector<double> h1(fHidden1Size), h2(fHidden2Size);
        for (int i=0;i<fHidden1Size;++i) {
            double u = fBias1[i];
            for (int j=0;j<fInputSize;++j) u += fWeights1[i][j]*x[j];
            h1[i] = std::tanh(u);
        }
        for (int i=0;i<fHidden2Size;++i) {
            double v = fBias2[i];
            for (int j=0;j<fHidden1Size;++j) v += fWeights2[i][j]*h1[j];
            h2[i] = std::tanh(v);
        }
        double z = fBias3;
        for (int j=0;j<fHidden2Size;++j) z += fWeights3[0][j]*h2[j];
        double p = sigmoid(z);

        // class weight
        double w = (y==1 ? wPos : wNeg);

        // cross-entropy + grad
        loss += -w*( y*std::log(p+1e-8) + (1-y)*std::log(1-p+1e-8) );
        double dz = w*(p - y);

        for (int j=0;j<fHidden2Size;++j) gW3[0][j] += dz * h2[j];
        gB3 += dz;

        // backprop to h2
        std::vector<double> dh2(fHidden2Size, 0.0);
        for (int j=0;j<fHidden2Size;++j) {
            dh2[j] = fWeights3[0][j]*dz * (1 - h2[j]*h2[j]); // tanh'
        }
        for (int i=0;i<fHidden2Size;++i) {
            for (int j=0;j<fHidden1Size;++j) gW2[i][j] += dh2[i]*h1[j];
            gB2[i] += dh2[i];
        }

        // backprop to h1
        std::vector<double> dh1(fHidden1Size, 0.0);
        for (int j=0;j<fHidden1Size;++j) {
            double s2 = 0.0;
            for (int i=0;i<fHidden2Size;++i) s2 += fWeights2[i][j]*dh2[i];
            dh1[j] = s2 * (1 - h1[j]*h1[j]); // tanh'
        }
        for (int i=0;i<fHidden1Size;++i) {
            for (int j=0;j<fInputSize;++j) gW1[i][j] += dh1[i]*x[j];
            gB1[i] += dh1[i];
        }
    }

    // Adam update
    auto adam_update = [&](double& w, double& m, double& v, double g) {
        m = b1*m + (1-b1)*g;
        v = b2*v + (1-b2)*g*g;
        double mhat = m / (1 - std::pow(b1, fTimeStep));
        double vhat = v / (1 - std::pow(b2, fTimeStep));
        w -= lr * mhat / (std::sqrt(vhat) + eps);
    };

    for (int i=0;i<fHidden1Size;++i) {
        for (int j=0;j<fInputSize;++j)
            adam_update(fWeights1[i][j], fMomentum1_w1[i][j], fMomentum2_w1[i][j], gW1[i][j]/n);
        adam_update(fBias1[i], fMomentum1_b1[i], fMomentum2_b1[i], gB1[i]/n);
    }
    for (int i=0;i<fHidden2Size;++i) {
        for (int j=0;j<fHidden1Size;++j)
            adam_update(fWeights2[i][j], fMomentum1_w2[i][j], fMomentum2_w2[i][j], gW2[i][j]/n);
        adam_update(fBias2[i], fMomentum1_b2[i], fMomentum2_b2[i], gB2[i]/n);
    }
    for (int j=0;j<fHidden2Size;++j)
        adam_update(fWeights3[0][j], fMomentum1_w3[0][j], fMomentum2_w3[0][j], gW3[0][j]/n);
    adam_update(fBias3, fMomentum1_b3, fMomentum2_b3, gB3/n);

    return loss / n;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& fn)
{
    std::ofstream f(fn.c_str());
    if (!f.is_open()) {
        std::cout << "Warning: cannot open weight file for write: " << fn << "\n";
        return;
    }
    f << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";
    for (const auto& row : fWeights1) {
        for (double w : row) f << w << " ";
        f << "\n";
    }
    for (double b : fBias1) f << b << " ";
    f << "\n";
    for (const auto& row : fWeights2) {
        for (double w : row) f << w << " ";
        f << "\n";
    }
    for (double b : fBias2) f << b << " ";
    f << "\n";
    for (double w : fWeights3[0]) f << w << " ";
    f << "\n";
    f << fBias3 << "\n";
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& fn)
{
    std::ifstream f(fn.c_str());
    if (!f.is_open()) return false;

    f >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.assign(1,            std::vector<double>(fHidden2Size));
    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);

    for (auto& row : fWeights1) for (double& w : row) f >> w;
    for (double& b : fBias1) f >> b;
    for (auto& row : fWeights2) for (double& w : row) f >> w;
    for (double& b : fBias2) f >> b;
    for (double& w : fWeights3[0]) f >> w;
    f >> fBias3;

    return true;
}
void PhotonTriggerML::NeuralNetwork::QuantizeWeights() { /* placeholder */ }

// ============================================================================
// Module implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML()
: fNeuralNetwork(std::make_unique<NeuralNetwork>()),
  fIsTraining(false),               // OFF by default — safer/steadier
  fTrainingEpochs(300),
  fTrainingStep(0),
  fBestValidationLoss(1e9),
  fEpochsSinceImprovement(0),
  fEventCount(0),
  fStationCount(0),
  fPhotonLikeCount(0),
  fHadronLikeCount(0),
  fEnergy(0),
  fCoreX(0), fCoreY(0),
  fPrimaryId(0),
  fPrimaryType("Unknown"),
  fPhotonScore(0), fConfidence(0),
  fDistance(0), fStationId(0),
  fIsActualPhoton(false),
  fOutputFile(nullptr), fMLTree(nullptr),
  fLogFileName("photon_trigger_ml_physics.log"),
  hPhotonScore(nullptr), hPhotonScorePhotons(nullptr),
  hPhotonScoreHadrons(nullptr), hConfidence(nullptr),
  hRisetime(nullptr), hAsymmetry(nullptr), hKurtosis(nullptr),
  hScoreVsEnergy(nullptr), hScoreVsDistance(nullptr),
  gROCCurve(nullptr),
  hConfusionMatrix(nullptr),
  hTrainingLoss(nullptr), hValidationLoss(nullptr), hAccuracyHistory(nullptr),
  fTruePositives(0), fFalsePositives(0), fTrueNegatives(0), fFalseNegatives(0),
  fPhotonThreshold(0.72),           // starting point (auto-updated)
  fEnergyMin(1e18), fEnergyMax(1e19),
  fOutputFileName("photon_trigger_ml.root"),
  fWeightsFileName("photon_trigger_weights_physics.txt"),
  fLoadPretrainedWeights(true)      // prefer pretrained + calibrate
{
    fInstance = this;
    fFeatureMeans.assign(17, 0.0);
    fFeatureStdDevs.assign(17, 1.0);
}

PhotonTriggerML::~PhotonTriggerML() {}

// Rolling calibration buffers/state (file-scope to keep header unchanged)
namespace {
struct CalibBuffer {
    std::deque<double> s;   // calibrated scores (after Platt fusion)
    std::deque<int>    y;   // labels
    size_t capacity = 6000;
} gBuf;
PlattCalib gPlatt;              // α, β
const size_t kWarmMin = 300;    // need at least this many labeled stations
const int    kAutoScanEvery = 8;
}

// Update Platt parameters with a few Newton steps on current buffer
static void update_platt(PlattCalib& P, const CalibBuffer& B) {
    if (B.s.size() < kWarmMin) return;

    // Optimize: p = sigmoid( α*logit(s) + β )
    double a = P.alpha, b = P.beta;
    const int iters = 8;
    for (int it=0; it<iters; ++it) {
        double gA=0, gB=0, hAA=0, hAB=0, hBB=0;
        size_t n = B.s.size();
        for (size_t i=0;i<n;++i) {
            double z = a*logit(B.s[i]) + b;
            double p = sigmoid(z);
            double r = double(B.y[i]) - p;
            double w = p*(1-p);
            double l = logit(B.s[i]);
            gA += r*l;     gB += r;
            hAA += -w*l*l; hAB += -w*l; hBB += -w;
        }
        // Solve 2x2 Newton step
        double det = hAA*hBB - hAB*hAB;
        if (std::fabs(det) < 1e-12) break;
        double dA = ( gB*hAB - gA*hBB) / det;
        double dB = (-gB*hAA + gA*hAB) / det;
        a += dA; b += dB;
        if (std::fabs(dA)+std::fabs(dB) < 1e-5) break;
    }
    P.alpha = a; P.beta = b;
}

// Scan thresholds and choose the one that maximizes MCC (then F1)
static double scan_best_threshold(const CalibBuffer& B) {
    if (B.s.size() < kWarmMin) return std::numeric_limits<double>::quiet_NaN();

    int N = (int)B.s.size();
    double bestThr = 0.70, bestMCC = -1e9, bestF1 = -1e9;

    for (double thr=0.55; thr<=0.85; thr+=0.01) {
        int TP=0,FP=0,TN=0,FN=0;
        for (int i=0;i<N;++i) {
            bool pred = (B.s[i] > thr);
            int y = B.y[i];
            if      (pred && y==1) ++TP;
            else if (pred && y==0) ++FP;
            else if (!pred && y==0) ++TN;
            else ++FN;
        }
        double denom = std::sqrt( (double)(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) ) + 1e-9;
        double mcc = ((double)TP*TN - (double)FP*FN)/denom;

        double prec = (TP+FP>0)? (double)TP/(TP+FP) : 0.0;
        double rec  = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
        double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;

        if (mcc > bestMCC + 1e-12 || (std::fabs(mcc-bestMCC)<1e-12 && f1>bestF1)) {
            bestMCC = mcc; bestF1 = f1; bestThr = thr;
        }
    }
    return bestThr;
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init()");

    // log file
    fLogFile.open(fLogFileName.c_str());
    if (fLogFile.is_open()) {
        time_t now = time(nullptr);
        fLogFile << "==========================================\n";
        fLogFile << "PhotonTriggerML Physics-Based Version Log\n";
        fLogFile << "Date: " << ctime(&now);
        fLogFile << "==========================================\n\n";
    }

    // network (small + stable)
    fNeuralNetwork->Initialize(17, 24, 12);

    // optionally load pretrained
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        std::cout << "[PhotonTriggerML] loaded weights: " << fWeightsFileName << "\n";
    } else {
        std::cout << "[PhotonTriggerML] random init (online training "
                  << (fIsTraining? "ON" : "OFF") << ")\n";
    }

    // output file/tree
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Cannot create ROOT output file.");
        return eFailure;
    }

    fMLTree = new TTree("MLTree","PhotonTriggerML");
    fMLTree->Branch("eventId",        &fEventCount,   "eventId/I");
    fMLTree->Branch("stationId",      &fStationId,    "stationId/I");
    fMLTree->Branch("energy",         &fEnergy,       "energy/D");
    fMLTree->Branch("distance",       &fDistance,     "distance/D");
    fMLTree->Branch("photonScore",    &fPhotonScore,  "photonScore/D");
    fMLTree->Branch("confidence",     &fConfidence,   "confidence/D");
    fMLTree->Branch("primaryId",      &fPrimaryId,    "primaryId/I");
    fMLTree->Branch("primaryType",    &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");

    // histograms
    hPhotonScore         = new TH1D("hPhotonScore","ML Photon Score (All);score;count",50,0,1);
    hPhotonScorePhotons  = new TH1D("hPhotonScorePhotons","ML Score (true photons);score;count",50,0,1);
    hPhotonScoreHadrons  = new TH1D("hPhotonScoreHadrons","ML Score (true hadrons);score;count",50,0,1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence          = new TH1D("hConfidence","|score-0.5|;abs(score-0.5);count",50,0,0.5);
    hRisetime            = new TH1D("hRisetime","Rise time 10–90%;ns;count",50,0,1000);
    hAsymmetry           = new TH1D("hAsymmetry","Pulse asymmetry;(fall-rise)/(fall+rise);count",50,-1,1);
    hKurtosis            = new TH1D("hKurtosis","Kurtosis;value;count",50,-5,20);
    hScoreVsEnergy       = new TH2D("hScoreVsEnergy","Score vs Energy;Energy [eV];score",50,1e17,1e20,50,0,1);
    hScoreVsDistance     = new TH2D("hScoreVsDistance","Score vs Distance;Distance [m];score",50,0,3000,50,0,1);
    hConfusionMatrix     = new TH2D("hConfusionMatrix","Confusion;Pred;Actual",2,-0.5,1.5,2,-0.5,1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2,"Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2,"Photon");
    hTrainingLoss        = new TH1D("hTrainingLoss","Training loss;step;loss",10000,0,10000);
    hValidationLoss      = new TH1D("hValidationLoss","Validation loss;step;loss",10000,0,10000);
    hAccuracyHistory     = new TH1D("hAccuracyHistory","Accuracy;step;acc [%]",10000,0,10000);

    // signal handlers
    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    ++fEventCount;
    ClearMLResults();

    // event-level ground truth
    fEnergy = 0.0; fCoreX=0; fCoreY=0; fPrimaryId = 0; fPrimaryType="Unknown"; fIsActualPhoton=false;
    if (event.HasSimShower()) {
        const ShowerSimData& sh = event.GetSimShower();
        fEnergy = sh.GetEnergy();
        fPrimaryId = sh.GetPrimaryParticle();
        switch (fPrimaryId) {
            case 22:   fPrimaryType="photon"; fIsActualPhoton=true; break;
            case 2212: fPrimaryType="proton"; break;
            case 1000026056: fPrimaryType="iron"; break;
            default: fPrimaryType = (fPrimaryId>1000000000? "nucleus":"unknown");
        }
        if (sh.GetNSimCores()>0) {
            const Detector& det = Detector::GetInstance();
            Point core = sh.GetSimCore(0);
            const CoordinateSystemPtr& cs = det.GetSiteCoordinateSystem();
            fCoreX = core.GetX(cs); fCoreY = core.GetY(cs);
        }
    }

    // stations
    int stationsInEvent = 0;
    if (event.HasSEvent()) {
        const sevt::SEvent& sev = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it = sev.StationsBegin();
             it != sev.StationsEnd(); ++it) {
            ProcessStation(*it);
            ++stationsInEvent;
        }
    }

    // Every few events: calibrate Platt + threshold from buffer
    if ((fEventCount % kAutoScanEvery) == 0) {
        update_platt(gPlatt, gBuf);
        double best = scan_best_threshold(gBuf);
        if (best == best) { // not NaN
            fPhotonThreshold = best;
        }
    }

    // Progress print every 50 events
    if ((fEventCount % 50) == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1,1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2,1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1,2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2,2, fTruePositives);
    }

    // optional online training (balanced, small steps)
    if (fIsTraining && !fTrainingFeatures.empty() && (fEventCount % 10)==0) {
        const int B = std::min<int>(200, (int)fTrainingFeatures.size());
        std::vector<std::vector<double>> bx(fTrainingFeatures.begin(), fTrainingFeatures.begin()+B);
        std::vector<int> by(fTrainingLabels.begin(), fTrainingLabels.begin()+B);
        double loss = 0.0;
        for (int e=0;e<4;++e) loss += fNeuralNetwork->Train(bx, by, 0.002);
        hTrainingLoss->SetBinContent(++fTrainingStep, loss/4.0);
        if ((int)fTrainingFeatures.size() > 4000) {
            fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin()+1000);
            fTrainingLabels.erase(fTrainingLabels.begin(), fTrainingLabels.begin()+1000);
        }
    }

    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // station position → distance to core
    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdet = detector.GetSDetector();
        const sdet::Station& detStation = sdet.GetStation(fStationId);
        const CoordinateSystemPtr& cs = detector.GetSiteCoordinateSystem();
        double sx = detStation.GetPosition().GetX(cs);
        double sy = detStation.GetPosition().GetY(cs);
        fDistance = std::hypot(sx - fCoreX, sy - fCoreY);
    } catch (...) {
        fDistance = -1.0;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    for (int p=0;p<3;++p) {
        int pmtId = firstPMT + p;
        if (!station.HasPMT(pmtId)) continue;
        const sevt::PMT& pmt = station.GetPMT(pmtId);

        std::vector<double> trace;
        if (!ExtractTraceData(pmt, trace) || trace.size()!=2048) continue;

        // sanity: must have a visible pulse
        auto mm = std::minmax_element(trace.begin(), trace.end());
        if ((*mm.second - *mm.first) < 8.0) continue;

        // features
        fFeatures = ExtractEnhancedFeatures(trace);
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        UpdateFeatureStatistics(fFeatures);
        std::vector<double> X = NormalizeFeatures(fFeatures);

        // NN probability
        double ml = fNeuralNetwork->Predict(X, false);

        // simple physics score in [-1, +1]
        double phys = 0.0;
        // sharp rise + narrow width + single-peak + concentrated charge
        if (fFeatures.risetime_10_90 < 160) phys += 0.6; else phys -= 0.2;
        if (fFeatures.pulse_width    < 260) phys += 0.4; else phys -= 0.1;
        if (fFeatures.secondary_peak_ratio < 0.25) phys += 0.3; else phys -= 0.3;
        if (fFeatures.peak_charge_ratio    > 0.16) phys += 0.3; else phys -= 0.1;

        // light penalty for very wide / ragged pulses
        if (fFeatures.pulse_width > 350) phys -= 0.25;
        if (fFeatures.num_peaks   >= 5)  phys -= 0.1;

        phys = std::max(-1.0, std::min(1.0, phys));

        // Platt fusion in logit space
        double fused = sigmoid( gPlatt.alpha*logit(ml) + gPlatt.beta + 0.8*phys );
        fPhotonScore = clamp01(fused);
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // small, bounded threshold seasoning by distance
        double thr = fPhotonThreshold;
        if (fDistance >= 0) {
            if (fDistance < 500)   thr = std::min(0.88, thr + 0.03);
            else if (fDistance > 1200) thr = std::max(0.58, thr - 0.02);
        }

        // decision
        bool identifiedAsPhoton = (fPhotonScore > thr);

        // training/validation buffers (for calibration)
        bounded_push(gBuf.s, fPhotonScore, gBuf.capacity);
        bounded_push(gBuf.y, fIsActualPhoton ? 1 : 0, gBuf.capacity);

        // update confusion
        if (identifiedAsPhoton) {
            ++fPhotonLikeCount;
            if (fIsActualPhoton) ++fTruePositives;
            else                 ++fFalsePositives;
        } else {
            ++fHadronLikeCount;
            if (!fIsActualPhoton) ++fTrueNegatives;
            else                  ++fFalseNegatives;
        }

        // results map
        MLResult res;
        res.photonScore        = fPhotonScore;
        res.identifiedAsPhoton = identifiedAsPhoton;
        res.isActualPhoton     = fIsActualPhoton;
        res.vemCharge          = fFeatures.total_charge;
        res.features           = fFeatures;
        res.primaryType        = fPrimaryType;
        res.confidence         = fConfidence;
        fMLResultsMap[fStationId] = res;

        // fill hists/tree
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);
        fMLTree->Fill();

        // optional training example (balanced CE handles skew)
        if (fIsTraining) {
            fTrainingFeatures.push_back(X);
            fTrainingLabels.push_back(fIsActualPhoton ? 1 : 0);
        }

        ++fStationCount;
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& out)
{
    out.clear();
    try {
        if (pmt.HasFADCTrace()) {
            const auto& tr = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            out.insert(out.end(), &tr[0], &tr[0] + 2048);
            return true;
        }
    } catch (...) {}

    try {
        if (pmt.HasSimData()) {
            const sevt::PMTSimData& sd = pmt.GetSimData();
            if (sd.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& tr = sd.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                 sevt::StationConstants::eTotal);
                out.insert(out.end(), &tr[0], &tr[0] + 2048);
                return true;
            }
        }
    } catch (...) {}

    return false;
}

PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
    EnhancedFeatures F;
    const int N = (int)trace.size();
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    // robust baseline: median of first 200 bins
    std::vector<double> first(trace.begin(), trace.begin()+std::min(200,N));
    std::nth_element(first.begin(), first.begin()+first.size()/2, first.end());
    double baseline = first[first.size()/2];

    std::vector<double> sig(N);
    int peak_bin = 0;
    double peak_val = -1e9, total = 0.0;
    for (int i=0;i<N;++i) {
        double v = trace[i] - baseline;
        if (v < 0) v = 0;
        sig[i] = v;
        if (v > peak_val) { peak_val = v; peak_bin = i; }
        total += v;
    }
    if (peak_val < 5.0 || total < 10.0) return F;

    F.peak_amplitude = peak_val / ADC_PER_VEM;
    F.total_charge   = total    / ADC_PER_VEM;
    F.peak_charge_ratio = F.peak_amplitude / (F.total_charge + 1e-6);

    // rise/fall bins (10/50/90%)
    auto find_left  = [&](double thr)->int {
        for (int i=peak_bin;i>=0;--i) if (sig[i] <= thr) return i;
        return 0;
    };
    auto find_right = [&](double thr)->int {
        for (int i=peak_bin;i<N;++i) if (sig[i] <= thr) return i;
        return N-1;
    };

    const double v10 = 0.10*peak_val, v50 = 0.50*peak_val, v90 = 0.90*peak_val;
    int b10 = find_left(v10);
    int b50 = find_left(v50);
    int b90 = find_left(v90);
    int f90 = find_right(v90);
    int f10 = find_right(v10);

    F.risetime_10_50 = std::abs(b50 - b10)*NS_PER_BIN;
    F.risetime_10_90 = std::abs(b90 - b10)*NS_PER_BIN;
    F.falltime_90_10 = std::abs(f10 - f90)*NS_PER_BIN;

    // FWHM
    int lhm = b10, rhm = f10;
    for (int i=b10;i<=peak_bin;++i) { if (sig[i]>=v50) { lhm = i; break; } }
    for (int i=peak_bin;i<N;++i)     { if (sig[i]<=v50) { rhm = i; break; } }
    F.pulse_width = std::abs(rhm - lhm)*NS_PER_BIN;

    // asymmetry
    double rise = F.risetime_10_90, fall = F.falltime_90_10;
    F.asymmetry = (fall - rise) / (fall + rise + 1e-6);

    // moments
    double mean_t = 0.0;
    for (int i=0;i<N;++i) mean_t += i*sig[i];
    mean_t /= (total + 1e-6);
    double var=0, skew=0, kurt=0;
    for (int i=0;i<N;++i) {
        double d = i - mean_t;
        double w = sig[i] / (total + 1e-6);
        var  += d*d*w;
        skew += d*d*d*w;
        kurt += d*d*d*d*w;
    }
    double sd = std::sqrt(var + 1e-9);
    F.time_spread = sd*NS_PER_BIN;
    F.skewness    = skew / (sd*sd*sd + 1e-9);
    F.kurtosis    = kurt / (var*var + 1e-9) - 3.0;

    // early/late fractions
    int Q = N/4;
    double early=0, late=0;
    for (int i=0;i<Q;++i) early += sig[i];
    for (int i=3*Q;i<N;++i) late += sig[i];
    F.early_fraction = early / (total + 1e-6);
    F.late_fraction  = late  / (total + 1e-6);

    // smoothness (second derivative RMS) above 10% of peak
    double accum = 0.0; int cnt = 0;
    for (int i=1;i<N-1;++i) {
        if (sig[i] > v10) {
            double s2 = sig[i+1] - 2*sig[i] + sig[i-1];
            accum += s2*s2; ++cnt;
        }
    }
    F.smoothness = std::sqrt( accum / (cnt + 1) );

    // high-frequency content (simple)
    double hf = 0.0;
    for (int i=1;i<N-1;++i) {
        double d = sig[i+1] - sig[i-1];
        hf += d*d;
    }
    F.high_freq_content = hf / (total*total + 1e-6);

    // peaks
    F.num_peaks = 0;
    double secondary = 0.0;
    double thr = 0.18*peak_val;
    for (int i=1;i<N-1;++i) {
        if (sig[i]>thr && sig[i]>sig[i-1] && sig[i]>sig[i+1]) {
            ++F.num_peaks;
            if (i != peak_bin) secondary = std::max(secondary, sig[i]);
        }
    }
    F.secondary_peak_ratio = secondary / (peak_val + 1e-6);

    return F;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& f)
{
    std::vector<double> r = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
    };

    static int n = 0; ++n;
    for (size_t i=0;i<r.size();++i) {
        double delta  = r[i] - fFeatureMeans[i];
        fFeatureMeans[i] += delta / n;
        double delta2 = r[i] - fFeatureMeans[i];
        double var = fFeatureStdDevs[i]*fFeatureStdDevs[i]*(n-2) + delta*delta2;
        fFeatureStdDevs[i] = std::sqrt( (n>1) ? var/std::max(1,n-1) : 1.0 );
        if (fFeatureStdDevs[i] < 1e-3) fFeatureStdDevs[i] = 1.0;
    }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& f)
{
    std::vector<double> r = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
    };
    // min-max windows (conservative)
    std::vector<double> mn = {0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
    std::vector<double> mx = {500,1000,1000,1000,1,10,100,1,100,20,5,1,1,1000,10,10,1};

    std::vector<double> z; z.reserve(r.size());
    for (size_t i=0;i<r.size();++i) {
        double v = (r[i]-mn[i])/(mx[i]-mn[i] + 1e-6);
        v = clamp01(v);
        z.push_back(v);
    }
    return z;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 0.0;

    const int B = std::min<int>(200, (int)fTrainingFeatures.size());
    std::vector<std::vector<double>> bx(fTrainingFeatures.begin(), fTrainingFeatures.begin()+B);
    std::vector<int> by(fTrainingLabels.begin(), fTrainingLabels.begin()+B);

    double loss = fNeuralNetwork->Train(bx, by, 0.002);

    // quick validation on buffer (if available), just for plotting
    if (gBuf.s.size() >= kWarmMin) {
        double val_loss = 0.0;
        size_t n = gBuf.s.size();
        for (size_t i=0;i<n;++i) {
            double y = gBuf.y[i];
            double p = gBuf.s[i];
            val_loss += - (y*std::log(p+1e-8) + (1-y)*std::log(1-p+1e-8));
        }
        val_loss /= n;
        hValidationLoss->SetBinContent(fTrainingStep, val_loss);
    }

    return loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) return;

    double acc  = 100.0 * (fTruePositives + fTrueNegatives) / total;
    double prec = (fTruePositives + fFalsePositives > 0)
                  ? 100.0 * fTruePositives / (fTruePositives + fFalsePositives) : 0;
    double rec  = (fTruePositives + fFalseNegatives > 0)
                  ? 100.0 * fTruePositives / (fTruePositives + fFalseNegatives) : 0;
    double f1   = (prec+rec > 0) ? 2*prec*rec/(prec+rec) : 0;

    std::cout << std::fixed << std::setprecision(2)
              << "[Evt " << fEventCount << "] Acc=" << acc
              << "% Prec=" << prec << "% Rec=" << rec << "% F1=" << f1
              << "%  Thr=" << std::setprecision(3) << fPhotonThreshold << "\n";

    if (fLogFile.is_open()) {
        fLogFile << std::fixed << std::setprecision(4)
                 << "Event " << fEventCount << " - Acc:" << acc
                 << "% Prec:" << prec << "% Rec:" << rec << "% F1:" << f1 << "%\n";
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
    int total = TP+FP+TN+FN; if (total==0) return;
    double acc  = 100.0*(TP+TN)/total;
    double prec = (TP+FP>0)? 100.0*TP/(TP+FP) : 0.0;
    double rec  = (TP+FN>0)? 100.0*TP/(TP+FN) : 0.0;
    double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;

    std::cout << "Final: Acc=" << std::fixed << std::setprecision(2) << acc
              << "% Prec=" << prec << "% Rec=" << rec << "% F1=" << f1
              << "%  (Thr=" << std::setprecision(3) << fPhotonThreshold << ")\n";
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    CalculatePerformanceMetrics();

    // Save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // Save ROOT file
    if (fOutputFile) {
        fOutputFile->cd();
        if (fMLTree) fMLTree->Write();
        if (hPhotonScore)         hPhotonScore->Write();
        if (hPhotonScorePhotons)  hPhotonScorePhotons->Write();
        if (hPhotonScoreHadrons)  hPhotonScoreHadrons->Write();
        if (hConfidence)          hConfidence->Write();
        if (hRisetime)            hRisetime->Write();
        if (hAsymmetry)           hAsymmetry->Write();
        if (hKurtosis)            hKurtosis->Write();
        if (hScoreVsEnergy)       hScoreVsEnergy->Write();
        if (hScoreVsDistance)     hScoreVsDistance->Write();
        if (hConfusionMatrix)     hConfusionMatrix->Write();
        if (hTrainingLoss)        hTrainingLoss->Write();
        if (hValidationLoss)      hValidationLoss->Write();
        if (hAccuracyHistory)     hAccuracyHistory->Write();
        fOutputFile->Close();
        delete fOutputFile; fOutputFile = nullptr;
    }

    // Human-readable text summary
    std::ofstream txt("photon_trigger_summary.txt");
    if (txt.is_open()) {
        int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
        int total = TP+FP+TN+FN;
        double acc  = (total>0)? 100.0*(TP+TN)/total : 0.0;
        double prec = (TP+FP>0)? 100.0*TP/(TP+FP) : 0.0;
        double rec  = (TP+FN>0)? 100.0*TP/(TP+FN) : 0.0;
        double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;

        txt << "PhotonTriggerML Summary\n";
        txt << "BaseThreshold=" << std::setprecision(6) << fPhotonThreshold << "\n";
        txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
        txt << std::fixed << std::setprecision(2);
        txt << "Accuracy=" << acc << "%  Precision=" << prec
            << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
        txt.close();
    }

    if (fLogFile.is_open()) fLogFile.close();
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    SaveAndDisplaySummary();
    return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& result)
{
    auto it = fMLResultsMap.find(stationId);
    if (it == fMLResultsMap.end()) return false;
    result = it->second;
    return true;
}

void PhotonTriggerML::ClearMLResults() { fMLResultsMap.clear(); }

