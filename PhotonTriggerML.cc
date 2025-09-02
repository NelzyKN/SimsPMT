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
#include <TCanvas.h>
#include <TStyle.h>
#include <TPaveText.h>
#include <TLegend.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <csignal>

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

// Tell summary code we're inside a signal handler (skip anything risky)
static volatile sig_atomic_t gCalledFromSignal = 0;

// ============================================================================
// Minimal, safe signal handler
// ============================================================================
static void PhotonTriggerMLSignalHandler(int sig)
{
    if (sig == SIGINT || sig == SIGTSTP) {
        std::cout << "\n\n[PhotonTriggerML] Interrupt received. Saving summary...\n" << std::endl;
        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); }
            catch (...) {}
        }
        std::signal(sig, SIG_DFL);
        std::raise(sig);
        _Exit(0);
    }
}

// ============================================================================
// NeuralNetwork (Adam, ReLU hidden, Sigmoid output)
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork()
: fInputSize(0), fHidden1Size(0), fHidden2Size(0),
  fBias3(0.0),
  fMomentum1_b3(0.0), fMomentum2_b3(0.0),
  fTimeStep(0),
  fDropoutRate(0.1),
  fIsQuantized(false), fQuantizationScale(127.0)
{
}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize   = input_size;
    fHidden1Size = hidden1_size;
    fHidden2Size = hidden2_size;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    auto randw = [&](int fan_in){ return dist(gen) / std::sqrt(std::max(1, fan_in)); };

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    for (int i=0;i<fHidden1Size;++i)
        for (int j=0;j<fInputSize;++j) fWeights1[i][j] = randw(fInputSize);

    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    for (int i=0;i<fHidden2Size;++i)
        for (int j=0;j<fHidden1Size;++j) fWeights2[i][j] = randw(fHidden1Size);

    fWeights3.assign(1, std::vector<double>(fHidden2Size));
    for (int j=0;j<fHidden2Size;++j) fWeights3[0][j] = randw(fHidden2Size);

    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);
    fBias3 = 0.0;

    // Adam moments (m, v)
    fMomentum1_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fMomentum2_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fMomentum1_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fMomentum2_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fMomentum1_w3.assign(1, std::vector<double>(fHidden2Size, 0.0));
    fMomentum2_w3.assign(1, std::vector<double>(fHidden2Size, 0.0));
    fMomentum1_b1.assign(fHidden1Size, 0.0);
    fMomentum2_b1.assign(fHidden1Size, 0.0);
    fMomentum1_b2.assign(fHidden2Size, 0.0);
    fMomentum2_b2.assign(fHidden2Size, 0.0);
    fMomentum1_b3 = 0.0;
    fMomentum2_b3 = 0.0;

    fTimeStep = 0;

    std::cout << "NeuralNetwork initialized: "
              << fInputSize << " -> " << fHidden1Size
              << " -> " << fHidden2Size << " -> 1" << std::endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if ((int)x.size() != fInputSize) return 0.5;

    // Hidden 1
    std::vector<double> h1(fHidden1Size, 0.0);
    for (int i=0;i<fHidden1Size;++i) {
        double s = fBias1[i];
        for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
        h1[i] = ReLU(s);
    }
    // dropout (light)
    if (training && fDropoutRate>0) {
        static thread_local std::mt19937 gen(5678);
        std::uniform_real_distribution<double> u(0.0,1.0);
        for (double &v : h1) { if (u(gen) < fDropoutRate) v = 0.0; }
    }

    // Hidden 2
    std::vector<double> h2(fHidden2Size, 0.0);
    for (int i=0;i<fHidden2Size;++i) {
        double s = fBias2[i];
        for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
        h2[i] = ReLU(s);
    }
    if (training && fDropoutRate>0) {
        static thread_local std::mt19937 gen2(9876);
        std::uniform_real_distribution<double> u(0.0,1.0);
        for (double &v : h2) { if (u(gen2) < fDropoutRate) v = 0.0; }
    }

    // Output
    double o = fBias3;
    for (int j=0;j<fHidden2Size;++j) o += fWeights3[0][j]*h2[j];
    return Sigmoid(o);
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& Y,
                                             double lr)
{
    if (X.empty() || X.size()!=Y.size()) return -1.0;

    // Class-balance weights
    int n = (int)X.size();
    int n1 = std::count(Y.begin(), Y.end(), 1);
    int n0 = n - n1;
    double w1 = (n1>0) ? 0.5 / double(n1) : 0.0;
    double w0 = (n0>0) ? 0.5 / double(n0) : 0.0;

    // Accumulate gradients
    std::vector<std::vector<double>> gW1(fHidden1Size, std::vector<double>(fInputSize,0.0));
    std::vector<std::vector<double>> gW2(fHidden2Size, std::vector<double>(fHidden1Size,0.0));
    std::vector<std::vector<double>> gW3(1, std::vector<double>(fHidden2Size,0.0));
    std::vector<double> gB1(fHidden1Size,0.0), gB2(fHidden2Size,0.0);
    double gB3 = 0.0;

    double loss = 0.0;

    for (int idx=0; idx<n; ++idx) {
        const auto &x = X[idx];
        int y = Y[idx];

        // forward
        std::vector<double> h1(fHidden1Size), a1(fHidden1Size);
        for (int i=0;i<fHidden1Size;++i) {
            double s = fBias1[i];
            for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
            a1[i] = s;
            h1[i] = ReLU(s);
        }
        std::vector<double> h2(fHidden2Size), a2(fHidden2Size);
        for (int i=0;i<fHidden2Size;++i) {
            double s = fBias2[i];
            for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
            a2[i] = s;
            h2[i] = ReLU(s);
        }
        double o = fBias3;
        for (int j=0;j<fHidden2Size;++j) o += fWeights3[0][j]*h2[j];
        double p = Sigmoid(o);

        double w = (y==1) ? w1 : w0;
        // cross entropy
        loss += -w*( y*std::log(p+1e-9) + (1-y)*std::log(1-p+1e-9) );

        // backprop
        double dO = w*(p - y);              // dL/do for sigmoid + CE

        for (int j=0;j<fHidden2Size;++j) gW3[0][j] += dO * h2[j];
        gB3 += dO;

        // backprop to h2
        std::vector<double> dH2(fHidden2Size,0.0);
        for (int j=0;j<fHidden2Size;++j)
            dH2[j] = dO * fWeights3[0][j] * ReLUDerivative(a2[j]);

        for (int i=0;i<fHidden2Size;++i) {
            for (int j=0;j<fHidden1Size;++j) gW2[i][j] += dH2[i]*h1[j];
            gB2[i] += dH2[i];
        }

        // backprop to h1
        std::vector<double> dH1(fHidden1Size,0.0);
        for (int j=0;j<fHidden1Size;++j) {
            double s = 0.0;
            for (int i=0;i<fHidden2Size;++i) s += fWeights2[i][j]*dH2[i];
            dH1[j] = s * ReLUDerivative(a1[j]);
        }
        for (int i=0;i<fHidden1Size;++i) {
            for (int j=0;j<fInputSize;++j) gW1[i][j] += dH1[i]*x[j];
            gB1[i] += dH1[i];
        }
    }

    // Adam update
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps   = 1e-8;
    fTimeStep++;

    auto bias_corr = [&](double m, double v)->double {
        double mhat = m / (1.0 - std::pow(beta1, fTimeStep));
        double vhat = v / (1.0 - std::pow(beta2, fTimeStep));
        return lr * mhat / (std::sqrt(vhat) + eps);
    };

    for (int i=0;i<fHidden1Size;++i) {
        for (int j=0;j<fInputSize;++j) {
            double g = gW1[i][j] / n;
            fMomentum1_w1[i][j] = beta1*fMomentum1_w1[i][j] + (1-beta1)*g;
            fMomentum2_w1[i][j] = beta2*fMomentum2_w1[i][j] + (1-beta2)*g*g;
            fWeights1[i][j] -= bias_corr(fMomentum1_w1[i][j], fMomentum2_w1[i][j]);
        }
        double g = gB1[i] / n;
        fMomentum1_b1[i] = beta1*fMomentum1_b1[i] + (1-beta1)*g;
        fMomentum2_b1[i] = beta2*fMomentum2_b1[i] + (1-beta2)*g*g;
        fBias1[i] -= bias_corr(fMomentum1_b1[i], fMomentum2_b1[i]);
    }

    for (int i=0;i<fHidden2Size;++i) {
        for (int j=0;j<fHidden1Size;++j) {
            double g = gW2[i][j] / n;
            fMomentum1_w2[i][j] = beta1*fMomentum1_w2[i][j] + (1-beta1)*g;
            fMomentum2_w2[i][j] = beta2*fMomentum2_w2[i][j] + (1-beta2)*g*g;
            fWeights2[i][j] -= bias_corr(fMomentum1_w2[i][j], fMomentum2_w2[i][j]);
        }
        double g = gB2[i] / n;
        fMomentum1_b2[i] = beta1*fMomentum1_b2[i] + (1-beta1)*g;
        fMomentum2_b2[i] = beta2*fMomentum2_b2[i] + (1-beta2)*g*g;
        fBias2[i] -= bias_corr(fMomentum1_b2[i], fMomentum2_b2[i]);
    }

    for (int j=0;j<fHidden2Size;++j) {
        double g = gW3[0][j] / n;
        fMomentum1_w3[0][j] = beta1*fMomentum1_w3[0][j] + (1-beta1)*g;
        fMomentum2_w3[0][j] = beta2*fMomentum2_w3[0][j] + (1-beta2)*g*g;
        fWeights3[0][j] -= bias_corr(fMomentum1_w3[0][j], fMomentum2_w3[0][j]);
    }
    {
        double g = gB3 / n;
        fMomentum1_b3 = beta1*fMomentum1_b3 + (1-beta1)*g;
        fMomentum2_b3 = beta2*fMomentum2_b3 + (1-beta2)*g*g;
        fBias3 -= bias_corr(fMomentum1_b3, fMomentum2_b3);
    }

    return loss / n;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream f(filename.c_str());
    if (!f.is_open()) {
        std::cout << "Error: cannot write weights to " << filename << std::endl;
        return;
    }
    f << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";

    for (const auto& row : fWeights1) {
        for (double w : row) { f << w << " "; }
        f << "\n";
    }
    for (double b : fBias1) { f << b << " "; }
    f << "\n";

    for (const auto& row : fWeights2) {
        for (double w : row) { f << w << " "; }
        f << "\n";
    }
    for (double b : fBias2) { f << b << " "; }
    f << "\n";

    for (double w : fWeights3[0]) { f << w << " "; }
    f << "\n" << fBias3 << "\n";

    f.close();
    std::cout << "Weights saved to " << filename << std::endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream f(filename.c_str());
    if (!f.is_open()) {
        std::cout << "Warning: cannot open weights file " << filename << std::endl;
        return false;
    }
    f >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.assign(1, std::vector<double>(fHidden2Size));
    fBias1.assign(fHidden1Size,0.0);
    fBias2.assign(fHidden2Size,0.0);

    for (auto &row : fWeights1)
        for (double &w : row) f >> w;
    for (double &b : fBias1) f >> b;

    for (auto &row : fWeights2)
        for (double &w : row) f >> w;
    for (double &b : fBias2) f >> b;

    for (double &w : fWeights3[0]) f >> w;
    f >> fBias3;

    std::cout << "Weights loaded from " << filename << std::endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights()
{
    fIsQuantized = true;
}

// ============================================================================
// PhotonTriggerML
// ============================================================================

PhotonTriggerML::PhotonTriggerML()
: fNeuralNetwork(std::make_unique<NeuralNetwork>()),
  fIsTraining(true),
  fTrainingEpochs(400),
  fTrainingStep(0),
  fBestValidationLoss(1e9),
  fEpochsSinceImprovement(0),
  fEventCount(0), fStationCount(0),
  fPhotonLikeCount(0), fHadronLikeCount(0),
  fEnergy(0), fCoreX(0), fCoreY(0),
  fPrimaryId(0), fPrimaryType("Unknown"),
  fPhotonScore(0), fConfidence(0),
  fDistance(0), fStationId(0), fIsActualPhoton(false),
  fOutputFile(nullptr), fMLTree(nullptr),
  fLogFileName("photon_trigger_ml_physics.log"),
  hPhotonScore(nullptr), hPhotonScorePhotons(nullptr), hPhotonScoreHadrons(nullptr),
  hConfidence(nullptr), hRisetime(nullptr), hAsymmetry(nullptr), hKurtosis(nullptr),
  hScoreVsEnergy(nullptr), hScoreVsDistance(nullptr), gROCCurve(nullptr),
  hConfusionMatrix(nullptr), hTrainingLoss(nullptr), hValidationLoss(nullptr), hAccuracyHistory(nullptr),
  fTruePositives(0), fFalsePositives(0), fTrueNegatives(0), fFalseNegatives(0),
  fPhotonThreshold(0.65),              // XML default  (see PhotonTriggerML.xml.in)
  fEnergyMin(1e18), fEnergyMax(1e19),
  fOutputFileName("photon_trigger_ml.root"),
  fWeightsFileName("photon_trigger_weights_physics.txt"),
  fLoadPretrainedWeights(false)
{
    fInstance = this;
    fFeatureMeans.assign(17, 0.0);
    fFeatureStdDevs.assign(17, 1.0);
}

PhotonTriggerML::~PhotonTriggerML() { }

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init()");

    // Logs
    fLogFile.open(fLogFileName.c_str());
    if (fLogFile.is_open()) {
        time_t now = time(0);
        fLogFile << "==========================================" << std::endl;
        fLogFile << "PhotonTriggerML Physics-Based Version Log" << std::endl;
        fLogFile << "Date: " << ctime(&now);
        fLogFile << "==========================================" << std::endl << std::endl;
    }

    // NN
    fNeuralNetwork->Initialize(17, 24, 12);
    if (fLoadPretrainedWeights) {
        fNeuralNetwork->LoadWeights(fWeightsFileName);
    }

    // ROOT file / tree
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Cannot create output ROOT file");
        return eFailure;
    }
    fMLTree = new TTree("MLTree", "PhotonTriggerML");
    fMLTree->Branch("eventId",        &fEventCount,     "eventId/I");
    fMLTree->Branch("stationId",      &fStationId,      "stationId/I");
    fMLTree->Branch("energy",         &fEnergy,         "energy/D");
    fMLTree->Branch("distance",       &fDistance,       "distance/D");
    fMLTree->Branch("photonScore",    &fPhotonScore,    "photonScore/D");
    fMLTree->Branch("confidence",     &fConfidence,     "confidence/D");
    fMLTree->Branch("primaryId",      &fPrimaryId,      "primaryId/I");
    fMLTree->Branch("primaryType",    &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");

    // Histograms
    hPhotonScore          = new TH1D("hPhotonScore", "ML Score (all);score;count", 50, 0, 1);
    hPhotonScorePhotons   = new TH1D("hPhotonScorePhotons", "ML Score (true photons);score;count", 50, 0, 1);
    hPhotonScoreHadrons   = new TH1D("hPhotonScoreHadrons", "ML Score (true hadrons);score;count", 50, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence           = new TH1D("hConfidence", "Confidence |score-0.5|;|score-0.5|;count", 50, 0, 0.5);
    hRisetime             = new TH1D("hRisetime",   "Rise time 10-90% [ns];ns;count", 60, 0, 600);
    hAsymmetry            = new TH1D("hAsymmetry",  "Pulse asymmetry; (fall-rise)/(fall+rise);count", 50, -1, 1);
    hKurtosis             = new TH1D("hKurtosis",   "Kurtosis;Kurtosis;count", 60, -5, 25);
    hScoreVsEnergy        = new TH2D("hScoreVsEnergy",   "Score vs Energy;E [eV];score", 50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance      = new TH2D("hScoreVsDistance", "Score vs Distance;R [m];score", 50, 0, 3000, 50, 0, 1);
    hConfusionMatrix      = new TH2D("hConfusionMatrix", "Confusion;Pred;True", 2,-0.5,1.5,2,-0.5,1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2,"Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2,"Photon");
    hTrainingLoss   = new TH1D("hTrainingLoss",   "Training loss;step;loss",    10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation loss;step;loss",  10000, 0, 10000);
    hAccuracyHistory= new TH1D("hAccuracyHistory","Accuracy [%];step;acc",      10000, 0, 10000);

    // Signals
    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    // Shower info
    fEnergy = 0.0; fCoreX = 0.0; fCoreY = 0.0;
    fPrimaryId = 0; fPrimaryType = "Unknown"; fIsActualPhoton = false;

    if (event.HasSimShower()) {
        const ShowerSimData& sh = event.GetSimShower();
        fEnergy     = sh.GetEnergy();
        fPrimaryId  = sh.GetPrimaryParticle();
        switch (fPrimaryId) {
            case 22:    fPrimaryType = "photon"; fIsActualPhoton = true; break;
            case 2212:  fPrimaryType = "proton"; break;
            case 1000026056: fPrimaryType = "iron"; break;
            default:    fPrimaryType = (fPrimaryId > 1000000000) ? "nucleus" : "unknown"; break;
        }
        fParticleTypeCounts[fPrimaryType]++;

        if (sh.GetNSimCores()>0) {
            const Detector& det = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
            Point core = sh.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }
    }

    // Stations
    int stationsInEvent = 0;
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
            stationsInEvent++;
        }
    }

    // Update confusion matrix hist every 10 events
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1,1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2,1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1,2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2,2, fTruePositives);
    }

    // Train step (whenever we have enough)
    if (fIsTraining && fTrainingFeatures.size() >= 128 && (fEventCount % 5 == 0)) {
        (void)TrainNetwork(); // hist updates handled inside TrainNetwork()
    }

    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // Distance from core
    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        double sx = detStation.GetPosition().GetX(siteCS);
        double sy = detStation.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((sx-fCoreX)*(sx-fCoreX) + (sy-fCoreY)*(sy-fCoreY));
    } catch (...) {
        fDistance = -1; // skip if geometry not available
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();
    for (int p=0; p<3; ++p) {
        int pmtId = firstPMT + p;
        if (!station.HasPMT(pmtId)) continue;

        const sevt::PMT& pmt = station.GetPMT(pmtId);

        // Read 2048-bin FADC trace
        std::vector<double> tr;
        if (!ExtractTraceData(pmt, tr) || tr.size()!=2048) continue;

        // Ignore noise-only traces
        auto [minIt, maxIt] = std::minmax_element(tr.begin(), tr.end());
        if ((*maxIt - *minIt) < 8.0) continue;

        // Feature extraction
        fFeatures = ExtractEnhancedFeatures(tr);
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);

        UpdateFeatureStatistics(fFeatures);
        std::vector<double> X = NormalizeFeatures(fFeatures);

        // ML prediction
        double ml = fNeuralNetwork->Predict(X, false);

        // Conservative physics-based nudges (small, bounded)
        double phys_adj = 0.0;
        const double tot_ns = fFeatures.risetime_10_90 + fFeatures.falltime_90_10;
        if (fFeatures.secondary_peak_ratio < 0.20) phys_adj += 0.02;
        if (fFeatures.peak_charge_ratio     > 0.18) phys_adj += 0.02;
        if (tot_ns > 220.0)                           phys_adj += 0.02;
        if (fFeatures.num_peaks >= 3)                phys_adj -= 0.03;

        double score = ml + phys_adj;
        if (score < 0.0) score = 0.0;
        if (score > 1.0) score = 1.0;
        fPhotonScore = score;
        fConfidence  = std::fabs(score - 0.5);

        // Distance/TOT-aware threshold (bounded)
        double thr = fPhotonThreshold;
        if (fDistance >= 1200.0) thr -= 0.03;
        if (fDistance <= 600.0)  thr += 0.02;

        if (tot_ns >= 230.0) thr -= 0.02;
        if (tot_ns >= 280.0) thr -= 0.03;
        if (tot_ns >= 350.0) thr -= 0.04;

        if (fFeatures.num_peaks >= 3) thr += 0.03;
        if (fFeatures.secondary_peak_ratio >= 0.35) thr += 0.02;

        if (thr < 0.60) thr = 0.60;
        if (thr > 0.82) thr = 0.82;

        bool isPhotonLike = (score > thr);

        // Store result for other modules
        MLResult res;
        res.photonScore = score;
        res.identifiedAsPhoton = isPhotonLike;
        res.isActualPhoton = fIsActualPhoton;
        res.vemCharge = fFeatures.total_charge;
        res.features  = fFeatures;
        res.primaryType = fPrimaryType;
        res.confidence = fConfidence;
        fMLResultsMap[fStationId] = res;

        // Update counters
        if (isPhotonLike) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
        }

        // Fill hists & tree
        hPhotonScore->Fill(score);
        (fIsActualPhoton? hPhotonScorePhotons : hPhotonScoreHadrons)->Fill(score);
        hScoreVsEnergy->Fill(fEnergy, score);
        hScoreVsDistance->Fill(fDistance, score);
        if (fMLTree) fMLTree->Fill();

        // Training caches (balanced later in TrainNetwork)
        if (fIsTraining) {
            fTrainingFeatures.push_back(X);
            fTrainingLabels.push_back(fIsActualPhoton ? 1 : 0);
            // Hold out some for validation
            if ((int)fValidationFeatures.size() < 200) {
                fValidationFeatures.push_back(X);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }

        fStationCount++;
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& out)
{
    out.clear();

    // Try direct FADC
    if (pmt.HasFADCTrace()) {
        try {
            const auto& tr = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            out.reserve(2048);
            for (int i=0;i<2048;++i) out.push_back(tr[i]);
            return true;
        } catch (...) {}
    }

    // Try simulation path
    if (pmt.HasSimData()) {
        try {
            const sevt::PMTSimData& sim = pmt.GetSimData();
            if (sim.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& tr = sim.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                  sevt::StationConstants::eTotal);
                out.reserve(2048);
                for (int i=0;i<2048;++i) out.push_back(tr[i]);
                return true;
            }
        } catch (...) {}
    }

    return false;
}

PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double /*baseline*/)
{
    EnhancedFeatures F;
    const int N = (int)trace.size();
    if (N==0) return F;

    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    // Baseline = average of first 100 bins
    double base = 0.0;
    int baseN = std::min(100, N);
    for (int i=0;i<baseN;++i) base += trace[i];
    base /= std::max(1, baseN);

    // Signal & peak
    std::vector<double> sig(N,0.0);
    double peak = 0.0;
    int    ibpk = 0;
    double total = 0.0;

    for (int i=0;i<N;++i) {
        double v = trace[i] - base;
        if (v < 0) v = 0;
        sig[i] = v;
        if (v > peak) { peak = v; ibpk = i; }
        total += v;
    }
    if (peak <= 0.0 || total <= 0.0) return F;

    // Basic
    F.peak_amplitude = peak / ADC_PER_VEM;
    F.total_charge   = total / ADC_PER_VEM;
    F.peak_charge_ratio = F.peak_amplitude / (F.total_charge + 1e-6);

    // Percent levels
    const double v10 = 0.10 * peak;
    const double v50 = 0.50 * peak;
    const double v90 = 0.90 * peak;

    int i10r=0, i50r=0, i90r=ibpk;
    for (int i=ibpk; i>=0; --i) {
        if (sig[i] <= v90) i90r = i;
        if (sig[i] <= v50 && i50r==0) i50r = i;
        if (sig[i] <= v10) { i10r = i; break; }
    }
    int i90f=ibpk, i10f=N-1;
    for (int i=ibpk; i<N; ++i) {
        if (sig[i] <= v90) i90f = i;
        if (sig[i] <= v10) { i10f = i; break; }
    }

    F.risetime_10_50 = std::abs(i50r - i10r) * NS_PER_BIN;
    F.risetime_10_90 = std::abs(i90r - i10r) * NS_PER_BIN;
    F.falltime_90_10 = std::abs(i10f - i90f) * NS_PER_BIN;

    // FWHM
    double half = 0.5 * peak;
    int iL=i10r, iR=i10f;
    for (int i=i10r; i<=ibpk; ++i) { if (sig[i] >= half) { iL = i; break; } }
    for (int i=ibpk;  i<N;   ++i) { if (sig[i] <= half) { iR = i; break; } }
    F.pulse_width = std::abs(iR - iL) * NS_PER_BIN;

    // Asymmetry
    const double rise = F.risetime_10_90;
    const double fall = F.falltime_90_10;
    F.asymmetry = (fall - rise) / (fall + rise + 1e-6);

    // Moments (time index in bins)
    double mu = 0.0;
    for (int i=0;i<N;++i) mu += i * sig[i];
    mu /= (total + 1e-6);

    double var=0.0, skew=0.0, kurt=0.0;
    for (int i=0;i<N;++i) {
        double d = (i - mu);
        double w = sig[i] / (total + 1e-6);
        var  += d*d*w;
        skew += d*d*d*w;
        kurt += d*d*d*d*w;
    }
    double sd = std::sqrt(var + 1e-9);
    F.time_spread = sd * NS_PER_BIN;
    F.skewness    = skew / (sd*sd*sd + 1e-9);
    F.kurtosis    = kurt / (var*var + 1e-9) - 3.0;

    // Early/late fractions
    int Q = N/4;
    double early=0.0, late=0.0;
    for (int i=0;i<Q; ++i) early += sig[i];
    for (int i=3*Q; i<N; ++i) late  += sig[i];
    F.early_fraction = early / (total + 1e-6);
    F.late_fraction  = late  / (total + 1e-6);

    // Smoothness & high-frequency content
    double s2=0.0; int cnt=0;
    for (int i=1;i<N-1;++i) {
        if (sig[i] > 0.1*peak) {
            double sec = sig[i+1] - 2*sig[i] + sig[i-1];
            s2 += sec*sec; cnt++;
        }
    }
    F.smoothness = std::sqrt( s2 / (cnt + 1) );

    double hf=0.0;
    for (int i=1;i<N-1;++i) {
        double df = sig[i+1] - sig[i-1];
        hf += df*df;
    }
    F.high_freq_content = hf / (total*total + 1e-9);

    // Peaks
    F.num_peaks = 0;
    double second_pk = 0.0;
    double thr = 0.15 * peak;
    for (int i=1;i<N-1;++i) {
        if (sig[i] > thr && sig[i] > sig[i-1] && sig[i] > sig[i+1]) {
            F.num_peaks++;
            if (i!=ibpk && sig[i] > second_pk) second_pk = sig[i];
        }
    }
    F.secondary_peak_ratio = second_pk / (peak + 1e-9);

    return F;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& f)
{
    // Welford running mean/std
    static long long n = 0;
    static std::vector<double> M(17,0.0), S(17,0.0);

    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
    };

    n++;
    for (size_t i=0;i<raw.size();++i) {
        double x = raw[i];
        double d = x - M[i];
        M[i] += d / n;
        S[i] += d * (x - M[i]);
        fFeatureMeans[i] = M[i];
        fFeatureStdDevs[i] = (n>1) ? std::sqrt( S[i]/(n-1) ) : 1.0;
        if (fFeatureStdDevs[i] < 1e-3) fFeatureStdDevs[i] = 1.0;
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

    std::vector<double> z; z.reserve(raw.size());
    for (size_t i=0;i<raw.size();++i) {
        double mu  = (i<fFeatureMeans.size())   ? fFeatureMeans[i]   : 0.0;
        double sig = (i<fFeatureStdDevs.size()) ? fFeatureStdDevs[i] : 1.0;
        double v = (raw[i] - mu) / (sig > 1e-6 ? sig : 1.0);
        // clip to +/- 5 sigma and scale to [0,1]
        if (v < -5.0) v = -5.0;
        if (v >  5.0) v =  5.0;
        v = 0.5 + 0.1 * v;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        z.push_back(v);
    }
    return z;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.size() < 64) return 1e9;

    // Build a balanced mini-batch (up to 128 total)
    std::vector<int> idx_ph, idx_hd;
    for (size_t i=0;i<fTrainingLabels.size();++i) {
        if (fTrainingLabels[i]==1) idx_ph.push_back((int)i);
        else                       idx_hd.push_back((int)i);
    }
    if (idx_ph.empty() || idx_hd.empty()) return 1e9;

    std::shuffle(idx_ph.begin(), idx_ph.end(), std::mt19937(42+fTrainingStep));
    std::shuffle(idx_hd.begin(), idx_hd.end(), std::mt19937(84+fTrainingStep));
    int k = std::min<int>(64, std::min(idx_ph.size(), idx_hd.size()));

    std::vector<std::vector<double>> Bx;
    std::vector<int> By;
    Bx.reserve(2*k); By.reserve(2*k);
    for (int i=0;i<k;++i) {
        Bx.push_back(fTrainingFeatures[idx_ph[i]]); By.push_back(1);
        Bx.push_back(fTrainingFeatures[idx_hd[i]]); By.push_back(0);
    }

    const double lr = 0.005;
    double train_loss = fNeuralNetwork->Train(Bx, By, lr);

    // --- Hist & validation bookkeeping happen here to avoid unused-variable warnings ---
    int step = (fEventCount>0) ? (fEventCount/5) : fTrainingStep;
    if (hTrainingLoss) hTrainingLoss->SetBinContent(step, train_loss);

    // Validation
    double vloss = 0.0;
    if (!fValidationFeatures.empty()) {
        int correct=0; int tot=0;
        for (size_t i=0;i<fValidationFeatures.size();++i) {
            double p = fNeuralNetwork->Predict(fValidationFeatures[i], false);
            int y = fValidationLabels[i];
            vloss += - ( y*std::log(p+1e-9) + (1-y)*std::log(1-p+1e-9) );
            bool pred = (p > fPhotonThreshold);
            if ((pred && y==1) || (!pred && y==0)) correct++;
            tot++;
        }
        vloss /= std::max(1,(int)fValidationFeatures.size());
        double acc = (tot>0) ? 100.0*correct/tot : 0.0;
        if (hValidationLoss)  hValidationLoss->SetBinContent(step, vloss);
        if (hAccuracyHistory) hAccuracyHistory->SetBinContent(step, acc);
    }

    if (vloss < fBestValidationLoss) {
        fBestValidationLoss = vloss;
        fEpochsSinceImprovement = 0;
        fNeuralNetwork->SaveWeights("best_" + fWeightsFileName);
    } else {
        fEpochsSinceImprovement++;
    }

    // Trim caches (keep last ~400)
    if (fTrainingFeatures.size() > 800) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(),
                                fTrainingFeatures.begin()+200);
        fTrainingLabels.erase(fTrainingLabels.begin(),
                              fTrainingLabels.begin()+200);
    }

    fTrainingStep++;
    return vloss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
    int total = TP+FP+TN+FN;
    if (total<=0) return;

    double acc = 100.0*(TP+TN)/total;
    double prec= (TP+FP>0)? 100.0*TP/(TP+FP):0.0;
    double rec = (TP+FN>0)? 100.0*TP/(TP+FN):0.0;
    double f1  = (prec+rec>0)? (2*prec*rec/(prec+rec)):0.0;
    double fracPhoton = (fPhotonLikeCount+fHadronLikeCount>0)?
                         100.0*fPhotonLikeCount/(fPhotonLikeCount+fHadronLikeCount):0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << fracPhoton << "%"
              << " │ " << std::setw(7) << acc << "%"
              << " │ " << std::setw(8) << prec << "%"
              << " │ " << std::setw(7) << rec << "%"
              << " │ " << std::setw(7) << f1 << "%│" << std::endl;

    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount
                 << " - Acc:" << acc
                 << "% Prec:" << prec
                 << "% Rec:" << rec
                 << "% F1:" << f1 << "%" << std::endl;
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
    int total = TP+FP+TN+FN;
    if (total==0) {
        std::cout << "No predictions." << std::endl;
        return;
    }
    double acc = 100.0*(TP+TN)/total;
    double prec= (TP+FP>0)? 100.0*TP/(TP+FP):0.0;
    double rec = (TP+FN>0)? 100.0*TP/(TP+FN):0.0;
    double f1  = (prec+rec>0)? (2*prec*rec/(prec+rec)):0.0;

    std::cout << "Accuracy: "  << std::fixed << std::setprecision(1) << acc  << "%\n"
              << "Precision: " << prec << "%\n"
              << "Recall:    " << rec  << "%\n"
              << "F1-Score:  " << f1   << "%\n" << std::endl;

    std::cout << "Confusion matrix (Pred columns: H, P):\n";
    std::cout << "True H: " << TN << "  " << FP << "\n";
    std::cout << "True P: " << FN << "  " << TP << "\n";
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    std::cout << "\n==========================================\n";
    std::cout << "PHOTONTRIGGERML FINAL SUMMARY\n";
    std::cout << "==========================================\n";
    std::cout << "Events: " << fEventCount << "  Stations: " << fStationCount << "\n";
    CalculatePerformanceMetrics();

    // Save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // Save ROOT
    if (fOutputFile) {
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

        fOutputFile->Write();
        fOutputFile->Close();
        delete fOutputFile; fOutputFile = nullptr;
    }

    // Text summary
    {
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
            int total = TP+FP+TN+FN;
            double acc = (total>0)? 100.0*(TP+TN)/total : 0.0;
            double prec= (TP+FP>0)? 100.0*TP/(TP+FP):0.0;
            double rec = (TP+FN>0)? 100.0*TP/(TP+FN):0.0;
            double f1  = (prec+rec>0)? (2*prec*rec/(prec+rec)):0.0;

            txt << "PhotonTriggerML Summary\n";
            txt << "BaseThreshold=" << fPhotonThreshold << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy=" << acc << "%  Precision=" << prec
                << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
        }
    }

    if (fLogFile.is_open()) fLogFile.close();
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish()");
    SaveAndDisplaySummary();
    return eSuccess;
}

// Static helpers
bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& result)
{
    auto it = fMLResultsMap.find(stationId);
    if (it==fMLResultsMap.end()) return false;
    result = it->second;
    return true;
}
void PhotonTriggerML::ClearMLResults() { fMLResultsMap.clear(); }

