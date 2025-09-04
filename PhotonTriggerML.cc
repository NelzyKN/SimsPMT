// PhotonTriggerML.cc
//
// Drop-in replacement implementing:
//  - Legacy logs & summaries (unchanged format).
//  - Extra diagnostics: per-class feature histograms + TSV stats.
//  - Safer, warning-clean code (no misleading indent, no unused vars).
//  - Balanced training + randomized mini-batches.
//  - Validation-driven threshold re-calibration.
//
// No header changes required.
//

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

// ---------------------------------------------------------------------------------------------------------------------
// Using
// ---------------------------------------------------------------------------------------------------------------------
using namespace std;
using namespace fwk;
using namespace evt;
using namespace det;
using namespace utl;

// ---------------------------------------------------------------------------------------------------------------------
// Static globals (file-scope; no header changes needed)
// ---------------------------------------------------------------------------------------------------------------------

// Instance pointer for signal handler
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

// Map: station id -> last ML result
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Signal guard to avoid touching external ROOT files from handler
static volatile sig_atomic_t gCalledFromSignal = 0;

// Feature names (17 features – order must match ExtractEnhancedFeatures/NormalizeFeatures)
static const char* kFeatName[17] = {
    "rt10_50", "rt10_90", "ft90_10", "fwhm",
    "asym", "peakA", "charge", "peakQfrac",
    "smooth", "kurt", "skew", "earlyF",
    "lateF", "tSpread", "hiFreq", "nPeaks",
    "secPeakFrac"
};

// Raw feature ranges used to build histograms (same ranges as min/max normalization)
static const double kFeatMin[17] = {0,   0,    0,    0,   -1, 0,   0,   0,   0,  -5, -5, 0, 0, 0, 0, 0, 0};
static const double kFeatMax[17] = {500, 1000, 1000, 1000, 1,  10,  200, 1,   100, 20, 5,  1, 1, 1000, 10, 10, 1};

// Per-class feature histograms & running stats (file-static so no header edits)
static TH1D* gFeatPhoton[17] = {nullptr};
static TH1D* gFeatHadron[17] = {nullptr};

// Running stats for TSV summary: count/sum/sumsq per feature per class
static long long gCountPhoton = 0;
static long long gCountHadron = 0;
static double gSumPhoton[17]  = {0.0};
static double gSumSqPhoton[17]= {0.0};
static double gSumHadron[17]  = {0.0};
static double gSumSqHadron[17]= {0.0};

// Scores TSV writer (file-static; always flushed)
static std::ofstream gScoresTSV;

// ---------------------------------------------------------------------------------------------------------------------
// Signal handler – emits summary without touching external PMT-trace ROOT files
// ---------------------------------------------------------------------------------------------------------------------
static void PhotonTriggerMLSignalHandler(int sig)
{
    if (sig == SIGINT || sig == SIGTSTP) {
        std::cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << std::endl;
        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); }
            catch (...) { /* best effort */ }
        }
        std::signal(sig, SIG_DFL);
        std::raise(sig);
        _Exit(0);
    }
}

// =====================================================================================================================
// Physics-Based Neural Network (small MLP) – unchanged API; safer/balanced training inside Train()
// =====================================================================================================================
PhotonTriggerML::NeuralNetwork::NeuralNetwork()
: fInputSize(0), fHidden1Size(0), fHidden2Size(0),
  fTimeStep(0), fDropoutRate(0.1),
  fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int h1, int h2)
{
    fInputSize    = input_size;
    fHidden1Size  = h1;
    fHidden2Size  = h2;

    std::cout << "Initializing Physics-Based Neural Network: "
              << input_size << " -> " << h1 << " -> " << h2 << " -> 1" << std::endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.assign(h1, std::vector<double>(input_size, 0.0));
    for (int i = 0; i < h1; ++i)
        for (int j = 0; j < input_size; ++j)
            fWeights1[i][j] = dist(gen) / std::sqrt((double)input_size);

    fWeights2.assign(h2, std::vector<double>(h1, 0.0));
    for (int i = 0; i < h2; ++i)
        for (int j = 0; j < h1; ++j)
            fWeights2[i][j] = dist(gen) / std::sqrt((double)h1);

    fWeights3.assign(1, std::vector<double>(h2, 0.0));
    for (int j = 0; j < h2; ++j) fWeights3[0][j] = dist(gen) / std::sqrt((double)h2);

    fBias1.assign(h1, 0.0);
    fBias2.assign(h2, 0.0);
    fBias3 = 0.0;

    // Momentums
    fMomentum1_w1.assign(h1, std::vector<double>(input_size, 0.0));
    fMomentum2_w1.assign(h1, std::vector<double>(input_size, 0.0));
    fMomentum1_w2.assign(h2, std::vector<double>(h1, 0.0));
    fMomentum2_w2.assign(h2, std::vector<double>(h1, 0.0));
    fMomentum1_w3.assign(1, std::vector<double>(h2, 0.0));
    fMomentum2_w3.assign(1, std::vector<double>(h2, 0.0));

    fMomentum1_b1.assign(h1, 0.0);
    fMomentum2_b1.assign(h1, 0.0);
    fMomentum1_b2.assign(h2, 0.0);
    fMomentum2_b2.assign(h2, 0.0);
    fMomentum1_b3 = 0.0;
    fMomentum2_b3 = 0.0;

    fTimeStep = 0;
    std::cout << "Neural Network initialized." << std::endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if ((int)x.size() != fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size, 0.0);
    for (int i = 0; i < fHidden1Size; ++i) {
        double s = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * x[j];
        double a = 1.0 / (1.0 + std::exp(-s));
        if (training) {
            double r = rand() / double(RAND_MAX);
            if (r < fDropoutRate) a = 0.0;
        }
        h1[i] = a;
    }

    std::vector<double> h2(fHidden2Size, 0.0);
    for (int i = 0; i < fHidden2Size; ++i) {
        double s = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
        double a = 1.0 / (1.0 + std::exp(-s));
        if (training) {
            double r = rand() / double(RAND_MAX);
            if (r < fDropoutRate) a = 0.0;
        }
        h2[i] = a;
    }

    double o = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) o += fWeights3[0][j] * h2[j];
    return 1.0 / (1.0 + std::exp(-o));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& Y,
                                             double lr)
{
    const int N = (int)X.size();
    if (N == 0 || (int)Y.size() != N) return -1.0;

    // Balanced class weights (avoid all-hadron or all-photon collapse)
    int n_ph = std::count(Y.begin(), Y.end(), 1);
    int n_hd = N - n_ph;
    double w_ph = (n_ph > 0) ? 0.5 * N / (double)n_ph : 1.0;
    double w_hd = (n_hd > 0) ? 0.5 * N / (double)n_hd : 1.0;

    std::vector<std::vector<double>> gw1(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    std::vector<std::vector<double>> gw2(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    std::vector<std::vector<double>> gw3(1, std::vector<double>(fHidden2Size, 0.0));
    std::vector<double> gb1(fHidden1Size, 0.0), gb2(fHidden2Size, 0.0);
    double gb3 = 0.0;

    double totalLoss = 0.0;

    for (int k = 0; k < N; ++k) {
        const std::vector<double>& x = X[k];
        const int y = Y[k];
        const double wt = (y == 1) ? w_ph : w_hd;

        // Forward
        std::vector<double> h1(fHidden1Size, 0.0), h1r(fHidden1Size, 0.0);
        for (int i = 0; i < fHidden1Size; ++i) {
            double s = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * x[j];
            h1r[i] = s;
            h1[i]  = 1.0 / (1.0 + std::exp(-s));
        }
        std::vector<double> h2(fHidden2Size, 0.0), h2r(fHidden2Size, 0.0);
        for (int i = 0; i < fHidden2Size; ++i) {
            double s = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
            h2r[i] = s;
            h2[i]  = 1.0 / (1.0 + std::exp(-s));
        }
        double oraw = fBias3;
        for (int j = 0; j < fHidden2Size; ++j) oraw += fWeights3[0][j] * h2[j];
        const double o = 1.0 / (1.0 + std::exp(-oraw));

        // Loss (X-entropy)
        totalLoss += -wt * (y * std::log(o + 1e-7) + (1 - y) * std::log(1 - o + 1e-7));

        // Gradients
        const double go = wt * (o - y);     // dL/d(oraw)
        for (int j = 0; j < fHidden2Size; ++j) gw3[0][j] += go * h2[j];
        gb3 += go;

        std::vector<double> gh2(fHidden2Size, 0.0);
        for (int i = 0; i < fHidden2Size; ++i) {
            double t = fWeights3[0][i] * go;
            t *= h2[i] * (1.0 - h2[i]);     // sigmoid'
            gh2[i] = t;
        }
        for (int i = 0; i < fHidden2Size; ++i) {
            for (int j = 0; j < fHidden1Size; ++j) gw2[i][j] += gh2[i] * h1[j];
            gb2[i] += gh2[i];
        }

        std::vector<double> gh1(fHidden1Size, 0.0);
        for (int j = 0; j < fHidden1Size; ++j) {
            double t = 0.0;
            for (int i = 0; i < fHidden2Size; ++i) t += fWeights2[i][j] * gh2[i];
            t *= h1[j] * (1.0 - h1[j]);
            gh1[j] = t;
        }
        for (int i = 0; i < fHidden1Size; ++i) {
            for (int j = 0; j < fInputSize; ++j) gw1[i][j] += gh1[i] * x[j];
            gb1[i] += gh1[i];
        }
    }

    // SGD with momentum
    fTimeStep++;
    const double mom = 0.9;
    for (int i = 0; i < fHidden1Size; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            const double g = gw1[i][j] / N;
            fMomentum1_w1[i][j] = mom * fMomentum1_w1[i][j] - lr * g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        const double g = gb1[i] / N;
        fMomentum1_b1[i] = mom * fMomentum1_b1[i] - lr * g;
        fBias1[i] += fMomentum1_b1[i];
    }
    for (int i = 0; i < fHidden2Size; ++i) {
        for (int j = 0; j < fHidden1Size; ++j) {
            const double g = gw2[i][j] / N;
            fMomentum1_w2[i][j] = mom * fMomentum1_w2[i][j] - lr * g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        const double g = gb2[i] / N;
        fMomentum1_b2[i] = mom * fMomentum1_b2[i] - lr * g;
        fBias2[i] += fMomentum1_b2[i];
    }
    for (int j = 0; j < fHidden2Size; ++j) {
        const double g = gw3[0][j] / N;
        fMomentum1_w3[0][j] = mom * fMomentum1_w3[0][j] - lr * g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }
    const double g = gb3 / N;
    fMomentum1_b3 = mom * fMomentum1_b3 - lr * g;
    fBias3 += fMomentum1_b3;

    return totalLoss / N;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& fn)
{
    std::ofstream file(fn.c_str());
    if (!file.is_open()) {
        std::cout << "Error: could not save weights to " << fn << std::endl;
        return;
    }
    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";

    for (const auto& row : fWeights1) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias1) file << b << " ";
    file << "\n";

    for (const auto& row : fWeights2) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias2) file << b << " ";
    file << "\n";

    for (double w : fWeights3[0]) file << w << " ";
    file << "\n" << fBias3 << "\n";

    file.close();
    std::cout << "Weights saved to " << fn << std::endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& fn)
{
    std::ifstream file(fn.c_str());
    if (!file.is_open()) {
        std::cout << "Warning: could not load weights from " << fn << std::endl;
        return false;
    }
    file >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fWeights3.assign(1, std::vector<double>(fHidden2Size, 0.0));
    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);

    for (auto& row : fWeights1) for (double& w : row) file >> w;
    for (double& b : fBias1) file >> b;
    for (auto& row : fWeights2) for (double& w : row) file >> w;
    for (double& b : fBias2) file >> b;
    for (double& w : fWeights3[0]) file >> w;
    file >> fBias3;

    file.close();
    std::cout << "Weights loaded from " << fn << std::endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights()
{
    fIsQuantized = true; // placeholder
}

// =====================================================================================================================
// PhotonTriggerML – constructor/init
// =====================================================================================================================
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
  fCoreX(0),
  fCoreY(0),
  fPrimaryId(0),
  fPrimaryType("Unknown"),
  fPhotonScore(0),
  fConfidence(0),
  fDistance(0),
  fStationId(0),
  fIsActualPhoton(false),
  fOutputFile(nullptr),
  fMLTree(nullptr),
  fLogFileName("photon_trigger_ml_physics.log"),
  hPhotonScore(nullptr),
  hPhotonScorePhotons(nullptr),
  hPhotonScoreHadrons(nullptr),
  hConfidence(nullptr),
  hRisetime(nullptr),
  hAsymmetry(nullptr),
  hKurtosis(nullptr),
  hScoreVsEnergy(nullptr),
  hScoreVsDistance(nullptr),
  gROCCurve(nullptr),
  hConfusionMatrix(nullptr),
  hTrainingLoss(nullptr),
  hValidationLoss(nullptr),
  hAccuracyHistory(nullptr),
  fTruePositives(0),
  fFalsePositives(0),
  fTrueNegatives(0),
  fFalseNegatives(0),
  fPhotonThreshold(0.65),
  fEnergyMin(1e18),
  fEnergyMax(1e19),
  fOutputFileName("photon_trigger_ml_physics.root"),
  fWeightsFileName("photon_trigger_weights_physics.txt"),
  fLoadPretrainedWeights(true)
{
    fInstance = this;
    fFeatureMeans.assign(17, 0.0);
    fFeatureStdDevs.assign(17, 1.0);

    std::cout << "\n==========================================\n"
              << "PhotonTriggerML Constructor\n"
              << "Output file: " << fOutputFileName << "\n"
              << "Log file: " << fLogFileName << "\n"
              << "==========================================\n";
}

PhotonTriggerML::~PhotonTriggerML()
{
    std::cout << "PhotonTriggerML Destructor called\n";
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init()");

    // Open legacy log (truncate) and flush on every write
    fLogFile.open(fLogFileName.c_str(), std::ios::out);
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    {
        time_t now = time(nullptr);
        fLogFile << "==========================================\n";
        fLogFile << "PhotonTriggerML Physics-Based Version Log\n";
        fLogFile << "Date: " << ctime(&now);
        fLogFile << "==========================================\n\n";
        fLogFile.flush();
    }

    // Scores TSV (truncate)
    gScoresTSV.open("photon_trigger_scores.tsv", std::ios::out);
    if (gScoresTSV.is_open()) {
        gScoresTSV << "event\tstation\tpmt\tenergy_eV\tdistance_m\ttruth(1=photon)\tpred_class\t"
                   << "score\tconfidence";
        for (int i = 0; i < 17; ++i) gScoresTSV << "\t" << kFeatName[i];
        gScoresTSV << "\n";
        gScoresTSV.flush();
    }

    // NN
    fNeuralNetwork->Initialize(17, 8, 4);
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        std::cout << "Loaded pre-trained weights from " << fWeightsFileName << "\n";
        fIsTraining = false;
    } else {
        std::cout << "Starting with random weights (training mode)\n";
    }

    // Output ROOT
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create ROOT output file");
        return eFailure;
    }

    // Tree
    fMLTree = new TTree("MLTree", "PhotonTriggerML Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("photonScore", &fPhotonScore, "photonScore/D");
    fMLTree->Branch("confidence", &fConfidence, "confidence/D");
    fMLTree->Branch("primaryId", &fPrimaryId, "primaryId/I");
    fMLTree->Branch("primaryType", &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");

    // Histograms
    hPhotonScore          = new TH1D("hPhotonScore",         "ML Photon Score (All);Score;Count", 50, 0, 1);
    hPhotonScorePhotons   = new TH1D("hPhotonScorePhotons",  "ML Score (True Photons);Score;Count", 50, 0, 1);
    hPhotonScoreHadrons   = new TH1D("hPhotonScoreHadrons",  "ML Score (True Hadrons);Score;Count", 50, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence           = new TH1D("hConfidence",          "ML Confidence;|Score-0.5|;Count", 50, 0, 0.5);
    hRisetime             = new TH1D("hRisetime",            "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry            = new TH1D("hAsymmetry",           "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis             = new TH1D("hKurtosis",            "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy        = new TH2D("hScoreVsEnergy",       "Score vs Energy;Energy [eV];Score", 50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance      = new TH2D("hScoreVsDistance",     "Score vs Distance;Distance [m];Score", 50, 0, 3000, 50, 0, 1);

    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");

    hTrainingLoss    = new TH1D("hTrainingLoss",    "Training Loss;Batch;Loss", 10000, 0, 10000);
    hValidationLoss  = new TH1D("hValidationLoss",  "Validation Loss;Batch;Loss", 10000, 0, 10000);
    hAccuracyHistory = new TH1D("hAccuracyHistory", "Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    // Per-class feature histograms (raw feature ranges)
    for (int i = 0; i < 17; ++i) {
        std::ostringstream n1, n2, t1, t2;
        n1 << "h_" << kFeatName[i] << "_photon";
        n2 << "h_" << kFeatName[i] << "_hadron";
        t1 << kFeatName[i] << " (photons);" << kFeatName[i] << ";count";
        t2 << kFeatName[i] << " (hadrons);" << kFeatName[i] << ";count";
        const int nb = 60;
        gFeatPhoton[i] = new TH1D(n1.str().c_str(), t1.str().c_str(), nb, kFeatMin[i], kFeatMax[i]);
        gFeatHadron[i] = new TH1D(n2.str().c_str(), t2.str().c_str(), nb, kFeatMin[i], kFeatMax[i]);
    }

    // Signals
    std::signal(SIGINT,  PhotonTriggerMLSignalHandler);
    std::signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    std::cout << "Initialization complete.\n";
    return eSuccess;
}

// =====================================================================================================================
// Run
// =====================================================================================================================
VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    if (fEventCount % 50 == 1) {
        std::cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n";
        std::cout <<   "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│\n";
        std::cout <<   "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n";
    }

    // Shower info
    fEnergy = 0; fCoreX = 0; fCoreY = 0; fPrimaryId = 0; fPrimaryType = "Unknown"; fIsActualPhoton = false;

    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy    = shower.GetEnergy();
        fPrimaryId = shower.GetPrimaryParticle();

        switch (fPrimaryId) {
            case 22: fPrimaryType = "photon"; fIsActualPhoton = true; break;
            case 11: case -11: fPrimaryType = "electron"; break;
            case 2212: fPrimaryType = "proton"; break;
            case 1000026056: fPrimaryType = "iron"; break;
            default: fPrimaryType = (fPrimaryId > 1000000000) ? "nucleus" : "unknown";
        }
        fParticleTypeCounts[fPrimaryType]++;

        if (shower.GetNSimCores() > 0) {
            const Detector& det = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }

        if (fEventCount <= 5) {
            std::cout << "\nEvent " << fEventCount
                      << ": Energy=" << fEnergy/1e18 << " EeV, Primary=" << fPrimaryType
                      << " (ID=" << fPrimaryId << ")\n";
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

    // Periodic metrics/log
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1, 1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2, 1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1, 2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2, 2, fTruePositives);
    }

    // Train with validation-driven threshold re-calibration
    if (fIsTraining && fTrainingFeatures.size() >= 40 && (fEventCount % 5 == 0)) {
        double val_loss = TrainNetwork();  // logs + sets hValidationLoss
        const int b = fEventCount / 5;
        hTrainingLoss->SetBinContent(b, val_loss);

        // Accuracy history
        const int correct = fTruePositives + fTrueNegatives;
        const int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total > 0) {
            double acc = 100.0 * double(correct) / double(total);
            hAccuracyHistory->SetBinContent(b, acc);
        }
    }

    return eSuccess;
}

// =====================================================================================================================
// Station processing
// =====================================================================================================================
void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // Distance to core
    try {
        const Detector& det = Detector::GetInstance();
        const sdet::SDetector& sdetector = det.GetSDetector();
        const sdet::Station& detSt = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
        const double sx = detSt.GetPosition().GetX(siteCS);
        const double sy = detSt.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((sx - fCoreX)*(sx - fCoreX) + (sy - fCoreY)*(sy - fCoreY));
    } catch (...) {
        fDistance = -1;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    for (int p = 0; p < 3; ++p) {
        const int pmtId = firstPMT + p;
        if (!station.HasPMT(pmtId)) continue;

        const sevt::PMT& pmt = station.GetPMT(pmtId);

        // Trace
        std::vector<double> trace;
        if (!ExtractTraceData(pmt, trace) || (int)trace.size() != 2048) continue;

        // Sanity signal window
        const auto minmax = std::minmax_element(trace.begin(), trace.end());
        if ((*minmax.second - *minmax.first) < 10.0) continue;

        // Features
        fFeatures = ExtractEnhancedFeatures(trace);

        // Minimal physics decorrelation (diagnostic only; no positive bias)
        const bool physicsTag =
            (fFeatures.risetime_10_90 < 150.0) &&
            (fFeatures.pulse_width     < 300.0) &&
            (fFeatures.secondary_peak_ratio < 0.30) &&
            (fFeatures.peak_charge_ratio    > 0.15);

        // Hist diagnostics
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);

        // Per-class feature hists + running stats (by truth)
        {
            const double feat[17] = {
                fFeatures.risetime_10_50,
                fFeatures.risetime_10_90,
                fFeatures.falltime_90_10,
                fFeatures.pulse_width,
                fFeatures.asymmetry,
                fFeatures.peak_amplitude,
                fFeatures.total_charge,
                fFeatures.peak_charge_ratio,
                fFeatures.smoothness,
                fFeatures.kurtosis,
                fFeatures.skewness,
                fFeatures.early_fraction,
                fFeatures.late_fraction,
                fFeatures.time_spread,
                fFeatures.high_freq_content,
                (double)fFeatures.num_peaks,
                fFeatures.secondary_peak_ratio
            };
            if (fIsActualPhoton) {
                gCountPhoton++;
                for (int i = 0; i < 17; ++i) {
                    if (gFeatPhoton[i]) gFeatPhoton[i]->Fill(feat[i]);
                    gSumPhoton[i]   += feat[i];
                    gSumSqPhoton[i] += feat[i]*feat[i];
                }
            } else {
                gCountHadron++;
                for (int i = 0; i < 17; ++i) {
                    if (gFeatHadron[i]) gFeatHadron[i]->Fill(feat[i]);
                    gSumHadron[i]   += feat[i];
                    gSumSqHadron[i] += feat[i]*feat[i];
                }
            }
        }

        // Update online stats (used by NormalizeFeatures if desired elsewhere)
        UpdateFeatureStatistics(fFeatures);

        // Normalize -> emphasize select physics features
        std::vector<double> x = NormalizeFeatures(fFeatures);
        if ((int)x.size() == 17) {
            x[1]  *= 2.0;   // risetime_10_90
            x[3]  *= 2.0;   // pulse width
            x[7]  *= 1.5;   // peak charge fraction
            x[16] *= 1.5;   // second peak fraction
        }

        // ML score
        double score = fNeuralNetwork->Predict(x, false);

        // Small spike penalty to improve precision on muon-like spikes
        double penalty = 0.0;
        if (fFeatures.pulse_width < 120.0 && fFeatures.peak_charge_ratio > 0.35) penalty -= 0.20;
        if (fFeatures.num_peaks <= 2   && fFeatures.secondary_peak_ratio > 0.60) penalty -= 0.10;

        fPhotonScore = std::max(0.0, std::min(1.0, score + penalty));
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // Write scores TSV (diagnostics)
        if (gScoresTSV.is_open()) {
            gScoresTSV << fEventCount << '\t' << fStationId << '\t' << pmtId << '\t'
                       << std::setprecision(12) << fEnergy << '\t' << fDistance << '\t'
                       << (fIsActualPhoton ? 1 : 0) << '\t' << ((fPhotonScore > fPhotonThreshold) ? 1 : 0) << '\t'
                       << std::setprecision(6) << fPhotonScore << '\t' << fConfidence;
            gScoresTSV << std::setprecision(6);
            gScoresTSV << '\t' << fFeatures.risetime_10_50
                       << '\t' << fFeatures.risetime_10_90
                       << '\t' << fFeatures.falltime_90_10
                       << '\t' << fFeatures.pulse_width
                       << '\t' << fFeatures.asymmetry
                       << '\t' << fFeatures.peak_amplitude
                       << '\t' << fFeatures.total_charge
                       << '\t' << fFeatures.peak_charge_ratio
                       << '\t' << fFeatures.smoothness
                       << '\t' << fFeatures.kurtosis
                       << '\t' << fFeatures.skewness
                       << '\t' << fFeatures.early_fraction
                       << '\t' << fFeatures.late_fraction
                       << '\t' << fFeatures.time_spread
                       << '\t' << fFeatures.high_freq_content
                       << '\t' << fFeatures.num_peaks
                       << '\t' << fFeatures.secondary_peak_ratio
                       << "\n";
            gScoresTSV.flush();
        }

        // Training/validation buffers
        const bool isVal = (fStationCount % 10 == 0);
        if (fIsTraining) {
            if (!isVal) {
                // light duplication for minority class stability (2x for photons only)
                if (fIsActualPhoton) {
                    for (int c = 0; c < 2; ++c) {
                        std::vector<double> v = x;
                        static std::mt19937 gen(1234567u);
                        std::normal_distribution<> n(0.0, 0.02);
                        for (double& z : v) z += n(gen);
                        fTrainingFeatures.push_back(v);
                        fTrainingLabels.push_back(1);
                    }
                } else {
                    fTrainingFeatures.push_back(x);
                    fTrainingLabels.push_back(0);
                }
            } else {
                fValidationFeatures.push_back(x);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }

        fStationCount++;
        const bool isPhoton = (fPhotonScore > fPhotonThreshold);

        // Store ML result for this station (last PMT processed wins; consistent with previous behavior)
        MLResult res;
        res.photonScore       = fPhotonScore;
        res.identifiedAsPhoton= isPhoton;
        res.isActualPhoton    = fIsActualPhoton;
        res.vemCharge         = fFeatures.total_charge;
        res.features          = fFeatures;
        res.primaryType       = fPrimaryType;
        res.confidence        = fConfidence;
        fMLResultsMap[fStationId] = res;

        // Confusion counters
        if (isPhoton) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) ++fTruePositives; else ++fFalsePositives;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) ++fTrueNegatives; else ++fFalseNegatives;
        }

        // Fill hists
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);

        // Tree
        if (fMLTree) fMLTree->Fill();

        // Early debug lines for photon-like truth or physics tag
        if ((fIsActualPhoton || physicsTag) && fStationCount <= 100) {
            std::cout << "  Station " << fStationId
                      << " PMT " << pmtId
                      << ": Score=" << std::fixed << std::setprecision(3) << fPhotonScore
                      << " (True: " << fPrimaryType << ")"
                      << " Rise=" << fFeatures.risetime_10_90 << "ns"
                      << " Width="<< fFeatures.pulse_width   << "ns"
                      << " PhysicsTag=" << (physicsTag ? "YES" : "NO") << "\n";
        }
    }
}

// =====================================================================================================================
// Trace extraction
// =====================================================================================================================
bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace)
{
    trace.clear();

    if (pmt.HasFADCTrace()) {
        try {
            const auto& t = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            for (int i = 0; i < 2048; ++i) trace.push_back(t[i]);
            return true;
        } catch (...) {}
    }

    if (pmt.HasSimData()) {
        try {
            const sevt::PMTSimData& sd = pmt.GetSimData();
            if (sd.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& t = sd.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                sevt::StationConstants::eTotal);
                for (int i = 0; i < 2048; ++i) trace.push_back(t[i]);
                return true;
            }
        } catch (...) {}
    }

    return false;
}

// =====================================================================================================================
// Feature extraction
// =====================================================================================================================
PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
    EnhancedFeatures F;                      // zeros by default
    const int N = (int)trace.size();
    if (N <= 0) return F;

    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    // Baseline from first 100 bins
    double base = 0.0;
    const int nb = std::min(100, N);
    for (int i = 0; i < nb; ++i) base += trace[i];
    base /= std::max(1, nb);

    // Signal, peak, total
    std::vector<double> sig(N, 0.0);
    int peakBin = 0;
    double peakVal = 0.0, total = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = trace[i] - base;
        if (s < 0.0) s = 0.0;
        sig[i] = s;
        if (s > peakVal) { peakVal = s; peakBin = i; }
        total += s;
    }
    if (peakVal < 5.0 || total < 10.0) return F;

    F.peak_amplitude   = peakVal / ADC_PER_VEM;
    F.total_charge     = total   / ADC_PER_VEM;
    F.peak_charge_ratio= F.peak_amplitude / (F.total_charge + 1e-3);

    // Rise/fall times
    const double v10 = 0.10 * peakVal;
    const double v50 = 0.50 * peakVal;
    const double v90 = 0.90 * peakVal;

    int r10 = 0, r50 = 0, r90 = peakBin;
    for (int i = peakBin; i >= 0; --i) {
        if (sig[i] <= v90 && r90 == peakBin) r90 = i;
        if (sig[i] <= v50 && r50 == 0)       r50 = i;
        if (sig[i] <= v10) { r10 = i; break; }
    }

    int f90 = peakBin, f10 = N - 1;
    for (int i = peakBin; i < N; ++i) {
        if (sig[i] <= v90 && f90 == peakBin) f90 = i;
        if (sig[i] <= v10) { f10 = i; break; }
    }

    F.risetime_10_50 = std::abs(r50 - r10) * NS_PER_BIN;
    F.risetime_10_90 = std::abs(r90 - r10) * NS_PER_BIN;
    F.falltime_90_10 = std::abs(f10 - f90) * NS_PER_BIN;

    // FWHM
    const double hm = 0.5 * peakVal;
    int rHM = r10;
    int fHM = f10;
    for (int i = r10; i <= peakBin; ++i) { if (sig[i] >= hm) { rHM = i; break; } }
    for (int i = peakBin; i < N;    ++i) { if (sig[i] <= hm) { fHM = i; break; } }
    F.pulse_width = std::abs(fHM - rHM) * NS_PER_BIN;

    // Asymmetry
    const double rise = F.risetime_10_90;
    const double fall = F.falltime_90_10;
    F.asymmetry = (fall - rise) / (fall + rise + 1e-3);

    // Moments
    double m1 = 0.0;
    for (int i = 0; i < N; ++i) m1 += i * sig[i];
    m1 /= (total + 1e-3);

    double var = 0.0, sk = 0.0, ku = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = i - m1;
        const double w = sig[i] / (total + 1e-3);
        var += d*d * w;
        sk  += d*d*d * w;
        ku  += d*d*d*d * w;
    }
    const double sd = std::sqrt(var + 1e-3);
    F.time_spread = sd * NS_PER_BIN;
    F.skewness    = sk / (sd*sd*sd + 1e-6);
    F.kurtosis    = ku / (var*var + 1e-6) - 3.0;

    // Early/Late fractions
    const int q = N / 4;
    double early = 0.0, late = 0.0;
    for (int i = 0; i < q; ++i)        early += sig[i];
    for (int i = 3*q; i < N; ++i)      late  += sig[i];
    F.early_fraction = early / (total + 1e-3);
    F.late_fraction  = late  / (total + 1e-3);

    // Smoothness (2nd derivative energy) above 10% peak
    double ssum = 0.0;
    int scount = 0;
    for (int i = 1; i < N - 1; ++i) {
        if (sig[i] > v10) {
            const double d2 = sig[i+1] - 2*sig[i] + sig[i-1];
            ssum += d2*d2;
            scount++;
        }
    }
    F.smoothness = std::sqrt(ssum / (scount + 1));

    // High-frequency content (first-difference energy)
    double hf_energy = 0.0;
    for (int i = 1; i < N - 1; ++i) {
        const double d = sig[i+1] - sig[i-1];
        hf_energy += d*d;
    }
    F.high_freq_content = hf_energy / (total*total + 1e-3);

    // Peaks (higher threshold to suppress noise)
    F.num_peaks = 0;
    double sec_peak = 0.0;
    const double thr = 0.25 * peakVal; // was 0.15
    for (int i = 1; i < N - 1; ++i) {
        if (sig[i] > thr && sig[i] > sig[i-1] && sig[i] > sig[i+1]) {
            F.num_peaks++;
            if (i != peakBin && sig[i] > sec_peak) sec_peak = sig[i];
        }
    }
    F.secondary_peak_ratio = sec_peak / (peakVal + 1e-3);

    return F;
}

// =====================================================================================================================
// Online feature stats (kept simple; not used for normalization here other than safeguarding)
// =====================================================================================================================
void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& f)
{
    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
    };

    static long long n = 0;
    n++;
    for (size_t i = 0; i < raw.size(); ++i) {
        const double delta = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += delta / (double)n;
        const double delta2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += delta * delta2;
    }
    if (n > 1) {
        for (size_t i = 0; i < raw.size(); ++i) {
            double v = fFeatureStdDevs[i] / (double)(n - 1);
            v = (v > 0.0) ? std::sqrt(v) : 1.0;
            if (v < 1e-3) v = 1.0;
            fFeatureStdDevs[i] = v;
        }
    }
}

// =====================================================================================================================
// Feature normalization – min/max ranges; clipped to [0,1]
// =====================================================================================================================
std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& f)
{
    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
    };

    std::vector<double> z;
    z.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        const double mn = kFeatMin[i], mx = kFeatMax[i];
        double v = (raw[i] - mn) / (mx - mn + 1e-3);
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        z.push_back(v);
    }
    return z;
}

// =====================================================================================================================
// Training + validation-driven threshold re-calibration
// =====================================================================================================================
double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;

    // Random mini-batch (max 128)
    const int maxN = std::min<int>(128, (int)fTrainingFeatures.size());
    std::vector<int> idx(fTrainingFeatures.size());
    std::iota(idx.begin(), idx.end(), 0);
    static std::mt19937 gen(987654321u);
    std::shuffle(idx.begin(), idx.end(), gen);

    std::vector<std::vector<double>> Bx; Bx.reserve(maxN);
    std::vector<int> By; By.reserve(maxN);
    for (int i = 0; i < maxN; ++i) {
        Bx.push_back(fTrainingFeatures[idx[i]]);
        By.push_back(fTrainingLabels[idx[i]]);
    }

    // Lean learning rate with a few epochs per batch
    const double lr = 0.007;
    double trainLoss = 0.0;
    const int epochs = 8;
    for (int e = 0; e < epochs; ++e) trainLoss += fNeuralNetwork->Train(Bx, By, lr);
    trainLoss /= (double)epochs;

    std::cout << "\n  Training with " << maxN
              << " samples (P:" << std::count(By.begin(), By.end(), 1)
              << " H:" << (int)By.size() - std::count(By.begin(), By.end(), 1)
              << ") Loss: " << std::fixed << std::setprecision(4) << trainLoss;

    // Validation loss + threshold sweep
    double valLoss = 0.0;
    if (!fValidationFeatures.empty()) {
        const int N = (int)fValidationFeatures.size();
        std::vector<double> P(N, 0.0);
        for (int i = 0; i < N; ++i) P[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

        int correct = 0;
        for (int i = 0; i < N; ++i) {
            const int y = fValidationLabels[i];
            const double p = P[i];
            valLoss += - (y * std::log(p + 1e-7) + (1 - y) * std::log(1 - p + 1e-7));
            const int pred = (p > fPhotonThreshold) ? 1 : 0;
            if (pred == y) ++correct;
        }
        valLoss /= (double)N;
        const double valAcc = 100.0 * double(correct) / double(N);
        std::cout << " Val: " << valLoss << " (Acc: " << valAcc << "%)";

        // Sweep 0.40 .. 0.90 step 0.02; choose best F1, fallback to balanced accuracy if F1 poor
        double bestF1 = -1.0, bestThr = fPhotonThreshold;
        double bestBA = -1.0, bestThrBA = fPhotonThreshold;

        for (double thr = 0.40; thr <= 0.90 + 1e-9; thr += 0.02) {
            int TP=0, FP=0, TN=0, FN=0;
            for (int i = 0; i < N; ++i) {
                int y = fValidationLabels[i];
                int pr = (P[i] > thr) ? 1 : 0;
                if (pr==1 && y==1) ++TP;
                else if (pr==1 && y==0) ++FP;
                else if (pr==0 && y==0) ++TN;
                else ++FN;
            }
            const double prec = (TP+FP>0) ? (double)TP/(TP+FP) : 0.0;
            const double rec  = (TP+FN>0) ? (double)TP/(TP+FN) : 0.0;
            const double F1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;
            const double TPR  = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            const double TNR  = (TN+FP>0)? (double)TN/(TN+FP) : 0.0;
            const double BA   = 0.5*(TPR+TNR);

            if (F1 > bestF1) { bestF1 = F1; bestThr = thr; }
            if (BA > bestBA) { bestBA = BA; bestThrBA = thr; }
        }

        // Update smoothly (20% toward chosen target)
        const double targetThr = (bestF1 >= 0.20) ? bestThr : bestThrBA;
        const double oldThr = fPhotonThreshold;
        fPhotonThreshold = 0.8 * fPhotonThreshold + 0.2 * targetThr;

        if (fLogFile.is_open()) {
            fLogFile << "Recalibrated threshold from " << oldThr
                    << " -> " << fPhotonThreshold
                    << " (bestF1=" << bestF1 << " at " << bestThr
                    << ", bestBA=" << bestBA << " at " << bestThrBA << ")\n";
            fLogFile.flush();
        }

        const int b = fEventCount / 5;
        hValidationLoss->SetBinContent(b, valLoss);
    }

    // Keep buffer bounded
    if ((int)fTrainingFeatures.size() > 2000) {
        const int cut = 400;
        fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin() + cut);
        fTrainingLabels.erase(fTrainingLabels.begin(), fTrainingLabels.begin() + cut);
    }

    fTrainingStep++;
    std::cout << std::endl;
    return valLoss;
}

// =====================================================================================================================
// Legacy metrics display + log
// =====================================================================================================================
void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) return;

    const double accuracy  = 100.0 * double(fTruePositives + fTrueNegatives) / double(total);
    const double precision = (fTruePositives + fFalsePositives > 0) ?
                             100.0 * double(fTruePositives) / double(fTruePositives + fFalsePositives) : 0.0;
    const double recall    = (fTruePositives + fFalseNegatives > 0) ?
                             100.0 * double(fTruePositives) / double(fTruePositives + fFalseNegatives) : 0.0;
    const double f1        = (precision + recall > 0.0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
    const double phFrac    = (fPhotonLikeCount + fHadronLikeCount > 0) ?
                             100.0 * double(fPhotonLikeCount) / double(fPhotonLikeCount + fHadronLikeCount) : 0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << phFrac << "%"
              << " │ " << std::setw(7) << accuracy  << "%"
              << " │ " << std::setw(8) << precision << "%"
              << " │ " << std::setw(7) << recall    << "%"
              << " │ " << std::setw(7) << f1        << "%│\n";

    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount
                 << " - Acc: "  << accuracy
                 << "% Prec: "  << precision
                 << "% Rec: "   << recall
                 << "% F1: "    << f1 << "%\n";
        fLogFile.flush();
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    std::cout << "\n==========================================\n"
              << "PERFORMANCE METRICS\n"
              << "==========================================\n";

    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) {
        std::cout << "No predictions made yet!\n";
        return;
    }

    const double accuracy  = 100.0 * double(fTruePositives + fTrueNegatives) / double(total);
    const double precision = (fTruePositives + fFalsePositives > 0) ?
                             100.0 * double(fTruePositives) / double(fTruePositives + fFalsePositives) : 0.0;
    const double recall    = (fTruePositives + fFalseNegatives > 0) ?
                             100.0 * double(fTruePositives) / double(fTruePositives + fFalseNegatives) : 0.0;
    const double f1        = (precision + recall > 0.0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    std::cout << "Accuracy:  " << std::fixed << std::setprecision(1) << accuracy  << "%\n";
    std::cout << "Precision: " << precision << "%\n";
    std::cout << "Recall:    " << recall    << "%\n";
    std::cout << "F1-Score:  " << f1        << "%\n\n";

    std::cout << "CONFUSION MATRIX:\n"
              << "                Predicted\n"
              << "             Hadron   Photon\n"
              << "Actual Hadron  " << std::setw(6) << fTrueNegatives << "   " << std::setw(6) << fFalsePositives << "\n"
              << "       Photon  " << std::setw(6) << fFalseNegatives << "   " << std::setw(6) << fTruePositives  << "\n\n";

    std::cout << "Total Stations: " << fStationCount << "\n";
    std::cout << "Photon-like: " << fPhotonLikeCount << " ("
              << 100.0 * double(fPhotonLikeCount) / std::max(1.0, double(fPhotonLikeCount + fHadronLikeCount))
              << "%)\n";
    std::cout << "Hadron-like: " << fHadronLikeCount << " ("
              << 100.0 * double(fHadronLikeCount) / std::max(1.0, double(fPhotonLikeCount + fHadronLikeCount))
              << "%)\n\n";

    std::cout << "PARTICLE TYPE BREAKDOWN:\n";
    for (const auto& kv : fParticleTypeCounts) {
        std::cout << "  " << kv.first << ": " << kv.second << " events\n";
    }
    std::cout << "==========================================\n";

    if (fLogFile.is_open()) {
        fLogFile << "\nFinal Performance Metrics:\n";
        fLogFile << "Accuracy: "  << accuracy  << "%\n";
        fLogFile << "Precision: " << precision << "%\n";
        fLogFile << "Recall: "    << recall    << "%\n";
        fLogFile << "F1-Score: "  << f1        << "%\n";
        fLogFile.flush();
    }
}

// =====================================================================================================================
// Final summary, file outputs, & safe in-place annotation of external PMT trace files
// =====================================================================================================================
void PhotonTriggerML::SaveAndDisplaySummary()
{
    std::cout << "\n==========================================\n"
              << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n"
              << "==========================================\n";
    std::cout << "Events processed: "   << fEventCount   << "\n";
    std::cout << "Stations analyzed: "  << fStationCount << "\n";

    CalculatePerformanceMetrics();

    // Save trained weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // ROOT outputs
    if (fOutputFile) {
        fOutputFile->cd();

        if (fMLTree) {
            fMLTree->Write();
            std::cout << "Wrote " << fMLTree->GetEntries() << " entries to MLTree\n";
        }

        // Main hists
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

        // Per-class feature histograms
        for (int i = 0; i < 17; ++i) {
            if (gFeatPhoton[i]) gFeatPhoton[i]->Write();
            if (gFeatHadron[i]) gFeatHadron[i]->Write();
        }

        std::cout << "Histograms written to " << fOutputFileName << "\n";
        fOutputFile->Close();
        delete fOutputFile; fOutputFile = nullptr;
    }

    // Minimal extra ROOT file with key plots (legacy helper)
    {
        TFile fsum("photon_trigger_summary.root", "RECREATE");
        if (!fsum.IsZombie()) {
            if (hConfusionMatrix)     hConfusionMatrix->Write("ConfusionMatrix");
            if (hPhotonScore)         hPhotonScore->Write("PhotonScore_All");
            if (hPhotonScorePhotons)  hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
            if (hPhotonScoreHadrons)  hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
            if (hConfidence)          hConfidence->Write("Confidence");
            // Feature hists (top diagnostics)
            for (int i = 0; i < 17; ++i) {
                if (gFeatPhoton[i]) gFeatPhoton[i]->Write();
                if (gFeatHadron[i]) gFeatHadron[i]->Write();
            }
            fsum.Close();
        }
    }

    // Legacy text summary
    {
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            const int TP = fTruePositives, FP = fFalsePositives, TN = fTrueNegatives, FN = fFalseNegatives;
            const int total = TP + FP + TN + FN;
            const double acc  = (total>0) ? 100.0*(TP+TN)/double(total) : 0.0;
            const double prec = (TP+FP>0)? 100.0*TP/double(TP+FP) : 0.0;
            const double rec  = (TP+FN>0)? 100.0*TP/double(TP+FN) : 0.0;
            const double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;
            txt << "PhotonTriggerML Summary\n";
            txt << "Threshold: " << fPhotonThreshold << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy=" << acc << "%  Precision=" << prec << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
            txt.close();
        }
    }

    // Per-feature classwise stats TSV (means/std + simple separation)
    {
        std::ofstream tsv("photon_trigger_param_stats.tsv");
        if (tsv.is_open()) {
            tsv << "feature\t"
                << "N_ph\tmean_ph\tstd_ph\t"
                << "N_hd\tmean_hd\tstd_hd\t"
                << "sep=(|mu_ph-mu_hd|/sqrt(0.5*(sig2_ph+sig2_hd)))\n";
            for (int i = 0; i < 17; ++i) {
                const double nph = (double)std::max(1LL, gCountPhoton);
                const double nhd = (double)std::max(1LL, gCountHadron);
                const double mu_ph = gSumPhoton[i] / nph;
                const double mu_hd = gSumHadron[i] / nhd;
                const double v_ph  = std::max(0.0, gSumSqPhoton[i] / nph - mu_ph*mu_ph);
                const double v_hd  = std::max(0.0, gSumSqHadron[i] / nhd - mu_hd*mu_hd);
                const double sd_ph = std::sqrt(v_ph);
                const double sd_hd = std::sqrt(v_hd);
                const double sep   = std::fabs(mu_ph - mu_hd) / std::sqrt(0.5*(v_ph + v_hd) + 1e-9);

                tsv << kFeatName[i] << '\t'
                    << (long long)gCountPhoton << '\t' << mu_ph << '\t' << sd_ph << '\t'
                    << (long long)gCountHadron << '\t' << mu_hd << '\t' << sd_hd << '\t'
                    << sep << "\n";
            }
            tsv.close();
        }
    }

    // Safe in-place annotation of external PMT traces (skip from signal handler)
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& title)->std::string {
            size_t p = title.find(" [ML:");
            return (p == std::string::npos) ? title : title.substr(0, p);
        };

        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool {
            if (!h) return false;
            const int n = h->GetNbinsX();
            if (n <= 0) return false;

            std::vector<double> tr(n);
            for (int i = 1; i <= n; ++i) tr[i-1] = h->GetBinContent(i);

            EnhancedFeatures F = ExtractEnhancedFeatures(tr);
            std::vector<double> X = NormalizeFeatures(F);
            if ((int)X.size() == 17) {
                X[1]  *= 2.0;
                X[3]  *= 2.0;
                X[7]  *= 1.5;
                X[16] *= 1.5;
            }
            double ml = fNeuralNetwork->Predict(X, false);

            double pen = 0.0;
            if (F.pulse_width < 120.0 && F.peak_charge_ratio > 0.35) pen -= 0.20;
            if (F.num_peaks <= 2   && F.secondary_peak_ratio > 0.60) pen -= 0.10;

            score = std::max(0.0, std::min(1.0, ml + pen));
            isPhoton = (score > fPhotonThreshold);
            return true;
        };

        std::function<void(TDirectory*)> annotate_dir = [&](TDirectory* dir){
            if (!dir) return;
            TIter next(dir->GetListOfKeys());
            while (TKey* key = (TKey*)next()) {
                TObject* obj = dir->Get(key->GetName());
                if (!obj) continue;

                if (obj->InheritsFrom(TDirectory::Class())) {
                    annotate_dir((TDirectory*)obj);
                } else if (obj->InheritsFrom(TH1::Class())) {
                    TH1* h = (TH1*)obj;
                    double sc = 0.0; bool ph = false;
                    if (score_from_hist(h, sc, ph)) {
                        const std::string base = strip_ml(h->GetTitle() ? h->GetTitle() : "");
                        std::ostringstream t;
                        t << base << " [ML: " << std::fixed << std::setprecision(3) << sc
                          << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                        h->SetTitle(t.str().c_str());
                        dir->cd();
                        h->Write(h->GetName(), TObject::kOverwrite);
                    }
                } else if (obj->InheritsFrom(TCanvas::Class())) {
                    TCanvas* c = (TCanvas*)obj;
                    TH1* hfirst = nullptr;
                    if (TList* prim = c->GetListOfPrimitives()) {
                        TIter nx(prim);
                        while (TObject* po = nx()) {
                            if (po->InheritsFrom(TH1::Class())) { hfirst = (TH1*)po; break; }
                        }
                    }
                    if (hfirst) {
                        double sc = 0.0; bool ph = false;
                        if (score_from_hist(hfirst, sc, ph)) {
                            const std::string baseC = strip_ml(c->GetTitle() ? c->GetTitle() : "");
                            const std::string baseH = strip_ml(hfirst->GetTitle() ? hfirst->GetTitle() : "");
                            std::ostringstream tc, th;
                            tc << baseC << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                            th << baseH << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                            c->SetTitle(tc.str().c_str());
                            hfirst->SetTitle(th.str().c_str());
                            c->Modified(); c->Update();
                            dir->cd();
                            c->Write(c->GetName(), TObject::kOverwrite);
                        }
                    }
                }
                delete obj;
            }
        };

        const char* candidates[] = {
            "pmt_traces_1EeV.root", "pmt_Traces_1EeV.root",
            "pmt_traces_1eev.root", "pmt_Traces_1eev.root"
        };
        TFile* fext = nullptr;
        for (const char* nm : candidates) {
            fext = TFile::Open(nm, "UPDATE");
            if (fext && !fext->IsZombie()) break;
            if (fext) { delete fext; fext = nullptr; }
        }
        if (fext && !fext->IsZombie()) {
            if (!fext->TestBit(TFile::kRecovered)) {
                annotate_dir(fext);
                fext->Write("", TObject::kOverwrite);
                std::cout << "Annotated ML tags in existing trace file: " << fext->GetName() << "\n";
            } else {
                std::cout << "Trace file appears recovered; skipping ML annotation for safety.\n";
            }
            fext->Close(); delete fext;
        } else {
            std::cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    // Close OPEN logs/TSVs
    if (gScoresTSV.is_open()) { gScoresTSV.flush(); gScoresTSV.close(); }
    if (fLogFile.is_open())   { fLogFile.flush();   fLogFile.close(); }

    std::cout << "==========================================\n";
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish()");
    SaveAndDisplaySummary();
    return eSuccess;
}

// =====================================================================================================================
// Accessors
// =====================================================================================================================
bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& res)
{
    auto it = fMLResultsMap.find(stationId);
    if (it != fMLResultsMap.end()) { res = it->second; return true; }
    return false;
}

void PhotonTriggerML::ClearMLResults()
{
    fMLResultsMap.clear();
}

