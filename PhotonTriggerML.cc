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
// Static globals (safe to keep out of the header)
// ============================================================================

// Static instance pointer for signal handler
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

// Static ML results map
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Flag used to detect if SaveAndDisplaySummary() is being invoked from a signal
static volatile sig_atomic_t gCalledFromSignal = 0;

// Per-trace score dump (restored)
static std::ofstream gScoresTSV;

// Keep all predictions/labels for post-run diagnostics (FN/FP lists & sweeps)
struct _PredRec {
    int eventId{0};
    int stationId{0};
    int pmtId{0};
    int label{0};                 // 1 photon, 0 hadron
    double score{0};              // final score (after penalty)
    double mlScore{0};            // raw NN score
    double penalty{0};            // penalty applied (muon spike etc.)
    double distance{0};
    // a few salient features for debugging
    double rt10_90{0};
    double width{0};
    int    nPeaks{0};
    double secPeakRatio{0};
    double peakChargeRatio{0};
    double totalCharge{0};
    double peakAmp{0};
    bool   physicsTag{false};
};
static std::vector<_PredRec> gAllRecs;

// ============================================================================
// Minimal signal handler: save, restore default handler, then re-raise.
// ============================================================================
void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        std::cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << std::endl;

        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try {
                PhotonTriggerML::fInstance->SaveAndDisplaySummary();
            } catch (...) {
                // avoid any throw on exit
            }
        }
        std::signal(signal, SIG_DFL);
        std::raise(signal);
        _Exit(0);
    }
}

// ============================================================================
// Tiny physics‑motivated NN (same architecture as before)
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.1),
    fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize   = input_size;
    fHidden1Size = hidden1_size;
    fHidden2Size = hidden2_size;

    std::cout << "Initializing Physics-Based Neural Network: "
              << input_size << " -> " << hidden1_size << " -> "
              << hidden2_size << " -> 1" << std::endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    for (int i = 0; i < fHidden1Size; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            fWeights1[i][j] = dist(gen) / std::sqrt(double(fInputSize));
        }
    }

    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    for (int i = 0; i < fHidden2Size; ++i) {
        for (int j = 0; j < fHidden1Size; ++j) {
            fWeights2[i][j] = dist(gen) / std::sqrt(double(fHidden1Size));
        }
    }

    fWeights3.assign(1, std::vector<double>(fHidden2Size));
    for (int j = 0; j < fHidden2Size; ++j) {
        fWeights3[0][j] = dist(gen) / std::sqrt(double(fHidden2Size));
    }

    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);
    fBias3 = 0.0;

    fMomentum1_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0));
    fMomentum2_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0));
    fMomentum1_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0));
    fMomentum2_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0));
    fMomentum1_w3.assign(1, std::vector<double>(fHidden2Size, 0));
    fMomentum2_w3.assign(1, std::vector<double>(fHidden2Size, 0));

    fMomentum1_b1.assign(fHidden1Size, 0);
    fMomentum2_b1.assign(fHidden1Size, 0);
    fMomentum1_b2.assign(fHidden2Size, 0);
    fMomentum2_b2.assign(fHidden2Size, 0);
    fMomentum1_b3 = 0;
    fMomentum2_b3 = 0;

    fTimeStep = 0;
    std::cout << "Neural Network initialized for physics-based discrimination!" << std::endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if (int(x.size()) != fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size, 0.0);
    for (int i = 0; i < fHidden1Size; ++i) {
        double s = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * x[j];
        h1[i] = 1.0 / (1.0 + std::exp(-s));
        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) h1[i] = 0.0;
    }

    std::vector<double> h2(fHidden2Size, 0.0);
    for (int i = 0; i < fHidden2Size; ++i) {
        double s = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
        h2[i] = 1.0 / (1.0 + std::exp(-s));
        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) h2[i] = 0.0;
    }

    double o = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) o += fWeights3[0][j] * h2[j];
    return 1.0 / (1.0 + std::exp(-o));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y,
                                             double lr)
{
    if (X.empty() || X.size() != y.size()) return -1.0;

    const int n = int(X.size());
    // very mild class weighting to avoid "all-photon" bias
    int nPho = std::count(y.begin(), y.end(), 1);
    const double wPho = (nPho > 0) ? 1.5 : 1.0;
    const double wHad = 1.0;

    std::vector<std::vector<double>> gw1(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    std::vector<std::vector<double>> gw2(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    std::vector<std::vector<double>> gw3(1, std::vector<double>(fHidden2Size, 0.0));
    std::vector<double> gb1(fHidden1Size, 0.0);
    std::vector<double> gb2(fHidden2Size, 0.0);
    double gb3 = 0.0;

    double totalLoss = 0.0;

    for (int k = 0; k < n; ++k) {
        const auto& x = X[k];
        const int t = y[k];
        const double clsW = (t == 1) ? wPho : wHad;

        // forward
        std::vector<double> h1(fHidden1Size), h1z(fHidden1Size);
        for (int i = 0; i < fHidden1Size; ++i) {
            double s = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * x[j];
            h1z[i] = s;
            h1[i] = 1.0 / (1.0 + std::exp(-s));
        }
        std::vector<double> h2(fHidden2Size), h2z(fHidden2Size);
        for (int i = 0; i < fHidden2Size; ++i) {
            double s = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
            h2z[i] = s;
            h2[i] = 1.0 / (1.0 + std::exp(-s));
        }
        double oz = fBias3;
        for (int j = 0; j < fHidden2Size; ++j) oz += fWeights3[0][j] * h2[j];
        double o = 1.0 / (1.0 + std::exp(-oz));

        // loss
        totalLoss += -clsW * (t * std::log(o + 1e-7) + (1 - t) * std::log(1 - o + 1e-7));

        // backprop
        double go = clsW * (o - t);
        for (int j = 0; j < fHidden2Size; ++j) gw3[0][j] += go * h2[j];
        gb3 += go;

        std::vector<double> gh2(fHidden2Size, 0.0);
        for (int j = 0; j < fHidden2Size; ++j) {
            gh2[j] = fWeights3[0][j] * go * h2[j] * (1.0 - h2[j]);
        }
        for (int i = 0; i < fHidden2Size; ++i) {
            for (int j = 0; j < fHidden1Size; ++j) gw2[i][j] += gh2[i] * h1[j];
            gb2[i] += gh2[i];
        }

        std::vector<double> gh1(fHidden1Size, 0.0);
        for (int j = 0; j < fHidden1Size; ++j) {
            double s = 0.0;
            for (int i = 0; i < fHidden2Size; ++i) s += fWeights2[i][j] * gh2[i];
            gh1[j] = s * h1[j] * (1.0 - h1[j]);
        }
        for (int i = 0; i < fHidden1Size; ++i) {
            for (int j = 0; j < fInputSize; ++j) gw1[i][j] += gh1[i] * x[j];
            gb1[i] += gh1[i];
        }
    }

    // SGD w/ momentum
    ++fTimeStep;
    const double mom = 0.9;

    for (int i = 0; i < fHidden1Size; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            const double g = gw1[i][j] / n;
            fMomentum1_w1[i][j] = mom * fMomentum1_w1[i][j] - lr * g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        const double gb = gb1[i] / n;
        fMomentum1_b1[i] = mom * fMomentum1_b1[i] - lr * gb;
        fBias1[i] += fMomentum1_b1[i];
    }

    for (int i = 0; i < fHidden2Size; ++i) {
        for (int j = 0; j < fHidden1Size; ++j) {
            const double g = gw2[i][j] / n;
            fMomentum1_w2[i][j] = mom * fMomentum1_w2[i][j] - lr * g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        const double gb = gb2[i] / n;
        fMomentum1_b2[i] = mom * fMomentum1_b2[i] - lr * gb;
        fBias2[i] += fMomentum1_b2[i];
    }

    for (int j = 0; j < fHidden2Size; ++j) {
        const double g = gw3[0][j] / n;
        fMomentum1_w3[0][j] = mom * fMomentum1_w3[0][j] - lr * g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }

    const double gb = gb3 / n;
    fMomentum1_b3 = mom * fMomentum1_b3 - lr * gb;
    fBias3 += fMomentum1_b3;

    return totalLoss / n;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Error: Could not save weights to " << filename << std::endl;
        return;
    }
    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";

    for (const auto& row : fWeights1) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias1) { file << b << " "; }
    file << "\n";

    for (const auto& row : fWeights2) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias2) { file << b << " "; }
    file << "\n";

    for (double w : fWeights3[0]) { file << w << " "; }
    file << "\n";
    file << fBias3 << "\n";

    file.close();
    std::cout << "Weights saved to " << filename << std::endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Warning: Could not load weights from " << filename << std::endl;
        return false;
    }

    file >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.assign(1, std::vector<double>(fHidden2Size));
    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);

    for (auto& row : fWeights1) {
        for (double& w : row) file >> w;
    }
    for (double& b : fBias1) file >> b;

    for (auto& row : fWeights2) {
        for (double& w : row) file >> w;
    }
    for (double& b : fBias2) file >> b;

    for (double& w : fWeights3[0]) file >> w;
    file >> fBias3;

    file.close();
    std::cout << "Weights loaded from " << filename << std::endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights() { fIsQuantized = true; }

// ============================================================================
// PhotonTriggerML – constructor / init
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
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
}

PhotonTriggerML::~PhotonTriggerML() { std::cout << "PhotonTriggerML Destructor called\n"; }

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization");

    // Log header (old format)
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    time_t now = time(0);
    fLogFile << "==========================================\n";
    fLogFile << "PhotonTriggerML Physics-Based Version Log\n";
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================\n\n";

    // NN init (17 inputs -> 8 -> 4 -> 1)
    fNeuralNetwork->Initialize(17, 8, 4);

    // Load pre-trained weights if available
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        std::cout << "Loaded pre-trained weights from " << fWeightsFileName << std::endl;
        fIsTraining = false; // inference-only if weights exist
    } else {
        std::cout << "Starting with random weights (training mode)" << std::endl;
    }

    // Output ROOT
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }

    // Tree
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

    // Hists
    hPhotonScore          = new TH1D("hPhotonScore",          "ML Photon Score (All);Score;Count", 50, 0, 1);
    hPhotonScorePhotons   = new TH1D("hPhotonScorePhotons",   "ML Score (True Photons);Score;Count", 50, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons   = new TH1D("hPhotonScoreHadrons",   "ML Score (True Hadrons);Score;Count", 50, 0, 1);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence           = new TH1D("hConfidence",           "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime             = new TH1D("hRisetime",             "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry            = new TH1D("hAsymmetry",            "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis             = new TH1D("hKurtosis",             "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy        = new TH2D("hScoreVsEnergy",        "Score vs Energy;Energy [eV];Score", 50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance      = new TH2D("hScoreVsDistance",      "Score vs Distance;Distance [m];Score", 50, 0, 3000, 50, 0, 1);

    hConfusionMatrix      = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual",
                                     2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");

    hTrainingLoss   = new TH1D("hTrainingLoss",   "Training Loss;Batch;Loss",      10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss",    10000, 0, 10000);
    hAccuracyHistory= new TH1D("hAccuracyHistory","Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    // Per-trace score TSV (restored & header)
    gScoresTSV.open("photon_trigger_scores.tsv");
    if (gScoresTSV.is_open()) {
        gScoresTSV << "event\tstation\tpmt\ttrueLabel\tpredLabel\tscore\tmlScore\tpenalty\t"
                   << "dist_m\trise10_90_ns\twidth_ns\tnPeaks\tsecPeakRatio\tpeakChargeRatio\t"
                   << "totalCharge_VEM\tpeakAmp_VEM\tprimary\n";
    }

    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    INFO("PhotonTriggerML initialized successfully");
    return eSuccess;
}

// ============================================================================

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    ++fEventCount;
    ClearMLResults();

    // Header line every 50 events (terminal table)
    if (fEventCount % 50 == 1) {
        std::cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n";
        std::cout <<   "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│\n";
        std::cout <<   "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n";
    }

    // Reset per event derived info
    fEnergy = 0;
    fCoreX = 0; fCoreY = 0;
    fPrimaryId = 0;
    fPrimaryType = "Unknown";
    fIsActualPhoton = false;

    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy    = shower.GetEnergy();
        fPrimaryId = shower.GetPrimaryParticle();

        switch (fPrimaryId) {
            case 22:            fPrimaryType = "photon"; fIsActualPhoton = true; break;
            case 11: case -11:  fPrimaryType = "electron"; break;
            case 2212:          fPrimaryType = "proton";  break;
            case 1000026056:    fPrimaryType = "iron";    break;
            default:            fPrimaryType = (fPrimaryId > 1000000000) ? "nucleus" : "unknown"; break;
        }
        fParticleTypeCounts[fPrimaryType]++;

        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }

        if (fEventCount <= 5) {
            std::cout << "\nEvent " << fEventCount
                      << ": Energy=" << fEnergy / 1e18 << " EeV"
                      << ", Primary=" << fPrimaryType
                      << " (ID=" << fPrimaryId << ")\n";
        }
    }

    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }

    // Periodic metrics print + confusion matrix content
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1, 1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2, 1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1, 2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2, 2, fTruePositives);
    }

    // Occasional training (small batches, precision-first)
    if (fIsTraining && fTrainingFeatures.size() >= 20 && (fEventCount % 5 == 0)) {
        const double vloss = TrainNetwork();

        const int b = fEventCount / 5;
        hTrainingLoss->SetBinContent(b, vloss);

        const int correct =
            (fTruePositives + fTrueNegatives);
        const int tot     =
            (fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives);
        if (tot > 0) {
            const double acc = 100.0 * double(correct) / double(tot);
            hAccuracyHistory->SetBinContent(b, acc);
        }

        if (vloss < fBestValidationLoss) {
            fBestValidationLoss = vloss;
            fEpochsSinceImprovement = 0;
            fNeuralNetwork->SaveWeights("best_" + fWeightsFileName);
        } else {
            ++fEpochsSinceImprovement;
        }
    }

    return eSuccess;
}

// ============================================================================

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // Station position -> distance to core
    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation  = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        const double stationX = detStation.GetPosition().GetX(siteCS);
        const double stationY = detStation.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((stationX - fCoreX) * (stationX - fCoreX) +
                              (stationY - fCoreY) * (stationY - fCoreY));
    } catch (...) {
        fDistance = -1.0;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    for (int p = 0; p < 3; ++p) {
        const int pmtId = p + firstPMT;
        if (!station.HasPMT(pmtId)) continue;

        const sevt::PMT& pmt = station.GetPMT(pmtId);

        std::vector<double> trace;
        const bool ok = ExtractTraceData(pmt, trace);
        if (!ok || trace.size() != 2048) continue;

        const double maxVal = *std::max_element(trace.begin(), trace.end());
        const double minVal = *std::min_element(trace.begin(), trace.end());
        if ((maxVal - minVal) < 10.0) continue; // trivial/noisy

        // Feature extraction
        fFeatures = ExtractEnhancedFeatures(trace);

        // A light physics tag (for diagnostics only; no +score bias)
        bool physicsPhotonLike = false;
        if (fFeatures.risetime_10_90 < 150.0 &&
            fFeatures.pulse_width      < 300.0 &&
            fFeatures.secondary_peak_ratio < 0.30 &&
            fFeatures.peak_charge_ratio    > 0.15) {
            physicsPhotonLike = true;
        }

        // Hist & running stats
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        UpdateFeatureStatistics(fFeatures);

        // Normalize + feature emphasis
        std::vector<double> x = NormalizeFeatures(fFeatures);
        if (x.size() >= 17) {
            x[1]  *= 2.0;   // risetime_10_90
            x[3]  *= 2.0;   // pulse_width
            x[7]  *= 1.5;   // peak_charge_ratio
            x[16] *= 1.5;   // secondary_peak_ratio
        }

        const double ml = fNeuralNetwork->Predict(x, false);

        // Spike penalty – helps reject narrow, muon-dominated hits mis-tagged as photons
        double penalty = 0.0;
        if (fFeatures.pulse_width < 120.0 && fFeatures.peak_charge_ratio > 0.35) penalty -= 0.20;
        if (fFeatures.num_peaks <= 2   && fFeatures.secondary_peak_ratio > 0.60) penalty -= 0.10;

        fPhotonScore = std::max(0.0, std::min(1.0, ml + penalty));
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // training sets (mild oversampling for photons)
        const bool isValidation = ((fStationCount % 10) == 0);
        if (fIsTraining) {
            if (!isValidation) {
                if (fIsActualPhoton) {
                    static std::mt19937 gen(1234567u);
                    std::normal_distribution<> noise(0.0, 0.02);
                    for (int k = 0; k < 2; ++k) {
                        std::vector<double> v = x;
                        for (double& z : v) z += noise(gen);
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

        ++fStationCount;

        const bool predPhoton = (fPhotonScore > fPhotonThreshold);

        // MLResult cache for other modules
        MLResult r;
        r.photonScore = fPhotonScore;
        r.identifiedAsPhoton = predPhoton;
        r.isActualPhoton = fIsActualPhoton;
        r.vemCharge = fFeatures.total_charge;
        r.features = fFeatures;
        r.primaryType = fPrimaryType;
        r.confidence  = fConfidence;
        fMLResultsMap[fStationId] = r;

        // confusion/marginals
        if (predPhoton) {
            ++fPhotonLikeCount;
            if (fIsActualPhoton) ++fTruePositives;
            else ++fFalsePositives;
        } else {
            ++fHadronLikeCount;
            if (!fIsActualPhoton) ++fTrueNegatives;
            else ++fFalseNegatives;
        }

        // hists
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);

        fMLTree->Fill();

        // Per-trace TSV (restored)
        if (gScoresTSV.is_open()) {
            gScoresTSV
                << fEventCount << '\t' << fStationId << '\t' << pmtId << '\t'
                << (fIsActualPhoton ? 1 : 0) << '\t' << (predPhoton ? 1 : 0) << '\t'
                << std::fixed << std::setprecision(6)
                << fPhotonScore << '\t' << ml << '\t' << penalty << '\t'
                << fDistance << '\t' << fFeatures.risetime_10_90 << '\t'
                << fFeatures.pulse_width << '\t' << fFeatures.num_peaks << '\t'
                << fFeatures.secondary_peak_ratio << '\t'
                << fFeatures.peak_charge_ratio << '\t'
                << fFeatures.total_charge << '\t'
                << fFeatures.peak_amplitude << '\t'
                << fPrimaryType
                << '\n';
        }

        // keep record for diagnostics at Finish
        _PredRec rec;
        rec.eventId = fEventCount;
        rec.stationId = fStationId;
        rec.pmtId = pmtId;
        rec.label = fIsActualPhoton ? 1 : 0;
        rec.score = fPhotonScore;
        rec.mlScore = ml;
        rec.penalty = penalty;
        rec.distance = fDistance;
        rec.rt10_90 = fFeatures.risetime_10_90;
        rec.width   = fFeatures.pulse_width;
        rec.nPeaks  = fFeatures.num_peaks;
        rec.secPeakRatio    = fFeatures.secondary_peak_ratio;
        rec.peakChargeRatio = fFeatures.peak_charge_ratio;
        rec.totalCharge     = fFeatures.total_charge;
        rec.peakAmp         = fFeatures.peak_amplitude;
        rec.physicsTag      = physicsPhotonLike;
        gAllRecs.push_back(rec);

        // Short debug line for early photon-like traces
        if ((fIsActualPhoton || physicsPhotonLike) && fStationCount <= 100) {
            std::cout << "  Station " << fStationId << " PMT " << pmtId
                      << ": Score=" << std::fixed << std::setprecision(3) << fPhotonScore
                      << " (True: " << fPrimaryType << ")"
                      << " Rise=" << fFeatures.risetime_10_90 << "ns"
                      << " Width=" << fFeatures.pulse_width << "ns"
                      << " PhysicsTag=" << (physicsPhotonLike ? "YES" : "NO") << "\n";
        }
    }
}

// ============================================================================

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
            const sevt::PMTSimData& sim = pmt.GetSimData();
            if (sim.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& t = sim.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                 sevt::StationConstants::eTotal);
                for (int i = 0; i < 2048; ++i) trace.push_back(t[i]);
                return true;
            }
        } catch (...) {}
    }
    return false;
}

PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
    EnhancedFeatures f;
    const int N = int(trace.size());
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    int peak_bin = 0;
    double peak_val = 0.0;
    double total = 0.0;

    std::vector<double> sig(N, 0.0);

    // baseline from first 100 bins
    double base = 0.0;
    const int nb = std::min(100, N);
    for (int i = 0; i < nb; ++i) base += trace[i];
    base /= std::max(1, nb);

    for (int i = 0; i < N; ++i) {
        const double v = trace[i] - base;
        sig[i] = (v > 0.0) ? v : 0.0;
        total += sig[i];
        if (sig[i] > peak_val) { peak_val = sig[i]; peak_bin = i; }
    }
    if (peak_val < 5.0 || total < 10.0) return f;

    f.peak_amplitude   = peak_val / ADC_PER_VEM;
    f.total_charge     = total / ADC_PER_VEM;
    f.peak_charge_ratio= f.peak_amplitude / (f.total_charge + 1e-3);

    const double v10 = 0.10 * peak_val;
    const double v50 = 0.50 * peak_val;
    const double v90 = 0.90 * peak_val;

    int i10r = 0, i50r = 0, i90r = peak_bin;
    for (int i = peak_bin; i >= 0; --i) {
        if (sig[i] <= v90 && i90r == peak_bin) i90r = i;
        if (sig[i] <= v50 && i50r == 0)       i50r = i;
        if (sig[i] <= v10) { i10r = i; break; }
    }
    int i90f = peak_bin, i10f = N - 1;
    for (int i = peak_bin; i < N; ++i) {
        if (sig[i] <= v90 && i90f == peak_bin) i90f = i;
        if (sig[i] <= v10) { i10f = i; break; }
    }

    f.risetime_10_50 = std::abs(i50r - i10r) * NS_PER_BIN;
    f.risetime_10_90 = std::abs(i90r - i10r) * NS_PER_BIN;
    f.falltime_90_10 = std::abs(i10f - i90f) * NS_PER_BIN;

    // FWHM
    const double hm = 0.5 * peak_val;
    int iHalfR = i10r, iHalfF = i10f;
    for (int i = i10r; i <= peak_bin; ++i) { if (sig[i] >= hm) { iHalfR = i; break; } }
    for (int i = peak_bin; i < N; ++i)      { if (sig[i] <= hm) { iHalfF = i; break; } }
    f.pulse_width = std::abs(iHalfF - iHalfR) * NS_PER_BIN;

    const double rise = f.risetime_10_90;
    const double fall = f.falltime_90_10;
    f.asymmetry = (fall - rise) / (fall + rise + 1e-3);

    // moments
    double mean_t = 0.0;
    for (int i = 0; i < N; ++i) mean_t += i * sig[i];
    mean_t /= (total + 1e-3);

    double var = 0.0, skew = 0.0, kurt = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = i - mean_t;
        const double w = sig[i] / (total + 1e-3);
        var  += d * d * w;
        skew += d * d * d * w;
        kurt += d * d * d * d * w;
    }
    const double sd = std::sqrt(var + 1e-3);
    f.time_spread = sd * NS_PER_BIN;
    f.skewness    = skew / (sd*sd*sd + 1e-3);
    f.kurtosis    = kurt / (var*var + 1e-3) - 3.0;

    // early/late
    const int q = N / 4;
    double early = 0.0, late = 0.0;
    for (int i = 0; i < q; ++i) early += sig[i];
    for (int i = 3*q; i < N; ++i) late += sig[i];
    f.early_fraction = early / (total + 1e-3);
    f.late_fraction  = late  / (total + 1e-3);

    // smoothness
    double ss = 0.0; int cnt = 0;
    for (int i = 1; i < N - 1; ++i) {
        if (sig[i] > 0.1 * peak_val) {
            const double s2 = sig[i+1] - 2.0 * sig[i] + sig[i-1];
            ss += s2 * s2; ++cnt;
        }
    }
    f.smoothness = std::sqrt(ss / (cnt + 1));

    // high-frequency content
    double hf = 0.0;
    for (int i = 1; i < N - 1; ++i) {
        const double d = sig[i+1] - sig[i-1];
        hf += d * d;
    }
    f.high_freq_content = hf / (total * total + 1e-3);

    // peak counting (stricter threshold)
    f.num_peaks = 0;
    double sec = 0.0;
    const double thr = 0.25 * peak_val;
    for (int i = 1; i < N - 1; ++i) {
        if (sig[i] > thr && sig[i] > sig[i-1] && sig[i] > sig[i+1]) {
            ++f.num_peaks;
            if (i != peak_bin && sig[i] > sec) sec = sig[i];
        }
    }
    f.secondary_peak_ratio = sec / (peak_val + 1e-3);

    return f;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& features)
{
    std::vector<double> raw = {
        features.risetime_10_50, features.risetime_10_90, features.falltime_90_10,
        features.pulse_width, features.asymmetry, features.peak_amplitude,
        features.total_charge, features.peak_charge_ratio, features.smoothness,
        features.kurtosis, features.skewness, features.early_fraction,
        features.late_fraction, features.time_spread, features.high_freq_content,
        double(features.num_peaks), features.secondary_peak_ratio
    };

    static int n = 0; // running count in this TU
    ++n;

    for (size_t i = 0; i < raw.size(); ++i) {
        const double delta = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += delta / n;
        const double delta2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += delta * delta2;
    }
    if (n > 1) {
        for (size_t i = 0; i < raw.size(); ++i) {
            fFeatureStdDevs[i] = std::sqrt(fFeatureStdDevs[i] / (n - 1));
            if (fFeatureStdDevs[i] < 1e-3) fFeatureStdDevs[i] = 1.0;
        }
    }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& f)
{
    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width, f.asymmetry,
        f.peak_amplitude, f.total_charge, f.peak_charge_ratio, f.smoothness, f.kurtosis,
        f.skewness, f.early_fraction, f.late_fraction, f.time_spread, f.high_freq_content,
        double(f.num_peaks), f.secondary_peak_ratio
    };
    const std::vector<double> mins = {0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
    const std::vector<double> maxs = {500,1000,1000,1000,1,10,100,1,100,20,5,1,1,1000,10,10,1};

    std::vector<double> z; z.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        double v = (raw[i] - mins[i]) / (maxs[i] - mins[i] + 1e-3);
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        z.push_back(v);
    }
    return z;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;

    std::cout << "\n  Training with " << fTrainingFeatures.size() << " samples...";

    std::vector<std::vector<double>> Bx;
    std::vector<int> By;

    const int kmax = std::min(100, int(fTrainingFeatures.size()));
    for (int i = 0; i < kmax; ++i) {
        Bx.push_back(fTrainingFeatures[i]);
        By.push_back(fTrainingLabels[i]);
    }

    const int nPho = std::count(By.begin(), By.end(), 1);
    const int nHad = int(By.size()) - nPho;
    std::cout << " (P:" << nPho << " H:" << nHad << ")";

    const double lr = 0.01;
    double tot = 0.0;
    for (int e = 0; e < 10; ++e) tot += fNeuralNetwork->Train(Bx, By, lr);
    const double trainLoss = tot / 10.0;

    std::cout << " Loss: " << std::fixed << std::setprecision(4) << trainLoss;

    // Validation and slight threshold recalibration for F1 (0.40–0.80)
    double valLoss = 0.0;
    if (!fValidationFeatures.empty()) {
        const int N = int(fValidationFeatures.size());
        std::vector<double> pred(N, 0.0);
        for (int i = 0; i < N; ++i) pred[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

        int correct = 0;
        for (int i = 0; i < N; ++i) {
            const int y  = fValidationLabels[i];
            const double p = pred[i];
            valLoss -= y * std::log(p + 1e-7) + (1 - y) * std::log(1 - p + 1e-7);
            const int yhat = (p > fPhotonThreshold) ? 1 : 0;
            if (yhat == y) ++correct;
        }
        valLoss /= N;
        const double acc = 100.0 * double(correct) / double(N);
        std::cout << " Val: " << valLoss << " (Acc: " << acc << "%)";

        double bestF1 = -1.0, bestThr = fPhotonThreshold;
        for (double thr = 0.40; thr <= 0.80 + 1e-12; thr += 0.02) {
            int TP=0, FP=0, TN=0, FN=0;
            for (int i = 0; i < N; ++i) {
                const int y = fValidationLabels[i];
                const int yhat = (pred[i] > thr) ? 1 : 0;
                if (yhat==1 && y==1) ++TP;
                else if (yhat==1 && y==0) ++FP;
                else if (yhat==0 && y==0) ++TN;
                else ++FN;
            }
            const double P = (TP+FP>0) ? double(TP)/double(TP+FP) : 0.0;
            const double R = (TP+FN>0) ? double(TP)/double(TP+FN) : 0.0;
            const double F1= (P+R>0) ? (2.0*P*R/(P+R)) : 0.0;
            if (F1 > bestF1) { bestF1 = F1; bestThr = thr; }
        }
        // nudge by 20% toward best
        fPhotonThreshold = 0.8 * fPhotonThreshold + 0.2 * bestThr;

        const int b = fEventCount / 5;
        hValidationLoss->SetBinContent(b, valLoss);
    }

    std::cout << std::endl;

    if (fTrainingFeatures.size() > 1000) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin() + 200);
        fTrainingLabels  .erase(fTrainingLabels  .begin(), fTrainingLabels  .begin() + 200);
    }
    ++fTrainingStep;
    return valLoss;
}

// ============================================================================

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) return;

    const double accuracy  = 100.0 * double(fTruePositives + fTrueNegatives) / double(total);
    const double precision = (fTruePositives + fFalsePositives > 0)
                           ? 100.0 * double(fTruePositives) / double(fTruePositives + fFalsePositives) : 0.0;
    const double recall    = (fTruePositives + fFalseNegatives > 0)
                           ? 100.0 * double(fTruePositives) / double(fTruePositives + fFalseNegatives) : 0.0;
    const double f1        = (precision + recall > 0.0)
                           ? (2.0 * precision * recall / (precision + recall)) : 0.0;

    const double photonFrac = (fPhotonLikeCount + fHadronLikeCount > 0)
                            ? 100.0 * double(fPhotonLikeCount) / double(fPhotonLikeCount + fHadronLikeCount) : 0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << photonFrac << "%"
              << " │ " << std::setw(7) << accuracy  << "%"
              << " │ " << std::setw(8) << precision << "%"
              << " │ " << std::setw(7) << recall    << "%"
              << " │ " << std::setw(7) << f1        << "%│\n";

    // Old event-line log format (diagnostics)
    if (fLogFile.is_open() && (fEventCount % 10 == 0)) {
        fLogFile << "Event " << fEventCount
                 << " - Acc:" << accuracy
                 << "% Prec:" << precision
                 << "% Rec:"  << recall
                 << "% F1:"   << f1 << "%\n";
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    std::cout << "\n==========================================\n";
    std::cout << "PERFORMANCE METRICS\n";
    std::cout << "==========================================\n";

    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) {
        std::cout << "No predictions made yet!\n";
        return;
    }

    const double accuracy  = 100.0 * double(fTruePositives + fTrueNegatives) / double(total);
    const double precision = (fTruePositives + fFalsePositives > 0)
                           ? 100.0 * double(fTruePositives) / double(fTruePositives + fFalsePositives) : 0.0;
    const double recall    = (fTruePositives + fFalseNegatives > 0)
                           ? 100.0 * double(fTruePositives) / double(fTruePositives + fFalseNegatives) : 0.0;
    const double f1        = (precision + recall > 0.0)
                           ? 2.0 * precision * recall / (precision + recall) : 0.0;

    std::cout << "Accuracy:  " << std::fixed << std::setprecision(1) << accuracy  << "%\n";
    std::cout << "Precision: " << precision << "%\n";
    std::cout << "Recall:    " << recall    << "%\n";
    std::cout << "F1-Score:  " << f1        << "%\n\n";

    std::cout << "CONFUSION MATRIX:\n";
    std::cout << "                Predicted\n";
    std::cout << "             Hadron   Photon\n";
    std::cout << "Actual Hadron  " << std::setw(6) << fTrueNegatives << "   " << std::setw(6) << fFalsePositives << "\n";
    std::cout << "       Photon  " << std::setw(6) << fFalseNegatives << "   " << std::setw(6) << fTruePositives  << "\n\n";

    std::cout << "Total Stations: " << fStationCount << "\n";
    std::cout << "Photon-like: " << fPhotonLikeCount << " ("
              << 100.0 * double(fPhotonLikeCount) / std::max(1.0, double(fPhotonLikeCount + fHadronLikeCount)) << "%)\n";
    std::cout << "Hadron-like: " << fHadronLikeCount << " ("
              << 100.0 * double(fHadronLikeCount) / std::max(1.0, double(fPhotonLikeCount + fHadronLikeCount)) << "%)\n";

    std::cout << "\nPARTICLE TYPE BREAKDOWN:\n";
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
    }
}

// ============================================================================

void PhotonTriggerML::SaveAndDisplaySummary()
{
    std::cout << "\n==========================================\n";
    std::cout << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n";
    std::cout << "==========================================\n";
    std::cout << "Events processed: " << fEventCount << "\n";
    std::cout << "Stations analyzed: " << fStationCount << "\n";

    CalculatePerformanceMetrics();

    // Persist weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // ROOT output
    if (fOutputFile) {
        fOutputFile->cd();
        if (fMLTree) {
            fMLTree->Write();
            std::cout << "Wrote " << fMLTree->GetEntries() << " entries to tree\n";
        }
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

        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = nullptr;
    }

    // Lightweight ROOT summary w/ confusion matrix (kept)
    {
        TFile fsum("photon_trigger_summary.root", "RECREATE");
        if (!fsum.IsZombie()) {
            if (hConfusionMatrix) hConfusionMatrix->Write("ConfusionMatrix");
            if (hPhotonScore)     hPhotonScore->Write("PhotonScore_All");
            if (hPhotonScorePhotons) hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
            if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
            if (hConfidence)      hConfidence->Write("Confidence");
            fsum.Close();
        }
    }

    // Restored human-readable summary text (BaseThreshold=…)
    {
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            const int TP = fTruePositives, FP = fFalsePositives, TN = fTrueNegatives, FN = fFalseNegatives;
            const int total = TP + FP + TN + FN;
            const double acc = (total>0) ? 100.0 * double(TP + TN) / double(total) : 0.0;
            const double prec= (TP+FP>0)? 100.0 * double(TP) / double(TP+FP) : 0.0;
            const double rec = (TP+FN>0)? 100.0 * double(TP) / double(TP+FN) : 0.0;
            const double f1  = (prec+rec>0)? (2.0*prec*rec/(prec+rec)) : 0.0;
            txt << "PhotonTriggerML Summary\n";
            txt << "BaseThreshold=" << std::fixed << std::setprecision(6) << fPhotonThreshold << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy=" << acc << "%  Precision=" << prec << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
            txt.close();
        }
    }

    // Diagnostics: misclassified & threshold sweep
    {
        // Final threshold classification
        std::ofstream bad("photon_trigger_misclassified.tsv");
        if (bad.is_open()) {
            bad << "type\ttrueLabel\tscore\tmlScore\tpenalty\tevent\tstation\tpmt\tdist_m\t"
                << "rise10_90_ns\twidth_ns\tnPeaks\tsecPeakRatio\tpeakChargeRatio\t"
                << "totalCharge_VEM\tpeakAmp_VEM\n";
            for (const auto& r : gAllRecs) {
                const bool pred = (r.score > fPhotonThreshold);
                if (pred == (r.label==1)) continue;
                bad << (pred ? "FP" : "FN") << '\t' << r.label << '\t'
                    << std::fixed << std::setprecision(6)
                    << r.score << '\t' << r.mlScore << '\t' << r.penalty << '\t'
                    << r.eventId << '\t' << r.stationId << '\t' << r.pmtId << '\t'
                    << r.distance << '\t' << r.rt10_90 << '\t' << r.width << '\t'
                    << r.nPeaks << '\t' << r.secPeakRatio << '\t' << r.peakChargeRatio << '\t'
                    << r.totalCharge << '\t' << r.peakAmp << '\n';
            }
            bad.close();
        }
        // Sweep
        std::ofstream sw("photon_trigger_threshold_sweep.tsv");
        if (sw.is_open()) {
            sw << "thr\tTP\tFP\tTN\tFN\tprecision\trecall\tF1\n";
            double bestF1 = -1.0, bestThr = fPhotonThreshold;
            for (double thr = 0.40; thr <= 0.90 + 1e-12; thr += 0.02) {
                int TP=0, FP=0, TN=0, FN=0;
                for (const auto& r : gAllRecs) {
                    const bool pred = (r.score > thr);
                    if (pred && r.label==1) ++TP;
                    else if (pred && r.label==0) ++FP;
                    else if (!pred && r.label==0) ++TN;
                    else ++FN;
                }
                const double P = (TP+FP>0)? double(TP)/double(TP+FP) : 0.0;
                const double R = (TP+FN>0)? double(TP)/double(TP+FN) : 0.0;
                const double F1= (P+R>0) ? (2.0*P*R/(P+R)) : 0.0;
                sw << std::fixed << std::setprecision(3) << thr << '\t'
                   << TP << '\t' << FP << '\t' << TN << '\t' << FN << '\t'
                   << std::setprecision(6) << P << '\t' << R << '\t' << F1 << '\n';
                if (F1 > bestF1) { bestF1 = F1; bestThr = thr; }
            }
            sw << "# best_thr\t" << std::fixed << std::setprecision(3) << bestThr
               << "\tbest_F1\t" << std::setprecision(6) << bestF1 << "\n";
            sw.close();
        }
    }

    // In-place annotation of existing trace files (skip in signal handler)
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
            if (X.size() >= 17) {
                X[1]  *= 2.0;
                X[3]  *= 2.0;
                X[7]  *= 1.5;
                X[16] *= 1.5;
            }
            const double ml = fNeuralNetwork->Predict(X, false);

            double penalty = 0.0;
            if (F.pulse_width < 120.0 && F.peak_charge_ratio > 0.35) penalty -= 0.20;
            if (F.num_peaks <= 2   && F.secondary_peak_ratio > 0.60) penalty -= 0.10;

            score   = std::max(0.0, std::min(1.0, ml + penalty));
            isPhoton= (score > fPhotonThreshold);
            return true;
        };

        std::function<void(TDirectory*)> annotate_dir = [&](TDirectory* dir){
            if (!dir) return;
            TIter next(dir->GetListOfKeys());
            TKey* key = nullptr;
            while ((key = (TKey*)next())) {
                TObject* obj = dir->Get(key->GetName());
                if (!obj) continue;

                if (obj->InheritsFrom(TDirectory::Class())) {
                    annotate_dir((TDirectory*)obj);
                } else if (obj->InheritsFrom(TH1::Class())) {
                    TH1* h = (TH1*)obj;
                    double sc=0.0; bool ph=false;
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
                    TH1* h = nullptr;
                    if (TList* prim = c->GetListOfPrimitives()) {
                        TIter nx(prim);
                        while (TObject* po = nx()) {
                            if (po->InheritsFrom(TH1::Class())) { h = (TH1*)po; break; }
                        }
                    }
                    if (h) {
                        double sc=0.0; bool ph=false;
                        if (score_from_hist(h, sc, ph)) {
                            const std::string baseC = strip_ml(c->GetTitle() ? c->GetTitle() : "");
                            const std::string baseH = strip_ml(h->GetTitle() ? h->GetTitle() : "");
                            std::ostringstream tc, th;
                            tc << baseC << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                            th << baseH << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                            c->SetTitle(tc.str().c_str());
                            h->SetTitle(th.str().c_str());
                            c->Modified(); c->Update();
                            dir->cd();
                            c->Write(c->GetName(), TObject::kOverwrite);
                        }
                    }
                }

                // Do NOT delete 'obj'; owned by directory/file. Avoids ROOT double-delete.
            }
        };

        const char* candidates[] = {
            "pmt_traces_1EeV.root", "pmt_Traces_1EeV.root",
            "pmt_traces_1eev.root", "pmt_Traces_1eev.root"
        };

        TFile* f = nullptr;
        for (const char* name : candidates) {
            f = TFile::Open(name, "UPDATE");
            if (f && !f->IsZombie()) break;
            if (f) { delete f; f = nullptr; }
        }
        if (f && !f->IsZombie()) {
            if (!f->TestBit(TFile::kRecovered)) {
                annotate_dir(f);
                f->Write("", TObject::kOverwrite);
                std::cout << "Annotated ML tags in existing trace file: " << f->GetName() << "\n";
            } else {
                std::cout << "Trace file appears recovered; skipping ML annotation for safety.\n";
            }
            f->Close(); delete f;
        } else {
            std::cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    if (gScoresTSV.is_open()) gScoresTSV.close();
    if (fLogFile.is_open())   fLogFile.close();

    std::cout << "==========================================\n";
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    SaveAndDisplaySummary();
    return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& result)
{
    auto it = fMLResultsMap.find(stationId);
    if (it != fMLResultsMap.end()) { result = it->second; return true; }
    return false;
}

void PhotonTriggerML::ClearMLResults() { fMLResultsMap.clear(); }

// Keep header-declared stub (used by older code paths); no-op here.
void PhotonTriggerML::WriteMLAnalysisToLog() {}

