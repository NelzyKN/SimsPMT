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

// Static instance pointer for signal handler
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

// Static ML results map
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Flag: SaveAndDisplaySummary() invoked from a signal handler?
static volatile sig_atomic_t gCalledFromSignal = 0;

// Keep the base threshold (for summary file "BaseThreshold=...")
static double gBaseThreshold = 0.0;

// Per-station diagnostic stream (TSV)
static std::ofstream gScoreCSV;
static bool gScoreCSVOpen = false;

// ============================================================================
// Minimal, safe signal handler
// ============================================================================
static void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        std::cout << "\n\nInterrupt received. Saving PhotonTriggerML summary...\n" << std::endl;

        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try {
                PhotonTriggerML::fInstance->SaveAndDisplaySummary();
            } catch (...) {
                // swallow to guarantee termination
            }
        }

        std::signal(signal, SIG_DFL);
        std::raise(signal);
        _Exit(0);
    }
}

// ============================================================================
// Physics-Based Neural Network (simple MLP, SGD+momentum)
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.1),
    fIsQuantized(false), fQuantizationScale(127.0)
{
}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize = input_size;
    fHidden1Size = hidden1_size;
    fHidden2Size = hidden2_size;

    std::cout << "Initializing Physics-Based Neural Network: "
              << input_size << " -> " << hidden1_size << " -> " << hidden2_size << " -> 1" << std::endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.resize(hidden1_size, std::vector<double>(input_size));
    for (int i = 0; i < hidden1_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            fWeights1[i][j] = dist(gen) / std::sqrt(static_cast<double>(input_size));
        }
    }

    fWeights2.resize(hidden2_size, std::vector<double>(hidden1_size));
    for (int i = 0; i < hidden2_size; ++i) {
        for (int j = 0; j < hidden1_size; ++j) {
            fWeights2[i][j] = dist(gen) / std::sqrt(static_cast<double>(hidden1_size));
        }
    }

    fWeights3.resize(1, std::vector<double>(hidden2_size));
    for (int j = 0; j < hidden2_size; ++j) {
        fWeights3[0][j] = dist(gen) / std::sqrt(static_cast<double>(hidden2_size));
    }

    fBias1.assign(hidden1_size, 0.0);
    fBias2.assign(hidden2_size, 0.0);
    fBias3 = 0.0;

    // Momentum buffers
    fMomentum1_w1.assign(hidden1_size, std::vector<double>(input_size, 0.0));
    fMomentum2_w1.assign(hidden1_size, std::vector<double>(input_size, 0.0));
    fMomentum1_w2.assign(hidden2_size, std::vector<double>(hidden1_size, 0.0));
    fMomentum2_w2.assign(hidden2_size, std::vector<double>(hidden1_size, 0.0));
    fMomentum1_w3.assign(1, std::vector<double>(hidden2_size, 0.0));
    fMomentum2_w3.assign(1, std::vector<double>(hidden2_size, 0.0));

    fMomentum1_b1.assign(hidden1_size, 0.0);
    fMomentum2_b1.assign(hidden1_size, 0.0);
    fMomentum1_b2.assign(hidden2_size, 0.0);
    fMomentum2_b2.assign(hidden2_size, 0.0);
    fMomentum1_b3 = 0.0;
    fMomentum2_b3 = 0.0;

    fTimeStep = 0;

    std::cout << "Neural Network initialized." << std::endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& features, bool training)
{
    if (static_cast<int>(features.size()) != fInputSize) {
        return 0.5;
    }

    // Hidden 1
    std::vector<double> h1(fHidden1Size, 0.0);
    for (int i = 0; i < fHidden1Size; ++i) {
        double sum = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) sum += fWeights1[i][j] * features[j];
        h1[i] = 1.0 / (1.0 + std::exp(-sum));
        if (training) {
            double r = std::rand() / double(RAND_MAX);
            if (r < fDropoutRate) h1[i] = 0.0;
        }
    }

    // Hidden 2
    std::vector<double> h2(fHidden2Size, 0.0);
    for (int i = 0; i < fHidden2Size; ++i) {
        double sum = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) sum += fWeights2[i][j] * h1[j];
        h2[i] = 1.0 / (1.0 + std::exp(-sum));
        if (training) {
            double r = std::rand() / double(RAND_MAX);
            if (r < fDropoutRate) h2[i] = 0.0;
        }
    }

    // Output
    double out = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) out += fWeights3[0][j] * h2[j];
    return 1.0 / (1.0 + std::exp(-out));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& features,
                                             const std::vector<int>& labels,
                                             double learning_rate)
{
    if (features.empty() || features.size() != labels.size()) return -1.0;

    const int N = static_cast<int>(features.size());
    double total_loss = 0.0;

    // Softer class weighting (precision-first)
    const int num_photons = std::count(labels.begin(), labels.end(), 1);
    const double w_photon = (num_photons > 0) ? 1.5 : 1.0;
    const double w_hadron = 1.0;

    // Accumulators
    std::vector<std::vector<double>> gw1(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    std::vector<std::vector<double>> gw2(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    std::vector<std::vector<double>> gw3(1, std::vector<double>(fHidden2Size, 0.0));
    std::vector<double> gb1(fHidden1Size, 0.0);
    std::vector<double> gb2(fHidden2Size, 0.0);
    double gb3 = 0.0;

    for (int n = 0; n < N; ++n) {
        const auto& x = features[n];
        const int y = labels[n];
        const double w = (y == 1) ? w_photon : w_hadron;

        // Forward
        std::vector<double> h1(fHidden1Size, 0.0), h1r(fHidden1Size, 0.0);
        for (int i = 0; i < fHidden1Size; ++i) {
            double s = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * x[j];
            h1r[i] = s;
            h1[i] = 1.0 / (1.0 + std::exp(-s));
        }
        std::vector<double> h2(fHidden2Size, 0.0), h2r(fHidden2Size, 0.0);
        for (int i = 0; i < fHidden2Size; ++i) {
            double s = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
            h2r[i] = s;
            h2[i] = 1.0 / (1.0 + std::exp(-s));
        }
        double oraw = fBias3;
        for (int j = 0; j < fHidden2Size; ++j) oraw += fWeights3[0][j] * h2[j];
        const double p = 1.0 / (1.0 + std::exp(-oraw));

        // Loss
        total_loss += -w * (y * std::log(p + 1e-7) + (1 - y) * std::log(1 - p + 1e-7));

        // Backprop
        const double dL_do = w * (p - y);
        for (int j = 0; j < fHidden2Size; ++j) gw3[0][j] += dL_do * h2[j];
        gb3 += dL_do;

        std::vector<double> dh2(fHidden2Size, 0.0);
        for (int j = 0; j < fHidden2Size; ++j) {
            double t = fWeights3[0][j] * dL_do;
            dh2[j] = t * h2[j] * (1.0 - h2[j]);
        }
        for (int i = 0; i < fHidden2Size; ++i) {
            for (int j = 0; j < fHidden1Size; ++j) gw2[i][j] += dh2[i] * h1[j];
            gb2[i] += dh2[i];
        }

        std::vector<double> dh1(fHidden1Size, 0.0);
        for (int j = 0; j < fHidden1Size; ++j) {
            double t = 0.0;
            for (int i = 0; i < fHidden2Size; ++i) t += fWeights2[i][j] * dh2[i];
            dh1[j] = t * h1[j] * (1.0 - h1[j]);
        }
        for (int i = 0; i < fHidden1Size; ++i) {
            for (int j = 0; j < fInputSize; ++j) gw1[i][j] += dh1[i] * x[j];
            gb1[i] += dh1[i];
        }
    }

    // SGD + momentum
    fTimeStep++;
    const double momentum = 0.9;

    for (int i = 0; i < fHidden1Size; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            const double g = gw1[i][j] / N;
            fMomentum1_w1[i][j] = momentum * fMomentum1_w1[i][j] - learning_rate * g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        const double gb = gb1[i] / N;
        fMomentum1_b1[i] = momentum * fMomentum1_b1[i] - learning_rate * gb;
        fBias1[i] += fMomentum1_b1[i];
    }

    for (int i = 0; i < fHidden2Size; ++i) {
        for (int j = 0; j < fHidden1Size; ++j) {
            const double g = gw2[i][j] / N;
            fMomentum1_w2[i][j] = momentum * fMomentum1_w2[i][j] - learning_rate * g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        const double gb = gb2[i] / N;
        fMomentum1_b2[i] = momentum * fMomentum1_b2[i] - learning_rate * gb;
        fBias2[i] += fMomentum1_b2[i];
    }

    for (int j = 0; j < fHidden2Size; ++j) {
        const double g = gw3[0][j] / N;
        fMomentum1_w3[0][j] = momentum * fMomentum1_w3[0][j] - learning_rate * g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }
    const double gb = gb3 / N;
    fMomentum1_b3 = momentum * fMomentum1_b3 - learning_rate * gb;
    fBias3 += fMomentum1_b3;

    return total_loss / N;
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
    for (double b : fBias1) {
        file << b << " ";
    }
    file << "\n";

    for (const auto& row : fWeights2) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias2) {
        file << b << " ";
    }
    file << "\n";

    for (double w : fWeights3[0]) {
        file << w << " ";
    }
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

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fWeights3.assign(1, std::vector<double>(fHidden2Size, 0.0));
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

void PhotonTriggerML::NeuralNetwork::QuantizeWeights()
{
    fIsQuantized = true;
}

// ============================================================================
// PhotonTriggerML (precision‑first, old log format + extra diagnostics)
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
    fPhotonThreshold(0.80), // classic baseline; auto-tuned during training
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_physics.root"),
    fWeightsFileName("photon_trigger_weights_physics.txt"),
    fLoadPretrainedWeights(true)
{
    fInstance = this;

    fFeatureMeans.assign(17, 0.0);
    fFeatureStdDevs.assign(17, 1.0);

    std::cout << "\n==========================================\n";
    std::cout << "PhotonTriggerML Constructor (PHYSICS)\n";
    std::cout << "Output file: " << fOutputFileName << "\n";
    std::cout << "Log file:    " << fLogFileName << "\n";
    std::cout << "==========================================\n";
}

PhotonTriggerML::~PhotonTriggerML()
{
    std::cout << "PhotonTriggerML Destructor called\n";
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization");

    // Log file
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

    // Keep the base threshold for "BaseThreshold=" in the summary
    gBaseThreshold = fPhotonThreshold;

    // Network
    fNeuralNetwork->Initialize(17, 8, 4);
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        std::cout << "Loaded pre-trained weights from " << fWeightsFileName << std::endl;
        fIsTraining = false; // inference-only if weights present
    } else {
        std::cout << "Starting with random weights (training mode)\n";
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
    hPhotonScore = new TH1D("hPhotonScore", "ML Photon Score (All);Score;Count", 50, 0, 1);
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons", "ML Score (True Photons);Score;Count", 50, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons", "ML Score (True Hadrons);Score;Count", 50, 0, 1);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence = new TH1D("hConfidence", "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime = new TH1D("hRisetime", "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry = new TH1D("hAsymmetry", "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis = new TH1D("hKurtosis", "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score", 50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 50, 0, 3000, 50, 0, 1);

    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual",
                                2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");

    hTrainingLoss = new TH1D("hTrainingLoss", "Training Loss;Batch;Loss", 10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss", 10000, 0, 10000);
    hAccuracyHistory = new TH1D("hAccuracyHistory", "Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    // TSV diagnostics
    gScoreCSV.open("photon_trigger_scores.tsv");
    if (gScoreCSV.good()) {
        gScoreCSV << "event\tstation\tpmt\tenergy_EeV\tdistance_m\tlabel\tpred_score\tidentified_as_photon"
                  << "\trisetime_10_90_ns\tpulse_width_ns\tpeak_over_charge\tsecondary_peak_ratio\n";
        gScoreCSVOpen = true;
    }

    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    std::cout << "Initialization complete!\n";
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    // Header every 50 events
    if (fEventCount % 50 == 1) {
        std::cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n";
        std::cout <<   "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│\n";
        std::cout <<   "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n";
    }

    // Shower info
    fEnergy = 0.0;
    fCoreX = 0.0; fCoreY = 0.0;
    fPrimaryId = 0;
    fPrimaryType = "Unknown";
    fIsActualPhoton = false;

    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        fPrimaryId = shower.GetPrimaryParticle();
        switch (fPrimaryId) {
            case 22:  fPrimaryType = "photon";  fIsActualPhoton = true; break;
            case 11: case -11: fPrimaryType = "electron"; break;
            case 2212: fPrimaryType = "proton"; break;
            case 1000026056: fPrimaryType = "iron"; break;
            default: fPrimaryType = (fPrimaryId > 1000000000) ? "nucleus" : "unknown";
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
                      << ": Energy=" << fEnergy/1e18 << " EeV"
                      << ", Primary=" << fPrimaryType
                      << " (ID=" << fPrimaryId << ")\n";
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
        hConfusionMatrix->SetBinContent(1, 1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2, 1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1, 2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2, 2, fTruePositives);
    }

    // Lightweight log line with station count (consumes 'stationsInEvent')
    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount << " Stations=" << stationsInEvent << "\n";
    }

    // Training
    if (fIsTraining && fTrainingFeatures.size() >= 20 && (fEventCount % 5 == 0)) {
        const double val_loss = TrainNetwork();

        const int batch_num = fEventCount / 5;
        hTrainingLoss->SetBinContent(batch_num, val_loss);

        const int correct = fTruePositives + fTrueNegatives;
        const int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total > 0) {
            const double accuracy = 100.0 * correct / total;
            hAccuracyHistory->SetBinContent(batch_num, accuracy);
        }

        if (val_loss < fBestValidationLoss) {
            fBestValidationLoss = val_loss;
            fEpochsSinceImprovement = 0;
            fNeuralNetwork->SaveWeights("best_" + fWeightsFileName);
        } else {
            fEpochsSinceImprovement++;
        }
    }

    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // Position / distance
    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        const double stationX = detStation.GetPosition().GetX(siteCS);
        const double stationY = detStation.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((stationX - fCoreX)*(stationX - fCoreX) +
                              (stationY - fCoreY)*(stationY - fCoreY));
    } catch (...) {
        fDistance = -1.0;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    for (int p = 0; p < 3; ++p) {
        const int pmtId = p + firstPMT;
        if (!station.HasPMT(pmtId)) continue;

        const sevt::PMT& pmt = station.GetPMT(pmtId);

        std::vector<double> trace_data;
        const bool traceFound = ExtractTraceData(pmt, trace_data);
        if (!traceFound || trace_data.size() != 2048) continue;

        const double maxVal = *std::max_element(trace_data.begin(), trace_data.end());
        const double minVal = *std::min_element(trace_data.begin(), trace_data.end());
        if (maxVal - minVal < 10.0) continue;

        // Features
        fFeatures = ExtractEnhancedFeatures(trace_data);
        const bool physicsPhotonLike = (fFeatures.risetime_10_90 < 150.0 &&
                                        fFeatures.pulse_width      < 300.0 &&
                                        fFeatures.secondary_peak_ratio < 0.30 &&
                                        fFeatures.peak_charge_ratio    > 0.15);

        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        UpdateFeatureStatistics(fFeatures);

        std::vector<double> X = NormalizeFeatures(fFeatures);
        // Physics emphasis (no biasing of score, only feature scaling):
        X[1]  *= 2.0;  // risetime_10_90
        X[3]  *= 2.0;  // pulse_width
        X[7]  *= 1.5;  // peak_charge_ratio
        X[16] *= 1.5;  // secondary_peak_ratio

        double ml_score = fNeuralNetwork->Predict(X, false);

        // Spike-dominance penalties (help suppress single-muon spikes)
        double penalty = 0.0;
        if (fFeatures.pulse_width < 120.0 && fFeatures.peak_charge_ratio > 0.35) penalty -= 0.20;
        if (fFeatures.num_peaks  <= 2     && fFeatures.secondary_peak_ratio > 0.60) penalty -= 0.10;

        fPhotonScore = std::max(0.0, std::min(1.0, ml_score + penalty));
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // Training caches (reduced oversampling)
        const bool isValidation = (fStationCount % 10 == 0);
        if (fIsTraining) {
            if (!isValidation) {
                if (fIsActualPhoton) {
                    for (int copy = 0; copy < 2; ++copy) {
                        std::vector<double> varX = X;
                        static std::mt19937 gen{1234567u};
                        std::normal_distribution<> noise(0.0, 0.02);
                        for (double &v : varX) v += noise(gen);
                        fTrainingFeatures.push_back(varX);
                        fTrainingLabels.push_back(1);
                    }
                } else {
                    fTrainingFeatures.push_back(X);
                    fTrainingLabels.push_back(0);
                }
            } else {
                fValidationFeatures.push_back(X);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }

        fStationCount++;
        const bool identifiedAsPhoton = (fPhotonScore > fPhotonThreshold);

        // Store shared result
        MLResult mlResult;
        mlResult.photonScore        = fPhotonScore;
        mlResult.identifiedAsPhoton = identifiedAsPhoton;
        mlResult.isActualPhoton     = fIsActualPhoton;
        mlResult.vemCharge          = fFeatures.total_charge;
        mlResult.features           = fFeatures;
        mlResult.primaryType        = fPrimaryType;
        mlResult.confidence         = fConfidence;
        fMLResultsMap[fStationId]   = mlResult;

        // Counters
        if (identifiedAsPhoton) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
        }

        // Hists
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);

        // Tree
        if (fMLTree) fMLTree->Fill();

        // TSV diagnostics
        if (gScoreCSVOpen) {
            gScoreCSV << fEventCount << '\t'
                      << fStationId  << '\t'
                      << pmtId       << '\t'
                      << std::fixed << std::setprecision(6) << (fEnergy/1e18) << '\t'
                      << std::setprecision(3) << fDistance << '\t'
                      << (fIsActualPhoton ? 1 : 0) << '\t'
                      << std::setprecision(6) << fPhotonScore << '\t'
                      << (identifiedAsPhoton ? 1 : 0) << '\t'
                      << std::setprecision(3) << fFeatures.risetime_10_90 << '\t'
                      << fFeatures.pulse_width << '\t'
                      << std::setprecision(4) << fFeatures.peak_charge_ratio << '\t'
                      << fFeatures.secondary_peak_ratio << '\n';
        }

        // Debug print (for first ~100 relevant stations)
        if ((fIsActualPhoton || physicsPhotonLike) && fStationCount <= 100) {
            std::cout << "  Station " << fStationId
                      << " PMT " << pmtId
                      << ": Score=" << std::fixed << std::setprecision(3) << fPhotonScore
                      << " (True: " << fPrimaryType << ")"
                      << " Rise=" << fFeatures.risetime_10_90 << "ns"
                      << " Width=" << fFeatures.pulse_width << "ns"
                      << " PhysicsTag=" << (physicsPhotonLike ? "YES" : "NO") << "\n";
        }
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace_data)
{
    trace_data.clear();

    // Try direct FADC trace
    if (pmt.HasFADCTrace()) {
        try {
            const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            for (int i = 0; i < 2048; ++i) trace_data.push_back(trace[i]);
            return true;
        } catch (...) {
        }
    }

    // Try simulation data
    if (pmt.HasSimData()) {
        try {
            const sevt::PMTSimData& simData = pmt.GetSimData();
            if (simData.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& fadcTrace = simData.GetFADCTrace(
                    sdet::PMTConstants::eHighGain,
                    sevt::StationConstants::eTotal
                );
                for (int i = 0; i < 2048; ++i) trace_data.push_back(fadcTrace[i]);
                return true;
            }
        } catch (...) {
        }
    }

    return false;
}

PhotonTriggerML::EnhancedFeatures PhotonTriggerML::ExtractEnhancedFeatures(
    const std::vector<double>& trace, double)
{
    EnhancedFeatures features{};
    const int trace_size = static_cast<int>(trace.size());
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    int    peak_bin    = 0;
    double peak_value  = 0.0;
    double total_signal= 0.0;
    std::vector<double> signal(trace_size, 0.0);

    // Baseline from first 100 bins
    double baseline = 0.0;
    const int nb = std::min(100, trace_size);
    for (int i = 0; i < nb; ++i) baseline += trace[i];
    baseline /= std::max(1, nb);

    for (int i = 0; i < trace_size; ++i) {
        double s = trace[i] - baseline;
        if (s < 0.0) s = 0.0;
        signal[i] = s;
        if (s > peak_value) { peak_value = s; peak_bin = i; }
        total_signal += s;
    }

    if (peak_value < 5.0 || total_signal < 10.0) {
        return features;
    }

    features.peak_amplitude    = peak_value / ADC_PER_VEM;
    features.total_charge      = total_signal / ADC_PER_VEM;
    features.peak_charge_ratio = features.peak_amplitude / (features.total_charge + 0.001);

    // Rise / fall
    const double p10 = 0.10 * peak_value;
    const double p50 = 0.50 * peak_value;
    const double p90 = 0.90 * peak_value;

    int b10r = 0, b50r = 0, b90r = peak_bin;
    for (int i = peak_bin; i >= 0; --i) {
        if (signal[i] <= p90 && b90r == peak_bin) b90r = i;
        if (signal[i] <= p50 && b50r == 0)       b50r = i;
        if (signal[i] <= p10) { b10r = i; break; }
    }

    int b90f = peak_bin, b10f = trace_size - 1;
    for (int i = peak_bin; i < trace_size; ++i) {
        if (signal[i] <= p90 && b90f == peak_bin) b90f = i;
        if (signal[i] <= p10) { b10f = i; break; }
    }

    features.risetime_10_50 = std::abs(b50r - b10r) * NS_PER_BIN;
    features.risetime_10_90 = std::abs(b90r - b10r) * NS_PER_BIN;
    features.falltime_90_10 = std::abs(b10f - b90f) * NS_PER_BIN;

    // FWHM
    const double half = 0.5 * peak_value;
    int bHalfRise = b10r;
    int bHalfFall = b10f;
    for (int i = b10r; i <= peak_bin; ++i) {
        if (signal[i] >= half) { bHalfRise = i; break; }
    }
    for (int i = peak_bin; i < trace_size; ++i) {
        if (signal[i] <= half) { bHalfFall = i; break; }
    }
    features.pulse_width = std::abs(bHalfFall - bHalfRise) * NS_PER_BIN;

    // Asymmetry
    const double rise = features.risetime_10_90;
    const double fall = features.falltime_90_10;
    features.asymmetry = (fall - rise) / (fall + rise + 0.001);

    // Moments
    double mean_time = 0.0;
    for (int i = 0; i < trace_size; ++i) mean_time += i * signal[i];
    mean_time /= (total_signal + 0.001);

    double var = 0.0, skew = 0.0, kurt = 0.0;
    for (int i = 0; i < trace_size; ++i) {
        const double d  = i - mean_time;
        const double w  = signal[i] / (total_signal + 0.001);
        var  += d*d * w;
        skew += d*d*d * w;
        kurt += d*d*d*d * w;
    }

    const double sd = std::sqrt(var + 1e-3);
    features.time_spread = sd * NS_PER_BIN;
    features.skewness    = skew / (sd*sd*sd + 1e-6);
    features.kurtosis    = kurt / (var*var + 1e-6) - 3.0;

    // Early / late fractions
    const int quarter = trace_size / 4;
    double early = 0.0, late = 0.0;
    for (int i = 0; i < quarter;        ++i) early += signal[i];
    for (int i = 3*quarter; i < trace_size; ++i) late  += signal[i];
    features.early_fraction = early / (total_signal + 0.001);
    features.late_fraction  = late  / (total_signal + 0.001);

    // Smoothness
    double sumsq = 0.0; int cnt = 0;
    for (int i = 1; i < trace_size-1; ++i) {
        if (signal[i] > 0.1 * peak_value) {
            const double sec = signal[i+1] - 2*signal[i] + signal[i-1];
            sumsq += sec*sec; cnt++;
        }
    }
    features.smoothness = std::sqrt(sumsq / (cnt + 1));

    // High-frequency content
    double hf = 0.0;
    for (int i = 1; i < trace_size-1; ++i) {
        double d = signal[i+1] - signal[i-1];
        hf += d*d;
    }
    features.high_freq_content = hf / (total_signal*total_signal + 0.001);

    // Peaks (slightly stricter threshold to suppress fake multi-peak)
    features.num_peaks = 0;
    double sec_peak = 0.0;
    const double thr = 0.25 * peak_value; // was 0.15
    for (int i = 1; i < trace_size-1; ++i) {
        if (signal[i] > thr && signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
            features.num_peaks++;
            if (i != peak_bin && signal[i] > sec_peak) sec_peak = signal[i];
        }
    }
    features.secondary_peak_ratio = sec_peak / (peak_value + 0.001);

    return features;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& f)
{
    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, static_cast<double>(f.num_peaks),
        f.secondary_peak_ratio
    };

    static int n = 0;
    n++;

    for (size_t i = 0; i < raw.size(); ++i) {
        const double delta  = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i]   += delta / n;
        const double delta2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += delta * delta2;
    }

    if (n > 1) {
        for (size_t i = 0; i < raw.size(); ++i) {
            fFeatureStdDevs[i] = std::sqrt(fFeatureStdDevs[i] / (n - 1));
            if (fFeatureStdDevs[i] < 0.001) fFeatureStdDevs[i] = 1.0;
        }
    }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& f)
{
    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, static_cast<double>(f.num_peaks),
        f.secondary_peak_ratio
    };

    const std::vector<double> mins = {0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
    const std::vector<double> maxs = {500,1000,1000,1000,1,10,100,1,100,20,5,1,1,1000,10,10,1};

    std::vector<double> z; z.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        double v = (raw[i] - mins[i]) / (maxs[i] - mins[i] + 0.001);
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

    // Use up to 100 samples (bounded runtime)
    std::vector<std::vector<double>> Bx;
    std::vector<int> By;
    const int maxN = std::min(100, static_cast<int>(fTrainingFeatures.size()));
    for (int i = 0; i < maxN; ++i) {
        Bx.push_back(fTrainingFeatures[i]);
        By.push_back(fTrainingLabels[i]);
    }

    const int P = std::count(By.begin(), By.end(), 1);
    const int H = static_cast<int>(By.size()) - P;
    std::cout << " (P:" << P << " H:" << H << ")";

    const double lr = 0.01;
    double total = 0.0;
    for (int e = 0; e < 10; ++e) total += fNeuralNetwork->Train(Bx, By, lr);
    const double train_loss = total / 10.0;
    std::cout << " Loss: " << std::fixed << std::setprecision(4) << train_loss;

    // Validation loss & threshold auto‑tuning (sweep 0.40–0.80 for F1)
    double val_loss = 0.0;
    if (!fValidationFeatures.empty()) {
        const int N = static_cast<int>(fValidationFeatures.size());
        std::vector<double> preds(N, 0.0);
        for (int i = 0; i < N; ++i) preds[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

        int correct = 0;
        for (int i = 0; i < N; ++i) {
            const int y = fValidationLabels[i];
            const double p = preds[i];
            val_loss -= y * std::log(p + 1e-7) + (1 - y) * std::log(1 - p + 1e-7);
            const int yhat = (p > fPhotonThreshold) ? 1 : 0;
            if (yhat == y) correct++;
        }
        val_loss /= N;
        const double acc = 100.0 * correct / N;
        std::cout << " Val: " << val_loss << " (Acc: " << acc << "%)";

        double bestF1 = -1.0, bestThr = fPhotonThreshold;
        for (double thr = 0.40; thr <= 0.80 + 1e-12; thr += 0.02) {
            int TP=0, FP=0, TN=0, FN=0;
            for (int i = 0; i < N; ++i) {
                const int y = fValidationLabels[i];
                const int yhat = (preds[i] > thr) ? 1 : 0;
                if (yhat==1 && y==1) TP++;
                else if (yhat==1 && y==0) FP++;
                else if (yhat==0 && y==0) TN++;
                else FN++;
            }
            const double prec = (TP+FP>0)? (double)TP/(TP+FP) : 0.0;
            const double rec  = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            const double F1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;
            if (F1 > bestF1) { bestF1 = F1; bestThr = thr; }
        }
        const double old = fPhotonThreshold;
        fPhotonThreshold = 0.8 * fPhotonThreshold + 0.2 * bestThr;
        if (fLogFile.is_open()) {
            fLogFile << "Recalibrated threshold from " << old
                     << " -> " << fPhotonThreshold
                     << " (best thr=" << bestThr << ", bestF1=" << bestF1 << ")\n";
        }
        const int batch_num = fEventCount / 5;
        hValidationLoss->SetBinContent(batch_num, val_loss);
    }

    std::cout << std::endl;
    // Limit cache size
    if (fTrainingFeatures.size() > 1000) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin() + 200);
        fTrainingLabels.erase(fTrainingLabels.begin(),   fTrainingLabels.begin()   + 200);
    }

    fTrainingStep++;
    return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) return;

    const double acc  = 100.0 * (fTruePositives + fTrueNegatives) / total;
    const double prec = (fTruePositives + fFalsePositives > 0) ?
                        100.0 * fTruePositives / (fTruePositives + fFalsePositives) : 0.0;
    const double rec  = (fTruePositives + fFalseNegatives > 0) ?
                        100.0 * fTruePositives / (fTruePositives + fFalseNegatives) : 0.0;
    const double f1   = (prec + rec > 0.0) ? 2.0 * prec * rec / (prec + rec) : 0.0;

    const double photon_frac = (fPhotonLikeCount + fHadronLikeCount > 0) ?
                               100.0 * fPhotonLikeCount / (fPhotonLikeCount + fHadronLikeCount) : 0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << photon_frac << "%"
              << " │ " << std::setw(7) << acc << "%"
              << " │ " << std::setw(8) << prec << "%"
              << " │ " << std::setw(7) << rec << "%"
              << " │ " << std::setw(7) << f1 << "%│\n";

    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount
                 << " - Acc: " << acc
                 << "% Prec: " << prec
                 << "% Rec: " << rec
                 << "% F1: " << f1 << "%\n";
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    std::cout << "\n==========================================\n";
    std::cout << "PERFORMANCE METRICS\n";
    std::cout << "==========================================\n";

    const int TP = fTruePositives, FP = fFalsePositives, TN = fTrueNegatives, FN = fFalseNegatives;
    const int total = TP + FP + TN + FN;
    if (total == 0) {
        std::cout << "No predictions made yet!\n";
        return;
    }

    const double acc  = 100.0 * (TP + TN) / total;
    const double prec = (TP + FP > 0) ? 100.0 * TP / (TP + FP) : 0.0;
    const double rec  = (TP + FN > 0) ? 100.0 * TP / (TP + FN) : 0.0;
    const double f1   = (prec + rec > 0.0) ? 2.0 * prec * rec / (prec + rec) : 0.0;

    std::cout << "Accuracy:  " << std::fixed << std::setprecision(1) << acc << "%\n";
    std::cout << "Precision: " << prec << "%\n";
    std::cout << "Recall:    " << rec  << "%\n";
    std::cout << "F1-Score:  " << f1   << "%\n\n";

    std::cout << "CONFUSION MATRIX:\n";
    std::cout << "                Predicted\n";
    std::cout << "             Hadron   Photon\n";
    std::cout << "Actual Hadron  " << std::setw(6) << TN << "   " << std::setw(6) << FP << "\n";
    std::cout << "       Photon  " << std::setw(6) << FN << "   " << std::setw(6) << TP << "\n\n";

    std::cout << "Total Stations: " << fStationCount << "\n";
    std::cout << "Photon-like: " << fPhotonLikeCount << " ("
              << 100.0 * fPhotonLikeCount / std::max(1, fPhotonLikeCount + fHadronLikeCount) << "%)\n";
    std::cout << "Hadron-like: " << fHadronLikeCount << " ("
              << 100.0 * fHadronLikeCount / std::max(1, fPhotonLikeCount + fHadronLikeCount) << "%)\n";

    std::cout << "\nPARTICLE TYPE BREAKDOWN:\n";
    for (const auto& kv : fParticleTypeCounts) {
        std::cout << "  " << kv.first << ": " << kv.second << " events\n";
    }
    std::cout << "==========================================\n";

    if (fLogFile.is_open()) {
        fLogFile << "\nFinal Performance Metrics:\n";
        fLogFile << "Accuracy: " << acc << "%\n";
        fLogFile << "Precision: " << prec << "%\n";
        fLogFile << "Recall: " << rec << "%\n";
        fLogFile << "F1-Score: " << f1 << "%\n";
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    std::cout << "\n==========================================\n";
    std::cout << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n";
    std::cout << "==========================================\n";
    std::cout << "Events processed: "  << fEventCount  << "\n";
    std::cout << "Stations analyzed: " << fStationCount << "\n";

    CalculatePerformanceMetrics();

    // Save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // Save primary ROOT file
    if (fOutputFile) {
        fOutputFile->cd();
        if (fMLTree) {
            fMLTree->Write();
            std::cout << "Wrote " << fMLTree->GetEntries() << " entries to MLTree\n";
        }
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

    // Small summary ROOT file
    {
        TFile fsum("photon_trigger_summary.root", "RECREATE");
        if (!fsum.IsZombie()) {
            if (hConfusionMatrix)     hConfusionMatrix->Write("ConfusionMatrix");
            if (hPhotonScore)         hPhotonScore->Write("PhotonScore_All");
            if (hPhotonScorePhotons)  hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
            if (hPhotonScoreHadrons)  hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
            if (hConfidence)          hConfidence->Write("Confidence");
            fsum.Close();
        }
    }

    // Old-style human-readable summary (reverted format)
    {
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            const int TP = fTruePositives, FP = fFalsePositives, TN = fTrueNegatives, FN = fFalseNegatives;
            const int total = TP + FP + TN + FN;
            const double acc  = (total>0) ? 100.0*(TP+TN)/total : 0.0;
            const double prec = (TP+FP>0)? 100.0*TP/(TP+FP):0.0;
            const double rec  = (TP+FN>0)? 100.0*TP/(TP+FN):0.0;
            const double f1   = (prec+rec>0)? (2.0*prec*rec/(prec+rec)) : 0.0;
            txt << "PhotonTriggerML Summary\n";
            txt << "BaseThreshold=" << gBaseThreshold << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy=" << acc << "%  Precision=" << prec << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
            txt.close();
        }
    }

    // Extra diagnostics (text only; no header changes required)
    // 1) Threshold scan over validation cache
    {
        std::ofstream scan("photon_trigger_threshold_scan.txt");
        if (scan.is_open()) {
            scan << "#thr\tTP\tFP\tTN\tFN\tAcc\tPrec\tRec\tF1\tBA\tMCC\n";
            if (!fValidationFeatures.empty()) {
                const int N = static_cast<int>(fValidationFeatures.size());
                std::vector<double> preds(N, 0.0);
                for (int i = 0; i < N; ++i) preds[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

                for (double thr = 0.30; thr <= 0.90 + 1e-12; thr += 0.01) {
                    int TP=0,FP=0,TN=0,FN=0;
                    for (int i = 0; i < N; ++i) {
                        const int y = fValidationLabels[i];
                        const int yhat = (preds[i] > thr) ? 1 : 0;
                        if (yhat==1 && y==1) TP++;
                        else if (yhat==1 && y==0) FP++;
                        else if (yhat==0 && y==0) TN++;
                        else FN++;
                    }
                    const double tot = std::max(1, TP+FP+TN+FN);
                    const double acc = 100.0 * (TP+TN) / tot;
                    const double prec= (TP+FP>0)? 100.0*TP/(TP+FP):0.0;
                    const double rec = (TP+FN>0)? 100.0*TP/(TP+FN):0.0;
                    const double F1  = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;
                    const double TPR = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
                    const double TNR = (TN+FP>0)? (double)TN/(TN+FP) : 0.0;
                    const double BA  = 100.0 * 0.5 * (TPR + TNR);
                    // MCC
                    const double num = (double)TP*TN - (double)FP*FN;
                    const double den = std::sqrt((double)(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
                    const double MCC = (den > 0.0) ? (num/den) : 0.0;
                    scan << std::fixed << std::setprecision(2)
                         << thr << '\t' << TP << '\t' << FP << '\t' << TN << '\t' << FN << '\t'
                         << acc << '\t' << prec << '\t' << rec << '\t' << F1 << '\t' << BA << '\t' << MCC << "\n";
                }
            } else {
                scan << "# validation cache empty\n";
            }
            scan.close();
        }
    }

    // 2) Feature stats (running means/stddevs)
    {
        std::ofstream fs("photon_trigger_feature_stats.txt");
        if (fs.is_open()) {
            static const char* names[17] = {
                "risetime_10_50","risetime_10_90","falltime_90_10","pulse_width",
                "asymmetry","peak_amplitude","total_charge","peak_charge_ratio",
                "smoothness","kurtosis","skewness","early_fraction","late_fraction",
                "time_spread","high_freq_content","num_peaks","secondary_peak_ratio"
            };
            fs << "#feature\tmean\tstd\n";
            fs << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < 17; ++i) {
                fs << names[i] << '\t' << fFeatureMeans[i] << '\t' << fFeatureStdDevs[i] << '\n';
            }
            fs.close();
        }
    }

    // 3) In-place ML annotation of existing trace files (skip if called from signal handler)
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& title)->std::string {
            size_t p = title.find(" [ML:");
            return (p == std::string::npos) ? title : title.substr(0, p);
        };

        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool {
            if (!h) return false;
            const int n = h->GetNbinsX();
            if (n <= 0) return false;

            std::vector<double> trace(n, 0.0);
            for (int i = 1; i <= n; ++i) trace[i-1] = h->GetBinContent(i);

            EnhancedFeatures F = ExtractEnhancedFeatures(trace);
            std::vector<double> X = NormalizeFeatures(F);
            if (X.size() >= 17) {
                X[1]  *= 2.0;
                X[3]  *= 2.0;
                X[7]  *= 1.5;
                X[16] *= 1.5;
            }
            double ml = fNeuralNetwork->Predict(X, false);

            double penalty = 0.0;
            if (F.pulse_width < 120.0 && F.peak_charge_ratio > 0.35) penalty -= 0.20;
            if (F.num_peaks <= 2 && F.secondary_peak_ratio > 0.60)  penalty -= 0.10;

            score = std::max(0.0, std::min(1.0, ml + penalty));
            isPhoton = (score > fPhotonThreshold);
            return true;
        };

        std::function<void(TDirectory*)> annotate_dir;
        annotate_dir = [&](TDirectory* dir) {
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
                delete obj;
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
                std::cout << "Trace file appears recovered; skipping ML annotation.\n";
            }
            f->Close(); delete f;
        } else {
            std::cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    // Close TSV
    if (gScoreCSVOpen) {
        gScoreCSV.close();
        gScoreCSVOpen = false;
    }

    // Close log
    if (fLogFile.is_open()) fLogFile.close();

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
    if (it != fMLResultsMap.end()) {
        result = it->second;
        return true;
    }
    return false;
}

void PhotonTriggerML::ClearMLResults()
{
    fMLResultsMap.clear();
}

