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
#include <TDirectory.h>   // for in-place annotation
#include <TKey.h>         // for iterating objects in ROOT files

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

// Flag used to detect if SaveAndDisplaySummary() is being invoked from a signal
// handler. We avoid touching external ROOT files (like pmt_traces_1EeV.root)
// in that case to prevent "recovered" files or hangs.
static volatile sig_atomic_t gCalledFromSignal = 0;

// ============================================================================
// Signal handler (minimal + safe)
//   - Print message
//   - Mark that we're in a signal context
//   - Best-effort summary (no external trace-file annotation inside it)
//   - Restore default handler and re-raise to terminate promptly
// ============================================================================
void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << endl;

        gCalledFromSignal = 1; // Tell summary code to skip external file updates

        if (PhotonTriggerML::fInstance) {
            try {
                PhotonTriggerML::fInstance->SaveAndDisplaySummary();
            } catch (...) {
                // swallow to guarantee termination
            }
        }

        // Terminate promptly (no hangs): hand signal back to default and raise.
        std::signal(signal, SIG_DFL);
        std::raise(signal);

        // Fallback in case raise() doesn't exit in this environment.
        _Exit(0);
    }
}

// ============================================================================
// Physics-Based Neural Network Implementation
//   (same structure as before; lighter class-weight default)
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.1),  // Very light dropout
    fIsQuantized(false), fQuantizationScale(127.0)
{
}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize = input_size;
    fHidden1Size = hidden1_size;
    fHidden2Size = hidden2_size;

    cout << "Initializing Physics-Based Neural Network: " << input_size << " -> "
         << hidden1_size << " -> " << hidden2_size << " -> 1" << endl;

    // Use fixed seed initially for reproducibility
    std::mt19937 gen(12345);

    // Initialize with small random weights
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.resize(hidden1_size, std::vector<double>(input_size));
    for (int i = 0; i < hidden1_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            fWeights1[i][j] = dist(gen) / sqrt(input_size);
        }
    }

    fWeights2.resize(hidden2_size, std::vector<double>(hidden1_size));
    for (int i = 0; i < hidden2_size; ++i) {
        for (int j = 0; j < hidden1_size; ++j) {
            fWeights2[i][j] = dist(gen) / sqrt(hidden1_size);
        }
    }

    fWeights3.resize(1, std::vector<double>(hidden2_size));
    for (int j = 0; j < hidden2_size; ++j) {
        fWeights3[0][j] = dist(gen) / sqrt(hidden2_size);
    }

    // Zero biases initially
    fBias1.resize(hidden1_size, 0.0);
    fBias2.resize(hidden2_size, 0.0);
    fBias3 = 0.0;

    // Initialize momentum for SGD with momentum
    fMomentum1_w1.resize(hidden1_size, std::vector<double>(input_size, 0));
    fMomentum2_w1.resize(hidden1_size, std::vector<double>(input_size, 0));
    fMomentum1_w2.resize(hidden2_size, std::vector<double>(hidden1_size, 0));
    fMomentum2_w2.resize(hidden2_size, std::vector<double>(hidden1_size, 0));
    fMomentum1_w3.resize(1, std::vector<double>(hidden2_size, 0));
    fMomentum2_w3.resize(1, std::vector<double>(hidden2_size, 0));

    fMomentum1_b1.resize(hidden1_size, 0);
    fMomentum2_b1.resize(hidden1_size, 0);
    fMomentum1_b2.resize(hidden2_size, 0);
    fMomentum2_b2.resize(hidden2_size, 0);
    fMomentum1_b3 = 0;
    fMomentum2_b3 = 0;

    fTimeStep = 0;

    cout << "Neural Network initialized for physics-based discrimination!" << endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& features, bool training)
{
    if (static_cast<int>(features.size()) != fInputSize) {
        return 0.5;
    }

    // Simple feedforward with sigmoid activations throughout
    std::vector<double> hidden1(fHidden1Size);
    for (int i = 0; i < fHidden1Size; ++i) {
        double sum = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) {
            sum += fWeights1[i][j] * features[j];
        }
        hidden1[i] = 1.0 / (1.0 + exp(-sum));  // Sigmoid

        // Light dropout
        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) {
            hidden1[i] = 0;
        }
    }

    std::vector<double> hidden2(fHidden2Size);
    for (int i = 0; i < fHidden2Size; ++i) {
        double sum = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) {
            sum += fWeights2[i][j] * hidden1[j];
        }
        hidden2[i] = 1.0 / (1.0 + exp(-sum));  // Sigmoid

        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) {
            hidden2[i] = 0;
        }
    }

    double output = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) {
        output += fWeights3[0][j] * hidden2[j];
    }

    return 1.0 / (1.0 + exp(-output));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& features,
                                             const std::vector<int>& labels,
                                             double learning_rate)
{
    if (features.empty() || features.size() != labels.size()) {
        return -1.0;
    }

    double total_loss = 0.0;
    int batch_size = static_cast<int>(features.size());

    // Count classes
    int num_photons = std::count(labels.begin(), labels.end(), 1);
    // int num_hadrons = batch_size - num_photons; // not used

    // Softer class weighting (precision-first)
    double photon_weight = (num_photons > 0) ? 1.5 : 1.0;
    double hadron_weight = 1.0;

    // Initialize gradients
    std::vector<std::vector<double>> grad_w1(fHidden1Size, std::vector<double>(fInputSize, 0));
    std::vector<std::vector<double>> grad_w2(fHidden2Size, std::vector<double>(fHidden1Size, 0));
    std::vector<std::vector<double>> grad_w3(1, std::vector<double>(fHidden2Size, 0));
    std::vector<double> grad_b1(fHidden1Size, 0);
    std::vector<double> grad_b2(fHidden2Size, 0);
    double grad_b3 = 0;

    for (int sample = 0; sample < batch_size; ++sample) {
        const auto& input = features[sample];
        int label = labels[sample];
        double weight = (label == 1) ? photon_weight : hadron_weight;

        // Forward pass
        std::vector<double> hidden1(fHidden1Size);
        std::vector<double> hidden1_raw(fHidden1Size);

        for (int i = 0; i < fHidden1Size; ++i) {
            double sum = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) {
                sum += fWeights1[i][j] * input[j];
            }
            hidden1_raw[i] = sum;
            hidden1[i] = 1.0 / (1.0 + exp(-sum));
        }

        std::vector<double> hidden2(fHidden2Size);
        std::vector<double> hidden2_raw(fHidden2Size);

        for (int i = 0; i < fHidden2Size; ++i) {
            double sum = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) {
                sum += fWeights2[i][j] * hidden1[j];
            }
            hidden2_raw[i] = sum;
            hidden2[i] = 1.0 / (1.0 + exp(-sum));
        }

        double output_raw = fBias3;
        for (int j = 0; j < fHidden2Size; ++j) {
            output_raw += fWeights3[0][j] * hidden2[j];
        }

        double output = 1.0 / (1.0 + exp(-output_raw));

        // Cross-entropy loss
        double loss = -weight * (label * log(output + 1e-7) +
                                (1 - label) * log(1 - output + 1e-7));
        total_loss += loss;

        // Backpropagation
        double output_grad = weight * (output - label);

        // Output layer gradients
        for (int j = 0; j < fHidden2Size; ++j) {
            grad_w3[0][j] += output_grad * hidden2[j];
        }
        grad_b3 += output_grad;

        // Hidden layer 2 gradients
        std::vector<double> hidden2_grad(fHidden2Size);
        for (int j = 0; j < fHidden2Size; ++j) {
            hidden2_grad[j] = fWeights3[0][j] * output_grad;
            hidden2_grad[j] *= hidden2[j] * (1 - hidden2[j]);  // Sigmoid derivative
        }

        for (int i = 0; i < fHidden2Size; ++i) {
            for (int j = 0; j < fHidden1Size; ++j) {
                grad_w2[i][j] += hidden2_grad[i] * hidden1[j];
            }
            grad_b2[i] += hidden2_grad[i];
        }

        // Hidden layer 1 gradients
        std::vector<double> hidden1_grad(fHidden1Size);
        for (int j = 0; j < fHidden1Size; ++j) {
            hidden1_grad[j] = 0;
            for (int i = 0; i < fHidden2Size; ++i) {
                hidden1_grad[j] += fWeights2[i][j] * hidden2_grad[i];
            }
            hidden1_grad[j] *= hidden1[j] * (1 - hidden1[j]);  // Sigmoid derivative
        }

        for (int i = 0; i < fHidden1Size; ++i) {
            for (int j = 0; j < fInputSize; ++j) {
                grad_w1[i][j] += hidden1_grad[i] * input[j];
            }
            grad_b1[i] += hidden1_grad[i];
        }
    }

    // Simple SGD with momentum
    fTimeStep++;
    double momentum = 0.9;

    // Update weights
    for (int i = 0; i < fHidden1Size; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            double grad = grad_w1[i][j] / batch_size;
            fMomentum1_w1[i][j] = momentum * fMomentum1_w1[i][j] - learning_rate * grad;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }

        double grad = grad_b1[i] / batch_size;
        fMomentum1_b1[i] = momentum * fMomentum1_b1[i] - learning_rate * grad;
        fBias1[i] += fMomentum1_b1[i];
    }

    for (int i = 0; i < fHidden2Size; ++i) {
        for (int j = 0; j < fHidden1Size; ++j) {
            double grad = grad_w2[i][j] / batch_size;
            fMomentum1_w2[i][j] = momentum * fMomentum1_w2[i][j] - learning_rate * grad;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }

        double grad = grad_b2[i] / batch_size;
        fMomentum1_b2[i] = momentum * fMomentum1_b2[i] - learning_rate * grad;
        fBias2[i] += fMomentum1_b2[i];
    }

    for (int j = 0; j < fHidden2Size; ++j) {
        double grad = grad_w3[0][j] / batch_size;
        fMomentum1_w3[0][j] = momentum * fMomentum1_w3[0][j] - learning_rate * grad;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }

    double grad = grad_b3 / batch_size;
    fMomentum1_b3 = momentum * fMomentum1_b3 - learning_rate * grad;
    fBias3 += fMomentum1_b3;

    return total_loss / batch_size;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        cout << "Error: Could not save weights to " << filename << endl;
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
    file << "\n";
    file << fBias3 << "\n";

    file.close();
    cout << "Weights saved to " << filename << endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        cout << "Warning: Could not load weights from " << filename << endl;
        return false;
    }

    file >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.resize(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.resize(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.resize(1, std::vector<double>(fHidden2Size));
    fBias1.resize(fHidden1Size);
    fBias2.resize(fHidden2Size);

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
    cout << "Weights loaded from " << filename << endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights()
{
    fIsQuantized = true;
}

// ============================================================================
// PhotonTriggerML Implementation (precision-first improvements)
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
    fPhotonThreshold(0.65),  // Raised default; auto-calibrated during training
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_physics.root"),
    fWeightsFileName("photon_trigger_weights_physics.txt"),
    fLoadPretrainedWeights(true)   // default to using provided weights if present
{
    fInstance = this;

    fFeatureMeans.resize(17, 0.0);
    fFeatureStdDevs.resize(17, 1.0);

    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Constructor (PHYSICS-BASED, precision-first)" << endl;
    cout << "Output file: " << fOutputFileName << endl;
    cout << "Log file: " << fLogFileName << endl;
    cout << "==========================================" << endl;
}

PhotonTriggerML::~PhotonTriggerML()
{
    cout << "PhotonTriggerML Destructor called" << endl;
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization (PHYSICS-BASED VERSION)");

    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Initialization (PHYSICS)" << endl;
    cout << "==========================================" << endl;

    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }

    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML Physics-Based Version Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << endl << endl;

    // Initialize neural network - very simple architecture
    cout << "Initializing Simple Physics-Based Neural Network..." << endl;
    fNeuralNetwork->Initialize(17, 8, 4); // Very small network

    // Try to load pre-trained weights
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        cout << "Loaded pre-trained weights from " << fWeightsFileName << endl;
        fIsTraining = false;
    } else {
        cout << "Starting with random weights (training mode)" << endl;
    }

    // Create output file
    cout << "Creating output file: " << fOutputFileName << endl;
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }

    // Create tree
    cout << "Creating ROOT tree..." << endl;
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

    // Create histograms
    cout << "Creating histograms..." << endl;
    hPhotonScore = new TH1D("hPhotonScore", "ML Photon Score (All);Score;Count", 50, 0, 1);
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons", "ML Score (True Photons);Score;Count", 50, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons", "ML Score (True Hadrons);Score;Count", 50, 0, 1);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence = new TH1D("hConfidence", "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime = new TH1D("hRisetime", "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry = new TH1D("hAsymmetry", "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis = new TH1D("hKurtosis", "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score",
                              50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score",
                                50, 0, 3000, 50, 0, 1);

    // Create confusion matrix histogram
    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual",
                                2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");

    // Create training progress histograms
    hTrainingLoss = new TH1D("hTrainingLoss", "Training Loss;Batch;Loss", 10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss", 10000, 0, 10000);
    hAccuracyHistory = new TH1D("hAccuracyHistory", "Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    cout << "Initialization complete!" << endl;
    cout << "==========================================" << endl << endl;

    INFO("PhotonTriggerML initialized successfully (PHYSICS-BASED VERSION)");

    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;

    // Clear previous ML results
    ClearMLResults();

    // Print header every 50 events
    if (fEventCount % 50 == 1) {
        cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐" << endl;
        cout << "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│" << endl;
        cout << "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤" << endl;
    }

    // Get shower information
    fEnergy = 0;
    fCoreX = 0;
    fCoreY = 0;
    fPrimaryId = 0;
    fPrimaryType = "Unknown";
    fIsActualPhoton = false;

    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        fPrimaryId = shower.GetPrimaryParticle();

        switch(fPrimaryId) {
            case 22: fPrimaryType = "photon"; fIsActualPhoton = true; break;
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

        // Print first few events for debugging
        if (fEventCount <= 5) {
            cout << "\nEvent " << fEventCount
                 << ": Energy=" << fEnergy/1e18 << " EeV"
                 << ", Primary=" << fPrimaryType
                 << " (ID=" << fPrimaryId << ")" << endl;
        }
    }

    // Process stations
    int stationsInEvent = 0;
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();

        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
            stationsInEvent++;
        }
    }

    // Update performance metrics and display
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();

        // Update confusion matrix
        hConfusionMatrix->SetBinContent(1, 1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2, 1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1, 2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2, 2, fTruePositives);
    }

    // Train network with precision-first settings
    if (fIsTraining && fTrainingFeatures.size() >= 20 && fEventCount % 5 == 0) {
        double val_loss = TrainNetwork();

        int batch_num = fEventCount / 5;
        hTrainingLoss->SetBinContent(batch_num, val_loss);

        // Calculate current accuracy for history
        int correct = fTruePositives + fTrueNegatives;
        int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total > 0) {
            double accuracy = 100.0 * correct / total;
            hAccuracyHistory->SetBinContent(batch_num, accuracy);
        }

        // Save best model
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

    // Get station position
    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        double stationX = detStation.GetPosition().GetX(siteCS);
        double stationY = detStation.GetPosition().GetY(siteCS);
        fDistance = sqrt(pow(stationX - fCoreX, 2) + pow(stationY - fCoreY, 2));
    } catch (...) {
        fDistance = -1;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    // Process each PMT
    for (int p = 0; p < 3; p++) {
        const int pmtId = p + firstPMT;

        if (!station.HasPMT(pmtId)) continue;

        const sevt::PMT& pmt = station.GetPMT(pmtId);

        // Get trace data
        vector<double> trace_data;
        bool traceFound = ExtractTraceData(pmt, trace_data);

        if (!traceFound || trace_data.size() != 2048) continue;

        // Check for significant signal
        double maxVal = *max_element(trace_data.begin(), trace_data.end());
        double minVal = *min_element(trace_data.begin(), trace_data.end());
        if (maxVal - minVal < 10.0) continue;

        // Extract features (baseline parameter is unused internally)
        fFeatures = ExtractEnhancedFeatures(trace_data);

        // Basic physics tag (for debugging only; no score bias applied)
        bool physicsPhotonLike = false;
        if (fFeatures.risetime_10_90 < 150 &&  // Sharp rise
            fFeatures.pulse_width < 300 &&     // Narrow pulse
            fFeatures.secondary_peak_ratio < 0.3 && // Single peak
            fFeatures.peak_charge_ratio > 0.15) {   // Concentrated charge
            physicsPhotonLike = true;
        }

        // Fill feature histograms
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);

        // Update feature statistics
        UpdateFeatureStatistics(fFeatures);

        // Normalize features with physics emphasis
        std::vector<double> normalized = NormalizeFeatures(fFeatures);

        // Emphasize key physics features
        normalized[1]  *= 2.0;  // risetime_10_90
        normalized[3]  *= 2.0;  // pulse_width
        normalized[7]  *= 1.5;  // peak_charge_ratio
        normalized[16] *= 1.5;  // secondary_peak_ratio

        // Get ML prediction (no physics bias)
        double ml_score = fNeuralNetwork->Predict(normalized, false);

        // Muon-spike penalty: push down very narrow, spike-dominated traces
        double penalty = 0.0;
        if (fFeatures.pulse_width < 120.0 && fFeatures.peak_charge_ratio > 0.35) {
            penalty -= 0.20;
        }
        if (fFeatures.num_peaks <= 2 && fFeatures.secondary_peak_ratio > 0.60) {
            penalty -= 0.10;
        }

        fPhotonScore = ml_score + penalty;
        fPhotonScore = std::max(0.0, std::min(1.0, fPhotonScore));  // Clip to [0,1]

        fConfidence = std::fabs(fPhotonScore - 0.5);

        // Store for training (reduced oversampling; no oversampling of physicsPhotonLike hadrons)
        bool isValidation = (fStationCount % 10 == 0);
        if (fIsTraining) {
            if (!isValidation) {
                if (fIsActualPhoton) {
                    for (int copy = 0; copy < 2; ++copy) {  // 2x oversampling (from 20x)
                        std::vector<double> varied = normalized;
                        std::normal_distribution<> noise(0, 0.02);
                        static std::mt19937 gen{1234567u};
                        for (auto& val : varied) val += noise(gen);
                        fTrainingFeatures.push_back(varied);
                        fTrainingLabels.push_back(1);
                    }
                } else {
                    // Regular hadron - add once
                    fTrainingFeatures.push_back(normalized);
                    fTrainingLabels.push_back(0);
                }
            } else {
                // Validation set
                fValidationFeatures.push_back(normalized);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }

        fStationCount++;
        bool identifiedAsPhoton = (fPhotonScore > fPhotonThreshold);

        // Store ML result
        MLResult mlResult;
        mlResult.photonScore = fPhotonScore;
        mlResult.identifiedAsPhoton = identifiedAsPhoton;
        mlResult.isActualPhoton = fIsActualPhoton;
        mlResult.vemCharge = fFeatures.total_charge;
        mlResult.features = fFeatures;
        mlResult.primaryType = fPrimaryType;
        mlResult.confidence = fConfidence;
        fMLResultsMap[fStationId] = mlResult;

        // Update counters
        if (identifiedAsPhoton) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++;
            else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++;
            else fFalseNegatives++;
        }

        // Fill histograms
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);

        if (fIsActualPhoton) {
            hPhotonScorePhotons->Fill(fPhotonScore);
        } else {
            hPhotonScoreHadrons->Fill(fPhotonScore);
        }

        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);

        // Fill tree
        fMLTree->Fill();

        // Debug output
        if ((fIsActualPhoton || physicsPhotonLike) && fStationCount <= 100) {
            cout << "  Station " << fStationId
                 << " PMT " << pmtId
                 << ": Score=" << fixed << setprecision(3) << fPhotonScore
                 << " (True: " << fPrimaryType << ")"
                 << " Rise=" << fFeatures.risetime_10_90 << "ns"
                 << " Width=" << fFeatures.pulse_width << "ns"
                 << " PhysicsTag=" << (physicsPhotonLike ? "YES" : "NO") << endl;
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
            for (int i = 0; i < 2048; i++) {
                trace_data.push_back(trace[i]);
            }
            return true;
        } catch (...) {}
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
                for (int i = 0; i < 2048; i++) {
                    trace_data.push_back(fadcTrace[i]);
                }
                return true;
            }
        } catch (...) {}
    }

    return false;
}

PhotonTriggerML::EnhancedFeatures PhotonTriggerML::ExtractEnhancedFeatures(
    const std::vector<double>& trace, double)  // baseline parameter unused
{
    EnhancedFeatures features;
    const int trace_size = trace.size();
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN = 25.0;

    // Find peak and calculate basic quantities
    int peak_bin = 0;
    double peak_value = 0;
    double total_signal = 0;
    std::vector<double> signal(trace_size);

    // Baseline estimation from first 100 bins
    double estimated_baseline = 0;
    for (int i = 0; i < 100; i++) {
        estimated_baseline += trace[i];
    }
    estimated_baseline /= 100.0;

    for (int i = 0; i < trace_size; i++) {
        signal[i] = trace[i] - estimated_baseline;
        if (signal[i] < 0) signal[i] = 0;

        if (signal[i] > peak_value) {
            peak_value = signal[i];
            peak_bin = i;
        }
        total_signal += signal[i];
    }

    if (peak_value < 5.0 || total_signal < 10.0) {
        return features;
    }

    features.peak_amplitude = peak_value / ADC_PER_VEM;
    features.total_charge = total_signal / ADC_PER_VEM;
    features.peak_charge_ratio = features.peak_amplitude / (features.total_charge + 0.001);

    // Calculate rise and fall times with careful edge detection
    double peak_10 = 0.1 * peak_value;
    double peak_50 = 0.5 * peak_value;
    double peak_90 = 0.9 * peak_value;

    int bin_10_rise = 0, bin_50_rise = 0, bin_90_rise = peak_bin;

    // Find rise times going backward from peak
    for (int i = peak_bin; i >= 0; i--) {
        if (signal[i] <= peak_90 && bin_90_rise == peak_bin) bin_90_rise = i;
        if (signal[i] <= peak_50 && bin_50_rise == 0) bin_50_rise = i;
        if (signal[i] <= peak_10) {
            bin_10_rise = i;
            break;
        }
    }

    int bin_90_fall = peak_bin, bin_10_fall = trace_size - 1;

    // Find fall times going forward from peak
    for (int i = peak_bin; i < trace_size; i++) {
        if (signal[i] <= peak_90 && bin_90_fall == peak_bin) bin_90_fall = i;
        if (signal[i] <= peak_10) {
            bin_10_fall = i;
            break;
        }
    }

    features.risetime_10_50 = std::abs(bin_50_rise - bin_10_rise) * NS_PER_BIN;
    features.risetime_10_90 = std::abs(bin_90_rise - bin_10_rise) * NS_PER_BIN;
    features.falltime_90_10 = std::abs(bin_10_fall - bin_90_fall) * NS_PER_BIN;

    // Calculate pulse width (FWHM)
    double half_max = peak_value / 2.0;
    int bin_half_rise = bin_10_rise;
    int bin_half_fall = bin_10_fall;

    for (int i = bin_10_rise; i <= peak_bin; i++) {
        if (signal[i] >= half_max) {
            bin_half_rise = i;
            break;
        }
    }

    for (int i = peak_bin; i < trace_size; i++) {
        if (signal[i] <= half_max) {
            bin_half_fall = i;
            break;
        }
    }

    features.pulse_width = std::abs(bin_half_fall - bin_half_rise) * NS_PER_BIN;

    // Calculate asymmetry
    double rise = features.risetime_10_90;
    double fall = features.falltime_90_10;
    features.asymmetry = (fall - rise) / (fall + rise + 0.001);

    // Statistical moments
    double mean_time = 0;
    for (int i = 0; i < trace_size; i++) {
        mean_time += i * signal[i];
    }
    mean_time /= (total_signal + 0.001);

    double variance = 0;
    double skewness = 0;
    double kurtosis = 0;

    for (int i = 0; i < trace_size; i++) {
        double diff = i - mean_time;
        double weight = signal[i] / (total_signal + 0.001);
        variance += diff * diff * weight;
        skewness += diff * diff * diff * weight;
        kurtosis += diff * diff * diff * diff * weight;
    }

    double std_dev = sqrt(variance + 0.001);
    features.time_spread = std_dev * NS_PER_BIN;
    features.skewness = skewness / (std_dev * std_dev * std_dev + 0.001);
    features.kurtosis = kurtosis / (variance * variance + 0.001) - 3.0;

    // Early and late fractions
    int quarter = trace_size / 4;
    double early_charge = 0, late_charge = 0;

    for (int i = 0; i < quarter; i++) {
        early_charge += signal[i];
    }

    for (int i = 3 * quarter; i < trace_size; i++) {
        late_charge += signal[i];
    }

    features.early_fraction = early_charge / (total_signal + 0.001);
    features.late_fraction = late_charge / (total_signal + 0.001);

    // Smoothness
    double sum_sq_diff = 0;
    int smooth_count = 0;

    for (int i = 1; i < trace_size - 1; i++) {
        if (signal[i] > 0.1 * peak_value) {
            double second_deriv = signal[i+1] - 2*signal[i] + signal[i-1];
            sum_sq_diff += second_deriv * second_deriv;
            smooth_count++;
        }
    }

    features.smoothness = sqrt(sum_sq_diff / (smooth_count + 1));

    // High frequency content
    double high_freq = 0;
    for (int i = 1; i < trace_size - 1; i++) {
        double diff = signal[i+1] - signal[i-1];
        high_freq += diff * diff;
    }
    features.high_freq_content = high_freq / (total_signal * total_signal + 0.001);

    // Peak counting with stricter threshold (reduce fake multi-peak)
    features.num_peaks = 0;
    double secondary_peak = 0;
    double peak_threshold = 0.25 * peak_value;  // was 0.15

    for (int i = 1; i < trace_size - 1; i++) {
        if (signal[i] > peak_threshold &&
            signal[i] > signal[i-1] &&
            signal[i] > signal[i+1]) {
            features.num_peaks++;
            if (i != peak_bin && signal[i] > secondary_peak) {
                secondary_peak = signal[i];
            }
        }
    }

    features.secondary_peak_ratio = secondary_peak / (peak_value + 0.001);

    return features;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& features)
{
    std::vector<double> raw = {
        features.risetime_10_50,
        features.risetime_10_90,
        features.falltime_90_10,
        features.pulse_width,
        features.asymmetry,
        features.peak_amplitude,
        features.total_charge,
        features.peak_charge_ratio,
        features.smoothness,
        features.kurtosis,
        features.skewness,
        features.early_fraction,
        features.late_fraction,
        features.time_spread,
        features.high_freq_content,
        static_cast<double>(features.num_peaks),
        features.secondary_peak_ratio
    };

    static int n = 0;
    n++;

    // Online mean and variance calculation
    for (size_t i = 0; i < raw.size(); ++i) {
        double delta = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += delta / n;
        double delta2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += delta * delta2;
    }

    if (n > 1) {
        for (size_t i = 0; i < raw.size(); ++i) {
            fFeatureStdDevs[i] = sqrt(fFeatureStdDevs[i] / (n - 1));
            if (fFeatureStdDevs[i] < 0.001) fFeatureStdDevs[i] = 1.0;
        }
    }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& features)
{
    std::vector<double> normalized;

    std::vector<double> raw = {
        features.risetime_10_50,
        features.risetime_10_90,
        features.falltime_90_10,
        features.pulse_width,
        features.asymmetry,
        features.peak_amplitude,
        features.total_charge,
        features.peak_charge_ratio,
        features.smoothness,
        features.kurtosis,
        features.skewness,
        features.early_fraction,
        features.late_fraction,
        features.time_spread,
        features.high_freq_content,
        static_cast<double>(features.num_peaks),
        features.secondary_peak_ratio
    };

    // Simple min-max normalization to [0,1]
    std::vector<double> mins = {0, 0, 0, 0, -1, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0};
    std::vector<double> maxs = {500, 1000, 1000, 1000, 1, 10, 100, 1, 100, 20, 5, 1, 1, 1000, 10, 10, 1};

    for (size_t i = 0; i < raw.size(); ++i) {
        double val = (raw[i] - mins[i]) / (maxs[i] - mins[i] + 0.001);
        val = std::max(0.0, std::min(1.0, val));  // Clip to [0,1]
        normalized.push_back(val);
    }

    return normalized;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;

    cout << "\n  Training with " << fTrainingFeatures.size() << " samples...";

    // Create batch with all available data
    std::vector<std::vector<double>> batch_features;
    std::vector<int> batch_labels;

    // Use ALL training data each time (cap to 100 to keep runtime bounded)
    int max_samples = std::min(100, static_cast<int>(fTrainingFeatures.size()));

    for (int i = 0; i < max_samples; ++i) {
        batch_features.push_back(fTrainingFeatures[i]);
        batch_labels.push_back(fTrainingLabels[i]);
    }

    // Count classes
    int num_photons = std::count(batch_labels.begin(), batch_labels.end(), 1);
    int num_hadrons = static_cast<int>(batch_labels.size()) - num_photons;

    cout << " (P:" << num_photons << " H:" << num_hadrons << ")";

    // Learning rate
    double learning_rate = 0.01;

    // Multiple epochs per batch
    double total_loss = 0;
    for (int epoch = 0; epoch < 10; epoch++) {
        double loss = fNeuralNetwork->Train(batch_features, batch_labels, learning_rate);
        total_loss += loss;
    }
    double train_loss = total_loss / 10.0;

    cout << " Loss: " << fixed << setprecision(4) << train_loss;

    // ---------------------------
    // Validation metrics + auto-threshold calibration (F1 over [0.40, 0.80])
    // ---------------------------
    double val_loss = 0;
    if (!fValidationFeatures.empty()) {
        int N = static_cast<int>(fValidationFeatures.size());
        std::vector<double> preds(N);
        for (int i = 0; i < N; ++i) {
            preds[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);
        }

        // X-entropy loss + accuracy at current threshold
        int correct = 0;
        for (int i = 0; i < N; ++i) {
            int label = fValidationLabels[i];
            double p = preds[i];
            val_loss -= label * log(p + 1e-7) + (1 - label) * log(1 - p + 1e-7);
            int pred_label = (p > fPhotonThreshold) ? 1 : 0;
            if (pred_label == label) correct++;
        }
        val_loss /= N;
        double val_acc = 100.0 * correct / N;
        cout << " Val: " << val_loss << " (Acc: " << val_acc << "%)";

        // F1 sweep
        double bestF1 = -1.0;
        double bestThr = fPhotonThreshold;
        for (double thr = 0.40; thr <= 0.80 + 1e-9; thr += 0.02) {
            int TP=0, FP=0, TN=0, FN=0;
            for (int i = 0; i < N; ++i) {
                int label = fValidationLabels[i];
                int pred  = (preds[i] > thr) ? 1 : 0;
                if (pred==1 && label==1) TP++;
                else if (pred==1 && label==0) FP++;
                else if (pred==0 && label==0) TN++;
                else FN++;
            }
            double precision = (TP+FP>0)? (double)TP/(TP+FP) : 0.0;
            double recall    = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            double F1        = (precision+recall>0)? 2.0*precision*recall/(precision+recall) : 0.0;
            if (F1 > bestF1) { bestF1 = F1; bestThr = thr; }
        }

        // Smooth update (20% toward best)
        double oldThr = fPhotonThreshold;
        fPhotonThreshold = 0.8 * fPhotonThreshold + 0.2 * bestThr;

        cout << " | Recalibrated threshold to " << fixed << setprecision(3) << fPhotonThreshold
             << " (best=" << bestThr << ", F1=" << std::setprecision(3) << bestF1 << ")";
        if (fLogFile.is_open()) {
            fLogFile << "Recalibrated threshold from " << oldThr
                     << " -> " << fPhotonThreshold
                     << " using validation bestF1=" << bestF1
                     << " at thr=" << bestThr << "\n";
        }

        int batch_num = fEventCount / 5;
        hValidationLoss->SetBinContent(batch_num, val_loss);
    }

    cout << endl;

    // Keep limited data size
    if (fTrainingFeatures.size() > 1000) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(),
                               fTrainingFeatures.begin() + 200);
        fTrainingLabels.erase(fTrainingLabels.begin(),
                             fTrainingLabels.begin() + 200);
    }

    fTrainingStep++;

    return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) return;

    double accuracy = 100.0 * (fTruePositives + fTrueNegatives) / total;
    double precision = (fTruePositives + fFalsePositives > 0) ?
                      100.0 * fTruePositives / (fTruePositives + fFalsePositives) : 0;
    double recall = (fTruePositives + fFalseNegatives > 0) ?
                   100.0 * fTruePositives / (fTruePositives + fFalseNegatives) : 0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;

    double photon_frac = (fPhotonLikeCount + fHadronLikeCount > 0) ?
                        100.0 * fPhotonLikeCount / (fPhotonLikeCount + fHadronLikeCount) : 0;

    // Display in table format
    cout << "│ " << setw(6) << fEventCount
         << " │ " << setw(8) << fStationCount
         << " │ " << fixed << setprecision(1) << setw(7) << photon_frac << "%"
         << " │ " << setw(7) << accuracy << "%"
         << " │ " << setw(8) << precision << "%"
         << " │ " << setw(7) << recall << "%"
         << " │ " << setw(7) << f1 << "%│" << endl;

    // Log to file
    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount
                << " - Acc: " << accuracy
                << "% Prec: " << precision
                << "% Rec: " << recall
                << "% F1: " << f1 << "%" << endl;
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    cout << "\n==========================================" << endl;
    cout << "PERFORMANCE METRICS" << endl;
    cout << "==========================================" << endl;

    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) {
        cout << "No predictions made yet!" << endl;
        return;
    }

    double accuracy = 100.0 * (fTruePositives + fTrueNegatives) / total;
    double precision = (fTruePositives + fFalsePositives > 0) ?
                      100.0 * fTruePositives / (fTruePositives + fFalsePositives) : 0;
    double recall = (fTruePositives + fFalseNegatives > 0) ?
                   100.0 * fTruePositives / (fTruePositives + fFalseNegatives) : 0;
    double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;

    cout << "Accuracy:  " << fixed << setprecision(1) << accuracy << "%" << endl;
    cout << "Precision: " << precision << "%" << endl;
    cout << "Recall:    " << recall << "%" << endl;
    cout << "F1-Score:  " << f1 << "%" << endl;
    cout << endl;

    cout << "CONFUSION MATRIX:" << endl;
    cout << "                Predicted" << endl;
    cout << "             Hadron   Photon" << endl;
    cout << "Actual Hadron  " << setw(6) << fTrueNegatives << "   " << setw(6) << fFalsePositives << endl;
    cout << "       Photon  " << setw(6) << fFalseNegatives << "   " << setw(6) << fTruePositives << endl;
    cout << endl;

    cout << "Total Stations: " << fStationCount << endl;
    cout << "Photon-like: " << fPhotonLikeCount << " ("
         << 100.0 * fPhotonLikeCount / max(1, fPhotonLikeCount + fHadronLikeCount) << "%)" << endl;
    cout << "Hadron-like: " << fHadronLikeCount << " ("
         << 100.0 * fHadronLikeCount / max(1, fPhotonLikeCount + fHadronLikeCount) << "%)" << endl;

    cout << "\nPARTICLE TYPE BREAKDOWN:" << endl;
    for (const auto& pair : fParticleTypeCounts) {
        cout << "  " << pair.first << ": " << pair.second << " events" << endl;
    }

    cout << "==========================================" << endl;

    // Log to file
    if (fLogFile.is_open()) {
        fLogFile << "\nFinal Performance Metrics:" << endl;
        fLogFile << "Accuracy: " << accuracy << "%" << endl;
        fLogFile << "Precision: " << precision << "%" << endl;
        fLogFile << "Recall: " << recall << "%" << endl;
        fLogFile << "F1-Score: " << f1 << "%" << endl;
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n==========================================" << endl;
    cout << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)" << endl;
    cout << "==========================================" << endl;

    cout << "Events processed: " << fEventCount << endl;
    cout << "Stations analyzed: " << fStationCount << endl;

    CalculatePerformanceMetrics();

    // Save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // Save ROOT file
    if (fOutputFile) {
        fOutputFile->cd();

        // Write tree
        if (fMLTree) {
            fMLTree->Write();
            cout << "Wrote " << fMLTree->GetEntries() << " entries to tree" << endl;
        }

        // Write histograms
        hPhotonScore->Write();
        hPhotonScorePhotons->Write();
        hPhotonScoreHadrons->Write();
        hConfidence->Write();
        hRisetime->Write();
        hAsymmetry->Write();
        hKurtosis->Write();
        hScoreVsEnergy->Write();
        hScoreVsDistance->Write();
        hConfusionMatrix->Write();
        hTrainingLoss->Write();
        hValidationLoss->Write();
        hAccuracyHistory->Write();

        cout << "Histograms written to " << fOutputFileName << endl;

        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = nullptr;
    }

    // ----------------------------------------------------------------------
    // Minimal extra summaries (separate files)
    // ----------------------------------------------------------------------
    {
        // Small ROOT file containing confusion matrix and a few key hists
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
    {
        // Human-readable metrics
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            const int TP = fTruePositives, FP = fFalsePositives, TN = fTrueNegatives, FN = fFalseNegatives;
            const int total = TP + FP + TN + FN;
            const double acc  = (total>0) ? 100.0*(TP+TN)/total : 0.0;
            const double prec = (TP+FP>0)? 100.0*TP/(TP+FP):0.0;
            const double rec  = (TP+FN>0)? 100.0*TP/(TP+FN):0.0;
            const double f1   = (prec+rec>0)? (2.0*prec*rec/(prec+rec)) : 0.0;
            txt << "PhotonTriggerML Summary\n";
            txt << "Threshold: " << fPhotonThreshold << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy=" << acc << "%  Precision=" << prec << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
            txt.close();
        }
    }

    // ----------------------------------------------------------------------
    // Annotate the **existing** PMT trace histograms with ML tag
    //   - Done only on normal Finish() (NOT when called from signal handler)
    //   - In place, by re-scoring each trace from its bin contents
    // ----------------------------------------------------------------------
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& title)->std::string {
            size_t p = title.find(" [ML:");
            return (p == std::string::npos) ? title : title.substr(0, p);
        };

        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool {
            if (!h) return false;
            const int n = h->GetNbinsX();
            if (n <= 0) return false;

            std::vector<double> trace(n);
            for (int i = 1; i <= n; ++i) trace[i-1] = h->GetBinContent(i);

            // Reuse the same feature/physics pipeline as in ProcessStation()
            EnhancedFeatures F = ExtractEnhancedFeatures(trace);
            std::vector<double> X = NormalizeFeatures(F);

            // Emphasis identical to ProcessStation():
            if (X.size() >= 17) {
                X[1]  *= 2.0;  // risetime_10_90
                X[3]  *= 2.0;  // pulse_width
                X[7]  *= 1.5;  // peak_charge_ratio
                X[16] *= 1.5;  // secondary_peak_ratio
            }

            double ml = fNeuralNetwork->Predict(X, false);

            // Apply the same spike penalty used at runtime
            double penalty = 0.0;
            if (F.pulse_width < 120.0 && F.peak_charge_ratio > 0.35) penalty -= 0.20;
            if (F.num_peaks <= 2 && F.secondary_peak_ratio > 0.60)  penalty -= 0.10;

            score = std::max(0.0, std::min(1.0, ml + penalty));
            isPhoton = (score > fPhotonThreshold);
            return true;
        };

        auto annotate_dir = [&](TDirectory* dir, auto&& annotate_ref)->void {
            if (!dir) return;
            TIter next(dir->GetListOfKeys());
            TKey* key;
            while ((key = (TKey*)next())) {
                TObject* obj = dir->Get(key->GetName());
                if (!obj) continue;

                if (obj->InheritsFrom(TDirectory::Class())) {
                    annotate_ref((TDirectory*)obj, annotate_ref);
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
                    // If your traces are saved as canvases, annotate the first TH1 on the canvas too
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

        // Open and annotate the existing PMTTraceModule file in place
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
            // Do not touch if ROOT thinks it was recovered (e.g., other module crashed)
            if (!f->TestBit(TFile::kRecovered)) {
                annotate_dir(f, annotate_dir);
                f->Write("", TObject::kOverwrite);
                cout << "Annotated ML tags in existing trace file: " << f->GetName() << endl;
            } else {
                cout << "Trace file appears recovered; skipping ML annotation for safety.\n";
            }
            f->Close(); delete f;
        } else {
            cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    // Close log file
    if (fLogFile.is_open()) {
        fLogFile.close();
    }

    cout << "==========================================" << endl;
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion (PHYSICS-BASED VERSION)");

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

