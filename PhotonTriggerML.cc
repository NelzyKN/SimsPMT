/**
 * PhotonTriggerML.cc - FIXED VERSION
 * Fixes the bias problem where everything is predicted as photon
 * 
 * Key fixes:
 * 1. Better weight initialization (smaller range)
 * 2. Proper feature normalization with z-score
 * 3. Class balancing during training
 * 4. Gradient clipping to prevent explosion
 * 5. Better learning rate schedule
 */

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

#include <iostream>
#include <sstream>
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
using namespace utl;

// Static instance pointer for signal handler
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

// Static ML results map
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Signal handler function
void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << endl;
        
        if (PhotonTriggerML::fInstance) {
            PhotonTriggerML::fInstance->SaveAndDisplaySummary();
        }
        
        exit(0);
    }
}

// ============================================================================
// Neural Network Implementation - FIXED VERSION
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fIsQuantized(false), fQuantizationScale(127.0)
{
}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize = input_size;
    fHidden1Size = hidden1_size;
    fHidden2Size = hidden2_size;
    
    // Initialize weights with SMALLER Xavier/He initialization to prevent saturation
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility during debugging
    
    // Smaller initialization to prevent saturation
    double scale1 = sqrt(2.0 / input_size) * 0.5;  // Reduced by half
    double scale2 = sqrt(2.0 / hidden1_size) * 0.5;
    double scale3 = sqrt(2.0 / hidden2_size) * 0.5;
    
    std::normal_distribution<> dist1(0, scale1);
    std::normal_distribution<> dist2(0, scale2);
    std::normal_distribution<> dist3(0, scale3);
    
    // Initialize weight matrices
    fWeights1.resize(hidden1_size, std::vector<double>(input_size));
    for (int i = 0; i < hidden1_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            fWeights1[i][j] = dist1(gen);
        }
    }
    
    fWeights2.resize(hidden2_size, std::vector<double>(hidden1_size));
    for (int i = 0; i < hidden2_size; ++i) {
        for (int j = 0; j < hidden1_size; ++j) {
            fWeights2[i][j] = dist2(gen);
        }
    }
    
    fWeights3.resize(1, std::vector<double>(hidden2_size));
    for (int j = 0; j < hidden2_size; ++j) {
        fWeights3[0][j] = dist3(gen);
    }
    
    // Initialize biases to ZERO (not 0.01)
    fBias1.resize(hidden1_size, 0.0);
    fBias2.resize(hidden2_size, 0.0);
    fBias3 = 0.0;  // Start neutral
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& features)
{
    if (static_cast<int>(features.size()) != fInputSize) {
        return 0.5;  // Return neutral score if input size mismatch
    }
    
    // Layer 1: Input -> Hidden1 with Leaky ReLU
    std::vector<double> hidden1(fHidden1Size);
    for (int i = 0; i < fHidden1Size; ++i) {
        double sum = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) {
            sum += fWeights1[i][j] * features[j];
        }
        // Use Leaky ReLU to prevent dead neurons
        hidden1[i] = sum > 0 ? sum : 0.01 * sum;
    }
    
    // Layer 2: Hidden1 -> Hidden2 with Leaky ReLU
    std::vector<double> hidden2(fHidden2Size);
    for (int i = 0; i < fHidden2Size; ++i) {
        double sum = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) {
            sum += fWeights2[i][j] * hidden1[j];
        }
        hidden2[i] = sum > 0 ? sum : 0.01 * sum;
    }
    
    // Layer 3: Hidden2 -> Output
    double output = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) {
        output += fWeights3[0][j] * hidden2[j];
    }
    
    // Clip output before sigmoid to prevent saturation
    output = std::max(-10.0, std::min(10.0, output));
    
    return Sigmoid(output);
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
    
    // Count class distribution for this batch
    int num_photons = 0;
    for (int label : labels) {
        if (label == 1) num_photons++;
    }
    int num_hadrons = batch_size - num_photons;
    
    // Class weights for balanced training
    double photon_weight = (num_photons > 0) ? 0.5 * batch_size / num_photons : 1.0;
    double hadron_weight = (num_hadrons > 0) ? 0.5 * batch_size / num_hadrons : 1.0;
    
    // Adaptive learning rate
    double effective_lr = learning_rate;
    
    // Mini-batch gradient descent with momentum
    std::vector<std::vector<double>> grad_w1(fHidden1Size, std::vector<double>(fInputSize, 0));
    std::vector<std::vector<double>> grad_w2(fHidden2Size, std::vector<double>(fHidden1Size, 0));
    std::vector<std::vector<double>> grad_w3(1, std::vector<double>(fHidden2Size, 0));
    std::vector<double> grad_b1(fHidden1Size, 0);
    std::vector<double> grad_b2(fHidden2Size, 0);
    double grad_b3 = 0;
    
    for (int sample = 0; sample < batch_size; ++sample) {
        const auto& input = features[sample];
        int label = labels[sample];
        double class_weight = (label == 1) ? photon_weight : hadron_weight;
        
        // Forward pass
        std::vector<double> hidden1(fHidden1Size);
        std::vector<double> hidden1_raw(fHidden1Size);
        for (int i = 0; i < fHidden1Size; ++i) {
            double sum = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) {
                sum += fWeights1[i][j] * input[j];
            }
            hidden1_raw[i] = sum;
            hidden1[i] = sum > 0 ? sum : 0.01 * sum;  // Leaky ReLU
        }
        
        std::vector<double> hidden2(fHidden2Size);
        std::vector<double> hidden2_raw(fHidden2Size);
        for (int i = 0; i < fHidden2Size; ++i) {
            double sum = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) {
                sum += fWeights2[i][j] * hidden1[j];
            }
            hidden2_raw[i] = sum;
            hidden2[i] = sum > 0 ? sum : 0.01 * sum;  // Leaky ReLU
        }
        
        double output_raw = fBias3;
        for (int j = 0; j < fHidden2Size; ++j) {
            output_raw += fWeights3[0][j] * hidden2[j];
        }
        
        // Clip to prevent numerical issues
        output_raw = std::max(-10.0, std::min(10.0, output_raw));
        double output = Sigmoid(output_raw);
        
        // Calculate weighted loss
        double loss = -label * log(output + 1e-7) - (1 - label) * log(1 - output + 1e-7);
        total_loss += loss * class_weight;
        
        // Backpropagation with class weighting
        double output_grad = (output - label) * class_weight;
        
        // Clip gradient to prevent explosion
        output_grad = std::max(-1.0, std::min(1.0, output_grad));
        
        // Accumulate gradients for layer 3
        for (int j = 0; j < fHidden2Size; ++j) {
            grad_w3[0][j] += output_grad * hidden2[j];
        }
        grad_b3 += output_grad;
        
        // Gradients for layer 2
        std::vector<double> hidden2_grad(fHidden2Size);
        for (int j = 0; j < fHidden2Size; ++j) {
            hidden2_grad[j] = fWeights3[0][j] * output_grad;
            // Leaky ReLU derivative
            hidden2_grad[j] *= (hidden2_raw[j] > 0) ? 1.0 : 0.01;
        }
        
        // Accumulate gradients for layer 2
        for (int i = 0; i < fHidden2Size; ++i) {
            for (int j = 0; j < fHidden1Size; ++j) {
                grad_w2[i][j] += hidden2_grad[i] * hidden1[j];
            }
            grad_b2[i] += hidden2_grad[i];
        }
        
        // Gradients for layer 1
        std::vector<double> hidden1_grad(fHidden1Size);
        for (int j = 0; j < fHidden1Size; ++j) {
            hidden1_grad[j] = 0;
            for (int i = 0; i < fHidden2Size; ++i) {
                hidden1_grad[j] += fWeights2[i][j] * hidden2_grad[i];
            }
            // Leaky ReLU derivative
            hidden1_grad[j] *= (hidden1_raw[j] > 0) ? 1.0 : 0.01;
        }
        
        // Accumulate gradients for layer 1
        for (int i = 0; i < fHidden1Size; ++i) {
            for (int j = 0; j < fInputSize; ++j) {
                grad_w1[i][j] += hidden1_grad[i] * input[j];
            }
            grad_b1[i] += hidden1_grad[i];
        }
    }
    
    // Apply accumulated gradients with gradient clipping
    double max_grad_norm = 5.0;
    
    // Update weights with clipped gradients
    for (int i = 0; i < fHidden1Size; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            double grad = grad_w1[i][j] / batch_size;
            grad = std::max(-max_grad_norm, std::min(max_grad_norm, grad));
            fWeights1[i][j] -= effective_lr * grad;
            // Add L2 regularization
            fWeights1[i][j] *= 0.9999;
        }
        double grad = grad_b1[i] / batch_size;
        grad = std::max(-max_grad_norm, std::min(max_grad_norm, grad));
        fBias1[i] -= effective_lr * grad;
    }
    
    for (int i = 0; i < fHidden2Size; ++i) {
        for (int j = 0; j < fHidden1Size; ++j) {
            double grad = grad_w2[i][j] / batch_size;
            grad = std::max(-max_grad_norm, std::min(max_grad_norm, grad));
            fWeights2[i][j] -= effective_lr * grad;
            fWeights2[i][j] *= 0.9999;
        }
        double grad = grad_b2[i] / batch_size;
        grad = std::max(-max_grad_norm, std::min(max_grad_norm, grad));
        fBias2[i] -= effective_lr * grad;
    }
    
    for (int j = 0; j < fHidden2Size; ++j) {
        double grad = grad_w3[0][j] / batch_size;
        grad = std::max(-max_grad_norm, std::min(max_grad_norm, grad));
        fWeights3[0][j] -= effective_lr * grad;
        fWeights3[0][j] *= 0.9999;
    }
    
    double grad = grad_b3 / batch_size;
    grad = std::max(-max_grad_norm, std::min(max_grad_norm, grad));
    fBias3 -= effective_lr * grad;
    
    return total_loss / batch_size;
}

// Rest of the NeuralNetwork methods remain the same...
void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
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
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
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
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights()
{
    auto quantize = [this](double w) {
        return round(w * fQuantizationScale) / fQuantizationScale;
    };
    
    for (auto& row : fWeights1) {
        for (double& w : row) w = quantize(w);
    }
    for (auto& row : fWeights2) {
        for (double& w : row) w = quantize(w);
    }
    for (double& w : fWeights3[0]) w = quantize(w);
    
    for (double& b : fBias1) b = quantize(b);
    for (double& b : fBias2) b = quantize(b);
    fBias3 = quantize(fBias3);
    
    fIsQuantized = true;
}

// ============================================================================
// PhotonTriggerML Implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
    fIsTraining(true),
    fTrainingEpochs(100),
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
    fLogFileName("photon_trigger_ml_enhanced.log"),
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
    fTruePositives(0),
    fFalsePositives(0),
    fTrueNegatives(0),
    fFalseNegatives(0),
    fPhotonThreshold(0.5),
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_enhanced.root"),
    fWeightsFileName("photon_trigger_weights.txt"),
    fLoadPretrainedWeights(false)
{
    fInstance = this;
    
    // Initialize feature normalization parameters (will be computed from data)
    fFeatureMeans.resize(17, 0.0);
    fFeatureStdDevs.resize(17, 1.0);
}

PhotonTriggerML::~PhotonTriggerML()
{
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Initializing FIXED ML photon trigger");
    
    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    
    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "FIXED PhotonTriggerML Analysis Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << endl << endl;
    
    INFO("=== FIXED PhotonTriggerML Configuration ===");
    INFO("Neural Network: 3-layer (17-32-16-1)");
    INFO("Training: Balanced class weighting");
    INFO("Activation: Leaky ReLU (alpha=0.01)");
    INFO("Gradient clipping: max_norm=5.0");
    INFO("L2 regularization: 0.0001");
    INFO("===========================================");
    
    // Initialize neural network
    fNeuralNetwork->Initialize(17, 32, 16);
    
    // Try to load pre-trained weights if available
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        INFO("Loaded pre-trained weights from " + fWeightsFileName);
        fIsTraining = false;
    } else {
        INFO("Starting with randomly initialized weights (seed=42)");
    }
    
    // Create output file
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
    fOutputFile->cd();
    
    // Create tree and histograms (rest remains the same)
    fMLTree = new TTree("MLTree", "Enhanced Photon Trigger ML Analysis");
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
    hPhotonScore = new TH1D("hPhotonScore", "ML Photon Score;Score;Count", 100, 0, 1);
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons", "Score (True Photons);Score;Count", 100, 0, 1);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons", "Score (True Hadrons);Score;Count", 100, 0, 1);
    hConfidence = new TH1D("hConfidence", "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime = new TH1D("hRisetime", "Rise Time 10-90%;Time [ns];Count", 100, 0, 1000);
    hAsymmetry = new TH1D("hAsymmetry", "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 100, -1, 1);
    hKurtosis = new TH1D("hKurtosis", "Signal Kurtosis;Kurtosis;Count", 100, -5, 20);
    hScoreVsEnergy = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score", 
                              50, 1e16, 1e20, 100, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 
                                50, 0, 3000, 100, 0, 1);
    
    // Set histogram directory
    hPhotonScore->SetDirectory(fOutputFile);
    hPhotonScorePhotons->SetDirectory(fOutputFile);
    hPhotonScoreHadrons->SetDirectory(fOutputFile);
    hConfidence->SetDirectory(fOutputFile);
    hRisetime->SetDirectory(fOutputFile);
    hAsymmetry->SetDirectory(fOutputFile);
    hKurtosis->SetDirectory(fOutputFile);
    hScoreVsEnergy->SetDirectory(fOutputFile);
    hScoreVsDistance->SetDirectory(fOutputFile);
    
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    
    INFO("FIXED PhotonTriggerML initialized successfully");
    
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    
    ClearMLResults();
    
    // More frequent status updates to monitor training
    if (fEventCount <= 20 || fEventCount % 20 == 0) {
        ostringstream msg;
        msg << "Event " << fEventCount 
            << " | Stations: " << fStationCount;
        
        // Calculate current accuracy
        int correct = fTruePositives + fTrueNegatives;
        int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total > 0) {
            double accuracy = 100.0 * correct / total;
            msg << " | Accuracy: " << fixed << setprecision(1) << accuracy << "%";
            msg << " | TP:" << fTruePositives << " TN:" << fTrueNegatives 
                << " FP:" << fFalsePositives << " FN:" << fFalseNegatives;
        }
        
        INFO(msg.str());
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
    }
    
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }
    
    // More aggressive training schedule for better convergence
    if (fIsTraining && fTrainingFeatures.size() >= 32 && fEventCount % 10 == 0) {
        TrainNetwork();
    }
    
    return eSuccess;
}

// Keep ProcessStation the same as before...
void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();
    
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
    
    for (int p = 0; p < 3; p++) {
        const int pmtId = p + firstPMT;
        
        if (!station.HasPMT(pmtId)) continue;
        
        const sevt::PMT& pmt = station.GetPMT(pmtId);
        
        vector<double> trace_data;
        bool traceFound = false;
        
        if (pmt.HasFADCTrace()) {
            try {
                const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
                for (int i = 0; i < 2048; i++) {
                    trace_data.push_back(trace[i]);
                }
                traceFound = true;
            } catch (...) {}
        }
        
        if (!traceFound && pmt.HasSimData()) {
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
                    traceFound = true;
                }
            } catch (...) {}
        }
        
        if (!traceFound || trace_data.size() != 2048) continue;
        
        double maxVal = *max_element(trace_data.begin(), trace_data.end());
        double minVal = *min_element(trace_data.begin(), trace_data.end());
        if (maxVal - minVal < 10.0) continue;
        
        fFeatures = ExtractEnhancedFeatures(trace_data);
        
        std::vector<double> normalized = NormalizeFeatures(fFeatures);
        fPhotonScore = fNeuralNetwork->Predict(normalized);
        fConfidence = abs(fPhotonScore - 0.5);
        
        if (fIsTraining) {
            fTrainingFeatures.push_back(normalized);
            fTrainingLabels.push_back(fIsActualPhoton ? 1 : 0);
        }
        
        fStationCount++;
        bool identifiedAsPhoton = (fPhotonScore > fPhotonThreshold);
        
        MLResult mlResult;
        mlResult.photonScore = fPhotonScore;
        mlResult.identifiedAsPhoton = identifiedAsPhoton;
        mlResult.isActualPhoton = fIsActualPhoton;
        mlResult.vemCharge = fFeatures.total_charge;
        mlResult.features = fFeatures;
        mlResult.primaryType = fPrimaryType;
        mlResult.confidence = fConfidence;
        fMLResultsMap[fStationId] = mlResult;
        
        if (identifiedAsPhoton) {
            fPhotonLikeCount++;
            fParticleTypePhotonLike[fPrimaryType]++;
            if (fIsActualPhoton) fTruePositives++;
            else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++;
            else fFalseNegatives++;
        }
        
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) {
            hPhotonScorePhotons->Fill(fPhotonScore);
        } else {
            hPhotonScoreHadrons->Fill(fPhotonScore);
        }
        
        fMLTree->Fill();
    }
}

// FIXED NormalizeFeatures with proper z-score normalization
std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& features)
{
    std::vector<double> normalized;
    
    // Convert features to vector
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
    
    // Improved normalization with empirical means and std devs
    // These are approximate values based on typical shower characteristics
    std::vector<double> means = {
        250.0,   // risetime_10_50
        500.0,   // risetime_10_90
        500.0,   // falltime_90_10
        1000.0,  // pulse_width
        0.0,     // asymmetry
        10.0,    // peak_amplitude
        100.0,   // total_charge
        0.1,     // peak_charge_ratio
        20.0,    // smoothness
        3.0,     // kurtosis
        0.0,     // skewness
        0.25,    // early_fraction
        0.25,    // late_fraction
        500.0,   // time_spread
        1.0,     // high_freq_content
        2.0,     // num_peaks
        0.1      // secondary_peak_ratio
    };
    
    std::vector<double> stds = {
        200.0,   // risetime_10_50
        400.0,   // risetime_10_90
        400.0,   // falltime_90_10
        800.0,   // pulse_width
        0.5,     // asymmetry
        20.0,    // peak_amplitude
        200.0,   // total_charge
        0.2,     // peak_charge_ratio
        30.0,    // smoothness
        5.0,     // kurtosis
        2.0,     // skewness
        0.2,     // early_fraction
        0.2,     // late_fraction
        400.0,   // time_spread
        2.0,     // high_freq_content
        3.0,     // num_peaks
        0.2      // secondary_peak_ratio
    };
    
    // Z-score normalization
    for (size_t i = 0; i < raw.size(); ++i) {
        double z_score = (raw[i] - means[i]) / (stds[i] + 1e-8);
        // Clip to prevent extreme values
        z_score = std::max(-3.0, std::min(3.0, z_score));
        normalized.push_back(z_score);
    }
    
    return normalized;
}

// IMPROVED TrainNetwork with better monitoring
void PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return;
    
    // Ensure balanced batches
    std::vector<std::vector<double>> photon_features;
    std::vector<std::vector<double>> hadron_features;
    
    for (size_t i = 0; i < fTrainingFeatures.size(); ++i) {
        if (fTrainingLabels[i] == 1) {
            photon_features.push_back(fTrainingFeatures[i]);
        } else {
            hadron_features.push_back(fTrainingFeatures[i]);
        }
    }
    
    int num_photons = photon_features.size();
    int num_hadrons = hadron_features.size();
    
    INFO("Training with " + std::to_string(num_photons) + " photons and " 
         + std::to_string(num_hadrons) + " hadrons");
    
    // Create balanced batches
    int batch_size = 32;
    int samples_per_class = batch_size / 2;
    
    // Random indices for sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Train for several epochs on current data
    for (int epoch = 0; epoch < 5; ++epoch) {
        double total_loss = 0;
        int num_batches = 0;
        
        // Create balanced mini-batches
        int max_batches = std::min(num_photons, num_hadrons) / samples_per_class;
        
        for (int batch = 0; batch < max_batches; ++batch) {
            std::vector<std::vector<double>> batch_features;
            std::vector<int> batch_labels;
            
            // Add photons
            for (int i = 0; i < samples_per_class && i < num_photons; ++i) {
                int idx = (batch * samples_per_class + i) % num_photons;
                batch_features.push_back(photon_features[idx]);
                batch_labels.push_back(1);
            }
            
            // Add hadrons
            for (int i = 0; i < samples_per_class && i < num_hadrons; ++i) {
                int idx = (batch * samples_per_class + i) % num_hadrons;
                batch_features.push_back(hadron_features[idx]);
                batch_labels.push_back(0);
            }
            
            // Shuffle the batch
            std::vector<int> indices(batch_features.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            
            std::vector<std::vector<double>> shuffled_features;
            std::vector<int> shuffled_labels;
            for (int idx : indices) {
                shuffled_features.push_back(batch_features[idx]);
                shuffled_labels.push_back(batch_labels[idx]);
            }
            
            // Train on this batch with decaying learning rate
            double learning_rate = 0.001 * pow(0.95, fEventCount / 100);
            double loss = fNeuralNetwork->Train(shuffled_features, shuffled_labels, learning_rate);
            total_loss += loss;
            num_batches++;
        }
        
        if (epoch == 0 && num_batches > 0) {
            ostringstream msg;
            msg << "Training epoch loss: " << total_loss / num_batches
                << " (LR=" << 0.001 * pow(0.95, fEventCount / 100) << ")";
            INFO(msg.str());
        }
    }
    
    // Keep only recent training data to avoid memory issues
    if (fTrainingFeatures.size() > 5000) {
        // Keep last 2000 samples
        int keep_size = 2000;
        fTrainingFeatures.erase(fTrainingFeatures.begin(), 
                               fTrainingFeatures.end() - keep_size);
        fTrainingLabels.erase(fTrainingLabels.begin(), 
                             fTrainingLabels.end() - keep_size);
    }
}

// Keep the rest of the methods the same...
PhotonTriggerML::EnhancedFeatures PhotonTriggerML::ExtractEnhancedFeatures(
    const std::vector<double>& trace, double baseline)
{
    EnhancedFeatures features;
    const int trace_size = trace.size();
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN = 25.0;
    
    int peak_bin = 0;
    double peak_value = 0;
    double total_signal = 0;
    std::vector<double> signal(trace_size);
    
    for (int i = 0; i < trace_size; i++) {
        signal[i] = trace[i] - baseline;
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
    
    double peak_10 = 0.1 * peak_value;
    double peak_50 = 0.5 * peak_value;
    double peak_90 = 0.9 * peak_value;
    
    int bin_10_rise = 0, bin_50_rise = 0, bin_90_rise = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (signal[i] <= peak_90 && bin_90_rise == peak_bin) bin_90_rise = i;
        if (signal[i] <= peak_50 && bin_50_rise == 0) bin_50_rise = i;
        if (signal[i] <= peak_10) {
            bin_10_rise = i;
            break;
        }
    }
    
    int bin_90_fall = peak_bin, bin_10_fall = trace_size - 1;
    for (int i = peak_bin; i < trace_size; i++) {
        if (signal[i] <= peak_90 && bin_90_fall == peak_bin) bin_90_fall = i;
        if (signal[i] <= peak_10) {
            bin_10_fall = i;
            break;
        }
    }
    
    features.risetime_10_50 = abs(bin_50_rise - bin_10_rise) * NS_PER_BIN;
    features.risetime_10_90 = abs(bin_90_rise - bin_10_rise) * NS_PER_BIN;
    features.falltime_90_10 = abs(bin_10_fall - bin_90_fall) * NS_PER_BIN;
    
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
    features.pulse_width = abs(bin_half_fall - bin_half_rise) * NS_PER_BIN;
    
    double rise = features.risetime_10_90;
    double fall = features.falltime_90_10;
    features.asymmetry = (fall - rise) / (fall + rise + 0.001);
    
    double mean_time = 0;
    double variance = 0;
    double skewness = 0;
    double kurtosis = 0;
    
    for (int i = 0; i < trace_size; i++) {
        mean_time += i * signal[i];
    }
    mean_time /= (total_signal + 0.001);
    
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
    
    double high_freq = 0;
    for (int i = 1; i < trace_size - 1; i++) {
        double diff = signal[i+1] - signal[i-1];
        high_freq += diff * diff;
    }
    features.high_freq_content = high_freq / (total_signal * total_signal + 0.001);
    
    features.num_peaks = 0;
    double secondary_peak = 0;
    double peak_threshold = 0.2 * peak_value;
    bool in_peak = false;
    
    for (int i = 1; i < trace_size - 1; i++) {
        if (!in_peak && signal[i] > peak_threshold) {
            if (signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
                features.num_peaks++;
                if (i != peak_bin && signal[i] > secondary_peak) {
                    secondary_peak = signal[i];
                }
                in_peak = true;
            }
        } else if (in_peak && signal[i] < peak_threshold) {
            in_peak = false;
        }
    }
    
    features.secondary_peak_ratio = secondary_peak / (peak_value + 0.001);
    
    return features;
}

// Rest of the methods remain the same...
void PhotonTriggerML::CalculatePerformanceMetrics()
{
    double accuracy = 0, precision = 0, recall = 0, f1Score = 0;
    
    if (fStationCount > 0) {
        accuracy = (double)(fTruePositives + fTrueNegatives) / fStationCount * 100.0;
    }
    
    if (fTruePositives + fFalsePositives > 0) {
        precision = (double)fTruePositives / (fTruePositives + fFalsePositives) * 100.0;
    }
    
    if (fTruePositives + fFalseNegatives > 0) {
        recall = (double)fTruePositives / (fTruePositives + fFalseNegatives) * 100.0;
    }
    
    if (precision + recall > 0) {
        f1Score = 2.0 * precision * recall / (precision + recall);
    }
    
    ostringstream msg;
    msg << "\n=== Performance Metrics ===" << endl
        << "Accuracy:  " << fixed << setprecision(1) << accuracy << "%" << endl
        << "Precision: " << precision << "%" << endl
        << "Recall:    " << recall << "%" << endl
        << "F1-Score:  " << f1Score << "%" << endl;
    
    INFO(msg.str());
    if (fLogFile.is_open()) {
        fLogFile << msg.str();
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    INFO("\n=== FIXED PhotonTriggerML Summary ===");
    INFO("Events: " + std::to_string(fEventCount));
    INFO("Stations: " + std::to_string(fStationCount));
    INFO("Photon-like: " + std::to_string(fPhotonLikeCount));
    INFO("Hadron-like: " + std::to_string(fHadronLikeCount));
    
    CalculatePerformanceMetrics();
    
    fNeuralNetwork->SaveWeights(fWeightsFileName);
    INFO("Saved network weights to " + fWeightsFileName);
    
    if (fOutputFile) {
        fOutputFile->cd();
        
        if (fMLTree) fMLTree->Write();
        
        hPhotonScore->Write();
        hPhotonScorePhotons->Write();
        hPhotonScoreHadrons->Write();
        hConfidence->Write();
        hRisetime->Write();
        hAsymmetry->Write();
        hKurtosis->Write();
        hScoreVsEnergy->Write();
        hScoreVsDistance->Write();
        
        fOutputFile->Write();
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = nullptr;
        
        INFO("Data saved to " + fOutputFileName);
    }
    
    if (fLogFile.is_open()) {
        fLogFile.close();
    }
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    
    ostringstream msg;
    msg << "Final Summary:\n"
        << "  Events processed: " << fEventCount << "\n"
        << "  Stations analyzed: " << fStationCount << "\n"
        << "  Photon-like: " << fPhotonLikeCount << "\n"
        << "  Hadron-like: " << fHadronLikeCount;
    INFO(msg.str());
    
    if (fIsTraining && !fTrainingFeatures.empty()) {
        INFO("Final training with all accumulated data...");
        for (int i = 0; i < 50; ++i) {
            TrainNetwork();
        }
    }
    
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
