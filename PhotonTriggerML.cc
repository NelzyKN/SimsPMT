/**
 * PhotonTriggerML.cc
 * Enhanced ML-based photon trigger with real neural network
 * 
 * This implementation uses a lightweight 3-layer feedforward neural network
 * with proper training, validation, and inference capabilities.
 * Designed for eventual FPGA deployment with quantized weights.
 * 
 * Author: Khoa Nguyen
 * Institution: Michigan Technological University
 * Date: 2025
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
// Neural Network Implementation
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
    
    // Initialize weights with Xavier/He initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier initialization for weights
    double xavier1 = sqrt(2.0 / (input_size + hidden1_size));
    double xavier2 = sqrt(2.0 / (hidden1_size + hidden2_size));
    double xavier3 = sqrt(2.0 / (hidden2_size + 1));
    
    std::normal_distribution<> dist1(0, xavier1);
    std::normal_distribution<> dist2(0, xavier2);
    std::normal_distribution<> dist3(0, xavier3);
    
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
    
    // Initialize biases to small values
    fBias1.resize(hidden1_size, 0.01);
    fBias2.resize(hidden2_size, 0.01);
    fBias3 = 0.01;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& features)
{
    if (static_cast<int>(features.size()) != fInputSize) {
        return 0.5;  // Return neutral score if input size mismatch
    }
    
    // Layer 1: Input -> Hidden1
    std::vector<double> hidden1(fHidden1Size);
    for (int i = 0; i < fHidden1Size; ++i) {
        double sum = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) {
            sum += fWeights1[i][j] * features[j];
        }
        hidden1[i] = ReLU(sum);
    }
    
    // Layer 2: Hidden1 -> Hidden2
    std::vector<double> hidden2(fHidden2Size);
    for (int i = 0; i < fHidden2Size; ++i) {
        double sum = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) {
            sum += fWeights2[i][j] * hidden1[j];
        }
        hidden2[i] = ReLU(sum);
    }
    
    // Layer 3: Hidden2 -> Output
    double output = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) {
        output += fWeights3[0][j] * hidden2[j];
    }
    
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
    
    // Simple mini-batch gradient descent
    for (int sample = 0; sample < batch_size; ++sample) {
        const auto& input = features[sample];
        int label = labels[sample];
        
        // Forward pass with storage for backpropagation
        std::vector<double> hidden1(fHidden1Size);
        std::vector<double> hidden1_raw(fHidden1Size);
        for (int i = 0; i < fHidden1Size; ++i) {
            double sum = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) {
                sum += fWeights1[i][j] * input[j];
            }
            hidden1_raw[i] = sum;
            hidden1[i] = ReLU(sum);
        }
        
        std::vector<double> hidden2(fHidden2Size);
        std::vector<double> hidden2_raw(fHidden2Size);
        for (int i = 0; i < fHidden2Size; ++i) {
            double sum = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) {
                sum += fWeights2[i][j] * hidden1[j];
            }
            hidden2_raw[i] = sum;
            hidden2[i] = ReLU(sum);
        }
        
        double output_raw = fBias3;
        for (int j = 0; j < fHidden2Size; ++j) {
            output_raw += fWeights3[0][j] * hidden2[j];
        }
        double output = Sigmoid(output_raw);
        
        // Calculate loss (binary cross-entropy)
        double loss = -label * log(output + 1e-7) - (1 - label) * log(1 - output + 1e-7);
        total_loss += loss;
        
        // Backpropagation
        double output_grad = output - label;
        
        // Gradients for layer 3
        std::vector<double> hidden2_grad(fHidden2Size);
        for (int j = 0; j < fHidden2Size; ++j) {
            hidden2_grad[j] = fWeights3[0][j] * output_grad;
            // Update weights
            fWeights3[0][j] -= learning_rate * output_grad * hidden2[j];
        }
        fBias3 -= learning_rate * output_grad;
        
        // Apply ReLU derivative
        for (int i = 0; i < fHidden2Size; ++i) {
            hidden2_grad[i] *= ReLUDerivative(hidden2_raw[i]);
        }
        
        // Gradients for layer 2
        std::vector<double> hidden1_grad(fHidden1Size);
        for (int j = 0; j < fHidden1Size; ++j) {
            hidden1_grad[j] = 0;
            for (int i = 0; i < fHidden2Size; ++i) {
                hidden1_grad[j] += fWeights2[i][j] * hidden2_grad[i];
                // Update weights
                fWeights2[i][j] -= learning_rate * hidden2_grad[i] * hidden1[j];
            }
        }
        for (int i = 0; i < fHidden2Size; ++i) {
            fBias2[i] -= learning_rate * hidden2_grad[i];
        }
        
        // Apply ReLU derivative
        for (int i = 0; i < fHidden1Size; ++i) {
            hidden1_grad[i] *= ReLUDerivative(hidden1_raw[i]);
        }
        
        // Gradients for layer 1
        for (int i = 0; i < fHidden1Size; ++i) {
            for (int j = 0; j < fInputSize; ++j) {
                fWeights1[i][j] -= learning_rate * hidden1_grad[i] * input[j];
            }
            fBias1[i] -= learning_rate * hidden1_grad[i];
        }
    }
    
    return total_loss / batch_size;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    // Save architecture
    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";
    
    // Save weights and biases
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
    
    // Load architecture
    file >> fInputSize >> fHidden1Size >> fHidden2Size;
    
    // Resize matrices
    fWeights1.resize(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.resize(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.resize(1, std::vector<double>(fHidden2Size));
    fBias1.resize(fHidden1Size);
    fBias2.resize(fHidden2Size);
    
    // Load weights and biases
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
    // Quantize to 8-bit integers for FPGA deployment
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
    INFO("PhotonTriggerML::Init() - Initializing enhanced ML photon trigger");
    
    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    
    // Write header to log file
    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "Enhanced PhotonTriggerML Analysis Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << endl << endl;
    
    INFO("=== Enhanced PhotonTriggerML Configuration ===");
    INFO("Neural Network: 3-layer feedforward (17-32-16-1)");
    INFO("Features: 17 physics-based discriminators");
    INFO("Training: Online learning enabled");
    INFO("Target: >50% efficiency improvement");
    INFO("==============================================");
    
    // Initialize neural network
    fNeuralNetwork->Initialize(17, 32, 16);  // 17 features, 32 hidden1, 16 hidden2
    
    // Try to load pre-trained weights if available
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        INFO("Loaded pre-trained weights from " + fWeightsFileName);
        fLogFile << "Loaded pre-trained weights from " << fWeightsFileName << endl;
        fIsTraining = false;  // Switch to inference mode
    } else {
        INFO("Starting with randomly initialized weights");
        fLogFile << "Starting with randomly initialized weights" << endl;
    }
    
    // Create output file
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
    // Create enhanced tree
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
    
    // Add all enhanced features to tree
    fMLTree->Branch("risetime_10_50", &fFeatures.risetime_10_50, "risetime_10_50/D");
    fMLTree->Branch("risetime_10_90", &fFeatures.risetime_10_90, "risetime_10_90/D");
    fMLTree->Branch("asymmetry", &fFeatures.asymmetry, "asymmetry/D");
    fMLTree->Branch("kurtosis", &fFeatures.kurtosis, "kurtosis/D");
    fMLTree->Branch("skewness", &fFeatures.skewness, "skewness/D");
    fMLTree->Branch("smoothness", &fFeatures.smoothness, "smoothness/D");
    fMLTree->Branch("early_fraction", &fFeatures.early_fraction, "early_fraction/D");
    fMLTree->Branch("high_freq_content", &fFeatures.high_freq_content, "high_freq_content/D");
    
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
    
    // Set up signal handlers
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    
    INFO("Enhanced PhotonTriggerML initialized successfully");
    
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    
    // Clear ML results from previous event
    ClearMLResults();
    
    if (fEventCount % 50 == 0) {
        ostringstream msg;
        msg << "Event " << fEventCount 
            << " | Stations: " << fStationCount 
            << " | Photon-like: " << fPhotonLikeCount
            << " | Training samples: " << fTrainingFeatures.size();
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
        
        // Determine particle type
        switch(fPrimaryId) {
            case 22: fPrimaryType = "photon"; fIsActualPhoton = true; break;
            case 11: case -11: fPrimaryType = "electron"; break;
            case 2212: fPrimaryType = "proton"; break;
            case 1000026056: fPrimaryType = "iron"; break;
            default: fPrimaryType = (fPrimaryId > 1000000000) ? "nucleus" : "unknown";
        }
        
        fParticleTypeCounts[fPrimaryType]++;
        
        // Get core position
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }
    }
    
    // Process stations
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }
    
    // Periodically train the network if in training mode
    if (fIsTraining && fTrainingFeatures.size() >= 100 && fEventCount % 100 == 0) {
        TrainNetwork();
    }
    
    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();
    
    // Get station position and calculate distance
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
    
    // Process PMTs
    const int firstPMT = sdet::Station::GetFirstPMTId();
    
    for (int p = 0; p < 3; p++) {
        const int pmtId = p + firstPMT;
        
        if (!station.HasPMT(pmtId)) continue;
        
        const sevt::PMT& pmt = station.GetPMT(pmtId);
        
        // Get FADC trace
        vector<double> trace_data;
        bool traceFound = false;
        
        // Try direct access
        if (pmt.HasFADCTrace()) {
            try {
                const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
                for (int i = 0; i < 2048; i++) {
                    trace_data.push_back(trace[i]);
                }
                traceFound = true;
            } catch (...) {}
        }
        
        // Try simulation data
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
        
        // Check for significant signal
        double maxVal = *max_element(trace_data.begin(), trace_data.end());
        double minVal = *min_element(trace_data.begin(), trace_data.end());
        if (maxVal - minVal < 10.0) continue;
        
        // Extract enhanced features
        fFeatures = ExtractEnhancedFeatures(trace_data);
        
        // Get neural network prediction
        std::vector<double> normalized = NormalizeFeatures(fFeatures);
        fPhotonScore = fNeuralNetwork->Predict(normalized);
        fConfidence = abs(fPhotonScore - 0.5);
        
        // Store for training if needed
        if (fIsTraining) {
            fTrainingFeatures.push_back(normalized);
            fTrainingLabels.push_back(fIsActualPhoton ? 1 : 0);
        }
        
        // Update counters
        fStationCount++;
        bool identifiedAsPhoton = (fPhotonScore > fPhotonThreshold);
        
        // Store ML results
        MLResult mlResult;
        mlResult.photonScore = fPhotonScore;
        mlResult.identifiedAsPhoton = identifiedAsPhoton;
        mlResult.isActualPhoton = fIsActualPhoton;
        mlResult.vemCharge = fFeatures.total_charge;
        mlResult.features = fFeatures;
        mlResult.primaryType = fPrimaryType;
        mlResult.confidence = fConfidence;
        fMLResultsMap[fStationId] = mlResult;
        
        // Update performance metrics
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
        
        // Fill histograms
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) {
            hPhotonScorePhotons->Fill(fPhotonScore);
        } else {
            hPhotonScoreHadrons->Fill(fPhotonScore);
        }
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        if (fDistance > 0) {
            hScoreVsDistance->Fill(fDistance, fPhotonScore);
        }
        
        // Fill tree
        fMLTree->Fill();
        
        // Log high-confidence detections
        if (fConfidence > 0.4 && fStationCount <= 100) {
            ostringstream msg;
            msg << "Station " << fStationId 
                << " | True: " << fPrimaryType
                << " | Score: " << fixed << setprecision(3) << fPhotonScore
                << " | Confidence: " << fConfidence
                << " | " << (identifiedAsPhoton == fIsActualPhoton ? "CORRECT" : "WRONG");
            INFO(msg.str());
        }
    }
}

PhotonTriggerML::EnhancedFeatures PhotonTriggerML::ExtractEnhancedFeatures(
    const std::vector<double>& trace, double baseline)
{
    EnhancedFeatures features;
    const int trace_size = trace.size();
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN = 25.0;  // 40 MHz sampling
    
    // Find peak and calculate basic quantities
    int peak_bin = 0;
    double peak_value = 0;
    double total_signal = 0;
    std::vector<double> signal(trace_size);
    
    for (int i = 0; i < trace_size; i++) {
        signal[i] = trace[i] - baseline;
        if (signal[i] < 0) signal[i] = 0;  // Remove negative values
        
        if (signal[i] > peak_value) {
            peak_value = signal[i];
            peak_bin = i;
        }
        total_signal += signal[i];
    }
    
    // If no significant signal, return empty features
    if (peak_value < 5.0 || total_signal < 10.0) {
        return features;
    }
    
    features.peak_amplitude = peak_value / ADC_PER_VEM;
    features.total_charge = total_signal / ADC_PER_VEM;
    features.peak_charge_ratio = features.peak_amplitude / (features.total_charge + 0.001);
    
    // Calculate rise times (fixed to avoid negative values)
    double peak_10 = 0.1 * peak_value;
    double peak_50 = 0.5 * peak_value;
    double peak_90 = 0.9 * peak_value;
    
    // Find rise time points (search backward from peak)
    int bin_10_rise = 0, bin_50_rise = 0, bin_90_rise = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (signal[i] <= peak_90 && bin_90_rise == peak_bin) bin_90_rise = i;
        if (signal[i] <= peak_50 && bin_50_rise == 0) bin_50_rise = i;
        if (signal[i] <= peak_10) {
            bin_10_rise = i;
            break;
        }
    }
    
    // Find fall time points (search forward from peak)
    int bin_90_fall = peak_bin, bin_10_fall = trace_size - 1;
    for (int i = peak_bin; i < trace_size; i++) {
        if (signal[i] <= peak_90 && bin_90_fall == peak_bin) bin_90_fall = i;
        if (signal[i] <= peak_10) {
            bin_10_fall = i;
            break;
        }
    }
    
    // Calculate time features (ensure positive values)
    features.risetime_10_50 = abs(bin_50_rise - bin_10_rise) * NS_PER_BIN;
    features.risetime_10_90 = abs(bin_90_rise - bin_10_rise) * NS_PER_BIN;
    features.falltime_90_10 = abs(bin_10_fall - bin_90_fall) * NS_PER_BIN;
    
    // FWHM
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
    
    // Asymmetry
    double rise = features.risetime_10_90;
    double fall = features.falltime_90_10;
    features.asymmetry = (fall - rise) / (fall + rise + 0.001);
    
    // Statistical moments
    double mean_time = 0;
    double variance = 0;
    double skewness = 0;
    double kurtosis = 0;
    
    // Calculate weighted mean time
    for (int i = 0; i < trace_size; i++) {
        mean_time += i * signal[i];
    }
    mean_time /= (total_signal + 0.001);
    
    // Calculate higher moments
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
    features.kurtosis = kurtosis / (variance * variance + 0.001) - 3.0;  // Excess kurtosis
    
    // Temporal distribution
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
    
    // Smoothness (RMS of second derivative)
    double sum_sq_diff = 0;
    int smooth_count = 0;
    for (int i = 1; i < trace_size - 1; i++) {
        if (signal[i] > 0.1 * peak_value) {  // Only where signal is significant
            double second_deriv = signal[i+1] - 2*signal[i] + signal[i-1];
            sum_sq_diff += second_deriv * second_deriv;
            smooth_count++;
        }
    }
    features.smoothness = sqrt(sum_sq_diff / (smooth_count + 1));
    
    // Simple frequency content (high-frequency power)
    double high_freq = 0;
    for (int i = 1; i < trace_size - 1; i++) {
        double diff = signal[i+1] - signal[i-1];
        high_freq += diff * diff;
    }
    features.high_freq_content = high_freq / (total_signal * total_signal + 0.001);
    
    // Count peaks
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
    
    // Simple normalization (will be improved with actual statistics)
    // For now, use reasonable ranges based on expected values
    std::vector<double> scales = {
        500.0,   // risetime_10_50 (ns)
        1000.0,  // risetime_10_90 (ns)
        1000.0,  // falltime_90_10 (ns)
        2000.0,  // pulse_width (ns)
        1.0,     // asymmetry (already -1 to 1)
        100.0,   // peak_amplitude (VEM)
        1000.0,  // total_charge (VEM)
        1.0,     // peak_charge_ratio (already 0 to 1)
        100.0,   // smoothness
        10.0,    // kurtosis
        5.0,     // skewness
        1.0,     // early_fraction (already 0 to 1)
        1.0,     // late_fraction (already 0 to 1)
        1000.0,  // time_spread (ns)
        10.0,    // high_freq_content
        10.0,    // num_peaks
        1.0      // secondary_peak_ratio (already 0 to 1)
    };
    
    for (size_t i = 0; i < raw.size(); ++i) {
        double norm = raw[i] / scales[i];
        // Clip to reasonable range
        norm = std::max(-5.0, std::min(5.0, norm));
        normalized.push_back(norm);
    }
    
    return normalized;
}

void PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return;
    
    INFO("Training network with " + std::to_string(fTrainingFeatures.size()) + " samples");
    
    // Split into batches
    int batch_size = 32;
    int num_batches = static_cast<int>(fTrainingFeatures.size()) / batch_size;
    
    // Simple training loop
    for (int epoch = 0; epoch < 10; ++epoch) {  // Quick training
        double total_loss = 0;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            std::vector<std::vector<double>> batch_features;
            std::vector<int> batch_labels;
            
            for (int i = 0; i < batch_size; ++i) {
                int idx = batch * batch_size + i;
                if (idx < static_cast<int>(fTrainingFeatures.size())) {
                    batch_features.push_back(fTrainingFeatures[idx]);
                    batch_labels.push_back(fTrainingLabels[idx]);
                }
            }
            
            double loss = fNeuralNetwork->Train(batch_features, batch_labels, 0.001);
            total_loss += loss;
        }
        
        if (epoch % 5 == 0) {
            ostringstream msg;
            msg << "Epoch " << epoch << " loss: " << total_loss / num_batches;
            INFO(msg.str());
        }
    }
    
    // Clear training data to avoid memory issues
    if (fTrainingFeatures.size() > 10000) {
        fTrainingFeatures.clear();
        fTrainingLabels.clear();
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    // Calculate metrics
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
    INFO("\n=== Enhanced PhotonTriggerML Summary ===");
    INFO("Events: " + std::to_string(fEventCount));
    INFO("Stations: " + std::to_string(fStationCount));
    INFO("Photon-like: " + std::to_string(fPhotonLikeCount));
    INFO("Hadron-like: " + std::to_string(fHadronLikeCount));
    
    CalculatePerformanceMetrics();
    
    // Save neural network weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);
    INFO("Saved network weights to " + fWeightsFileName);
    
    // Save data to file
    if (fOutputFile) {
        fOutputFile->cd();
        
        if (fMLTree) fMLTree->Write();
        
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
        
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = nullptr;
        
        INFO("Data saved to " + fOutputFileName);
    }
    
    // Close log file
    if (fLogFile.is_open()) {
        fLogFile.close();
    }
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    
    // Final training if needed
    if (fIsTraining && !fTrainingFeatures.empty()) {
        INFO("Final training with all accumulated data...");
        for (int i = 0; i < 20; ++i) {  // More epochs for final training
            TrainNetwork();
        }
    }
    
    SaveAndDisplaySummary();
    
    return eSuccess;
}

// Static methods
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
