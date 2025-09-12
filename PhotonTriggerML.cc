// PhotonTriggerML.cc - BALANCED VERSION WITH IMPROVED TRAINING
// This version properly handles class imbalance and improves photon detection

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

#include <iostream>
#include <sstream>
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

// Static instance pointer for signal handler
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

// Static ML results map
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Buffer size constants
const int PhotonTriggerML::kMaxPhotonBuffer;
const int PhotonTriggerML::kMaxHadronBuffer;

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
// Improved Neural Network Implementation
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fNumLayers(0), fTimeStep(0),
    fDropoutRate(0.3), fL2Lambda(0.0001)
{
}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, const std::vector<int>& hidden_sizes)
{
    fInputSize = input_size;
    fHiddenSizes = hidden_sizes;
    fNumLayers = hidden_sizes.size() + 1; // Hidden layers + output layer
    
    cout << "Initializing Improved Neural Network: " << input_size;
    for (int size : hidden_sizes) {
        cout << " -> " << size;
    }
    cout << " -> 1" << endl;
    
    // Random initialization
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    
    // Initialize weights and biases for each layer
    fWeights.clear();
    fBiases.clear();
    fMomentum1_w.clear();
    fMomentum2_w.clear();
    fMomentum1_b.clear();
    fMomentum2_b.clear();
    
    // Batch norm parameters
    fBatchNormMean.clear();
    fBatchNormVar.clear();
    fBatchNormGamma.clear();
    fBatchNormBeta.clear();
    
    int prev_size = input_size;
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        int curr_size = hidden_sizes[i];
        
        // He initialization for ReLU variants
        double scale = sqrt(2.0 / prev_size);
        
        // Weight matrix for this layer
        std::vector<std::vector<double>> layer_weights(curr_size, std::vector<double>(prev_size));
        for (int j = 0; j < curr_size; ++j) {
            for (int k = 0; k < prev_size; ++k) {
                layer_weights[j][k] = dist(gen) * scale;
            }
        }
        fWeights.push_back(layer_weights);
        
        // Bias vector for this layer
        std::vector<double> layer_bias(curr_size, 0.01);
        fBiases.push_back(layer_bias);
        
        // Adam optimizer state
        fMomentum1_w.push_back(std::vector<std::vector<double>>(curr_size, std::vector<double>(prev_size, 0)));
        fMomentum2_w.push_back(std::vector<std::vector<double>>(curr_size, std::vector<double>(prev_size, 0)));
        fMomentum1_b.push_back(std::vector<double>(curr_size, 0));
        fMomentum2_b.push_back(std::vector<double>(curr_size, 0));
        
        // Batch norm parameters
        fBatchNormMean.push_back(std::vector<double>(curr_size, 0));
        fBatchNormVar.push_back(std::vector<double>(curr_size, 1));
        fBatchNormGamma.push_back(std::vector<double>(curr_size, 1));
        fBatchNormBeta.push_back(std::vector<double>(curr_size, 0));
        
        prev_size = curr_size;
    }
    
    // Output layer (single neuron for binary classification)
    double scale = sqrt(2.0 / prev_size);
    std::vector<std::vector<double>> output_weights(1, std::vector<double>(prev_size));
    for (int k = 0; k < prev_size; ++k) {
        output_weights[0][k] = dist(gen) * scale * 0.1; // Smaller initial weights for output
    }
    fWeights.push_back(output_weights);
    
    std::vector<double> output_bias(1, 0.0); // No initial bias for output
    fBiases.push_back(output_bias);
    
    fMomentum1_w.push_back(std::vector<std::vector<double>>(1, std::vector<double>(prev_size, 0)));
    fMomentum2_w.push_back(std::vector<std::vector<double>>(1, std::vector<double>(prev_size, 0)));
    fMomentum1_b.push_back(std::vector<double>(1, 0));
    fMomentum2_b.push_back(std::vector<double>(1, 0));
    
    fTimeStep = 0;
    
    cout << "Neural Network initialized with improved architecture" << endl;
    cout << "  - Dropout rate: " << fDropoutRate << endl;
    cout << "  - L2 regularization: " << fL2Lambda << endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& features, bool training)
{
    if (static_cast<int>(features.size()) != fInputSize) {
        return 0.5;
    }
    
    // Forward pass through the network
    std::vector<double> activations = features;
    
    for (size_t layer = 0; layer < fWeights.size() - 1; ++layer) {
        std::vector<double> next_activations(fWeights[layer].size());
        
        // Linear transformation
        for (size_t i = 0; i < fWeights[layer].size(); ++i) {
            double sum = fBiases[layer][i];
            for (size_t j = 0; j < activations.size(); ++j) {
                sum += fWeights[layer][i][j] * activations[j];
            }
            
            // Simplified batch norm (using running statistics)
            if (!training && layer < fBatchNormMean.size()) {
                double mean = fBatchNormMean[layer][i];
                double var = fBatchNormVar[layer][i];
                double gamma = fBatchNormGamma[layer][i];
                double beta = fBatchNormBeta[layer][i];
                sum = gamma * (sum - mean) / sqrt(var + 1e-8) + beta;
            }
            
            // Leaky ReLU activation
            next_activations[i] = LeakyReLU(sum);
            
            // Dropout during training
            if (training && (rand() / double(RAND_MAX)) < fDropoutRate) {
                next_activations[i] = 0;
            } else if (training) {
                next_activations[i] /= (1.0 - fDropoutRate);
            }
        }
        
        activations = next_activations;
    }
    
    // Output layer (sigmoid activation)
    double output = fBiases.back()[0];
    for (size_t j = 0; j < activations.size(); ++j) {
        output += fWeights.back()[0][j] * activations[j];
    }
    
    return Sigmoid(output);
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& features,
                                             const std::vector<int>& labels,
                                             const std::vector<double>& weights,
                                             double learning_rate)
{
    if (features.empty() || features.size() != labels.size()) {
        return -1.0;
    }
    
    double total_loss = 0.0;
    int batch_size = static_cast<int>(features.size());
    
    // Initialize gradient accumulators
    std::vector<std::vector<std::vector<double>>> grad_w(fWeights.size());
    std::vector<std::vector<double>> grad_b(fBiases.size());
    
    for (size_t layer = 0; layer < fWeights.size(); ++layer) {
        grad_w[layer].resize(fWeights[layer].size(), 
                            std::vector<double>(fWeights[layer][0].size(), 0));
        grad_b[layer].resize(fBiases[layer].size(), 0);
    }
    
    // Process each sample
    for (int sample = 0; sample < batch_size; ++sample) {
        const auto& input = features[sample];
        int label = labels[sample];
        double sample_weight = weights[sample];
        
        // Forward pass with intermediate storage
        std::vector<std::vector<double>> layer_inputs;
        std::vector<std::vector<double>> layer_outputs;
        std::vector<std::vector<bool>> dropout_masks;
        
        layer_inputs.push_back(input);
        std::vector<double> activations = input;
        
        // Forward through hidden layers
        for (size_t layer = 0; layer < fWeights.size() - 1; ++layer) {
            std::vector<double> pre_activation(fWeights[layer].size());
            std::vector<double> post_activation(fWeights[layer].size());
            std::vector<bool> dropout_mask(fWeights[layer].size());
            
            for (size_t i = 0; i < fWeights[layer].size(); ++i) {
                double sum = fBiases[layer][i];
                for (size_t j = 0; j < activations.size(); ++j) {
                    sum += fWeights[layer][i][j] * activations[j];
                }
                pre_activation[i] = sum;
                
                // Leaky ReLU
                post_activation[i] = LeakyReLU(sum);
                
                // Dropout
                dropout_mask[i] = (rand() / double(RAND_MAX)) >= fDropoutRate;
                if (!dropout_mask[i]) {
                    post_activation[i] = 0;
                } else {
                    post_activation[i] /= (1.0 - fDropoutRate);
                }
            }
            
            layer_outputs.push_back(pre_activation);
            dropout_masks.push_back(dropout_mask);
            activations = post_activation;
            layer_inputs.push_back(activations);
        }
        
        // Output layer
        double output_raw = fBiases.back()[0];
        for (size_t j = 0; j < activations.size(); ++j) {
            output_raw += fWeights.back()[0][j] * activations[j];
        }
        double output = Sigmoid(output_raw);
        
        // Weighted binary cross-entropy loss
        double loss = -sample_weight * (label * log(output + 1e-7) + 
                                        (1 - label) * log(1 - output + 1e-7));
        total_loss += loss;
        
        // Backward pass
        double output_grad = sample_weight * (output - label);
        
        // Output layer gradients
        for (size_t j = 0; j < activations.size(); ++j) {
            grad_w.back()[0][j] += output_grad * activations[j];
        }
        grad_b.back()[0] += output_grad;
        
        // Backpropagate through hidden layers
        std::vector<double> prev_grad(activations.size());
        for (size_t j = 0; j < activations.size(); ++j) {
            prev_grad[j] = fWeights.back()[0][j] * output_grad;
        }
        
        for (int layer = fWeights.size() - 2; layer >= 0; --layer) {
            std::vector<double> layer_grad(fWeights[layer].size());
            
            for (size_t i = 0; i < fWeights[layer].size(); ++i) {
                if (dropout_masks[layer][i]) {
                    double grad = prev_grad[i];
                    grad *= LeakyReLUDerivative(layer_outputs[layer][i]);
                    grad /= (1.0 - fDropoutRate);
                    layer_grad[i] = grad;
                    
                    for (size_t j = 0; j < layer_inputs[layer].size(); ++j) {
                        grad_w[layer][i][j] += grad * layer_inputs[layer][j];
                    }
                    grad_b[layer][i] += grad;
                }
            }
            
            if (layer > 0) {
                prev_grad.resize(layer_inputs[layer].size());
                for (size_t j = 0; j < layer_inputs[layer].size(); ++j) {
                    prev_grad[j] = 0;
                    for (size_t i = 0; i < fWeights[layer].size(); ++i) {
                        if (dropout_masks[layer][i]) {
                            prev_grad[j] += fWeights[layer][i][j] * layer_grad[i];
                        }
                    }
                }
            }
        }
    }
    
    // Update weights with Adam optimizer
    fTimeStep++;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    
    for (size_t layer = 0; layer < fWeights.size(); ++layer) {
        for (size_t i = 0; i < fWeights[layer].size(); ++i) {
            for (size_t j = 0; j < fWeights[layer][i].size(); ++j) {
                double grad = grad_w[layer][i][j] / batch_size;
                
                // L2 regularization
                grad += fL2Lambda * fWeights[layer][i][j];
                
                // Adam update
                fMomentum1_w[layer][i][j] = beta1 * fMomentum1_w[layer][i][j] + (1 - beta1) * grad;
                fMomentum2_w[layer][i][j] = beta2 * fMomentum2_w[layer][i][j] + (1 - beta2) * grad * grad;
                
                double m_hat = fMomentum1_w[layer][i][j] / (1 - pow(beta1, fTimeStep));
                double v_hat = fMomentum2_w[layer][i][j] / (1 - pow(beta2, fTimeStep));
                
                fWeights[layer][i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }
            
            // Update bias
            double grad = grad_b[layer][i] / batch_size;
            fMomentum1_b[layer][i] = beta1 * fMomentum1_b[layer][i] + (1 - beta1) * grad;
            fMomentum2_b[layer][i] = beta2 * fMomentum2_b[layer][i] + (1 - beta2) * grad * grad;
            
            double m_hat = fMomentum1_b[layer][i] / (1 - pow(beta1, fTimeStep));
            double v_hat = fMomentum2_b[layer][i] / (1 - pow(beta2, fTimeStep));
            
            fBiases[layer][i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
    
    return total_loss / batch_size;
}

double PhotonTriggerML::NeuralNetwork::CalculateOptimalThreshold(
    const std::vector<std::vector<double>>& val_features,
    const std::vector<int>& val_labels)
{
    if (val_features.empty()) return 0.5;
    
    // Get predictions for validation set
    std::vector<double> predictions;
    for (const auto& features : val_features) {
        predictions.push_back(Predict(features, false));
    }
    
    // Try different thresholds and find best F1 score
    double best_threshold = 0.5;
    double best_f1 = 0;
    
    for (double threshold = 0.1; threshold <= 0.9; threshold += 0.05) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool predicted = (predictions[i] > threshold);
            bool actual = (val_labels[i] == 1);
            
            if (predicted && actual) tp++;
            else if (predicted && !actual) fp++;
            else if (!predicted && actual) fn++;
            else tn++;
        }
        
        double precision = (tp + fp > 0) ? double(tp) / (tp + fp) : 0;
        double recall = (tp + fn > 0) ? double(tp) / (tp + fn) : 0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;
        
        if (f1 > best_f1) {
            best_f1 = f1;
            best_threshold = threshold;
        }
    }
    
    cout << "Optimal threshold: " << best_threshold << " (F1: " << best_f1 << ")" << endl;
    return best_threshold;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Could not save weights to " << filename << endl;
        return;
    }
    
    // Save architecture
    file << fInputSize << " " << fHiddenSizes.size() << "\n";
    for (int size : fHiddenSizes) {
        file << size << " ";
    }
    file << "\n";
    
    // Save weights and biases
    for (size_t layer = 0; layer < fWeights.size(); ++layer) {
        for (const auto& row : fWeights[layer]) {
            for (double w : row) file << w << " ";
            file << "\n";
        }
        for (double b : fBiases[layer]) file << b << " ";
        file << "\n";
    }
    
    file.close();
    cout << "Weights saved to " << filename << endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        cout << "Warning: Could not load weights from " << filename << endl;
        return false;
    }
    
    // Load architecture
    int input_size, num_hidden;
    file >> input_size >> num_hidden;
    
    std::vector<int> hidden_sizes(num_hidden);
    for (int i = 0; i < num_hidden; ++i) {
        file >> hidden_sizes[i];
    }
    
    // Reinitialize with loaded architecture
    Initialize(input_size, hidden_sizes);
    
    // Load weights and biases
    for (size_t layer = 0; layer < fWeights.size(); ++layer) {
        for (auto& row : fWeights[layer]) {
            for (double& w : row) file >> w;
        }
        for (double& b : fBiases[layer]) file >> b;
    }
    
    file.close();
    cout << "Weights loaded from " << filename << endl;
    return true;
}

// ============================================================================
// PhotonTriggerML Implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
    fIsTraining(true),
    fTrainingEpochs(200),
    fBatchSize(64),
    fLearningRate(0.001),
    fLearningRateDecay(0.99),
    fBestValidationLoss(1e9),
    fEpochsSinceImprovement(0),
    fPatienceEpochs(20),
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
    fLogFileName("photon_trigger_ml_balanced.log"),
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
    fPhotonThreshold(0.5),
    fAdaptiveThreshold(0.5),
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_balanced.root"),
    fWeightsFileName("photon_trigger_weights_balanced.txt"),
    fLoadPretrainedWeights(false),
    fSaveFeatures(true),
    fApplyTrigger(true),
    fNormalizationInitialized(false),
    fPhotonWeight(1.0),
    fHadronWeight(1.0)
{
    fInstance = this;
    
    // Initialize feature normalization vectors (20 features)
    fFeatureMedians.resize(20, 0.0);
    fFeatureIQRs.resize(20, 1.0);
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Constructor (BALANCED)" << endl;
    cout << "Improved training strategy:" << endl;
    cout << "  - Balanced batch sampling" << endl;
    cout << "  - SMOTE-like synthetic samples" << endl;
    cout << "  - Dynamic threshold adjustment" << endl;
    cout << "  - Better regularization" << endl;
    cout << "==========================================" << endl;
}

PhotonTriggerML::~PhotonTriggerML()
{
    cout << "PhotonTriggerML Destructor called" << endl;
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization (BALANCED)");
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Initialization" << endl;
    cout << "==========================================" << endl;
    
    // Read configuration from XML
    CentralConfig* cc = CentralConfig::GetInstance();
    Branch topBranch = cc->GetTopBranch("PhotonTriggerML");
    
    if (topBranch.GetChild("EnergyMin")) {
        topBranch.GetChild("EnergyMin").GetData(fEnergyMin);
    }
    if (topBranch.GetChild("EnergyMax")) {
        topBranch.GetChild("EnergyMax").GetData(fEnergyMax);
    }
    if (topBranch.GetChild("PhotonThreshold")) {
        topBranch.GetChild("PhotonThreshold").GetData(fPhotonThreshold);
        fAdaptiveThreshold = fPhotonThreshold;
    }
    if (topBranch.GetChild("OutputFile")) {
        topBranch.GetChild("OutputFile").GetData(fOutputFileName);
    }
    if (topBranch.GetChild("SaveFeatures")) {
        topBranch.GetChild("SaveFeatures").GetData(fSaveFeatures);
    }
    if (topBranch.GetChild("ApplyTrigger")) {
        topBranch.GetChild("ApplyTrigger").GetData(fApplyTrigger);
    }
    
    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    
    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML BALANCED Version Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "Configuration:" << endl;
    fLogFile << "  - Energy range: " << fEnergyMin/1e18 << " - " << fEnergyMax/1e18 << " EeV" << endl;
    fLogFile << "  - Initial threshold: " << fPhotonThreshold << endl;
    fLogFile << "  - Batch size: " << fBatchSize << endl;
    fLogFile << "  - Learning rate: " << fLearningRate << endl;
    fLogFile << "==========================================" << endl << endl;
    
    // Initialize neural network with better architecture
    cout << "Initializing Balanced Neural Network..." << endl;
    std::vector<int> hidden_sizes = {32, 24, 16};  // Deeper network with gradual reduction
    fNeuralNetwork->Initialize(20, hidden_sizes);  // 20 input features
    
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
    fMLTree = new TTree("MLTree", "PhotonTriggerML Balanced Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("photonScore", &fPhotonScore, "photonScore/D");
    fMLTree->Branch("confidence", &fConfidence, "confidence/D");
    fMLTree->Branch("primaryId", &fPrimaryId, "primaryId/I");
    fMLTree->Branch("primaryType", &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");
    
    if (fSaveFeatures) {
        fMLTree->Branch("risetime", &fFeatures.risetime_10_90, "risetime/D");
        fMLTree->Branch("pulseWidth", &fFeatures.pulse_width, "pulseWidth/D");
        fMLTree->Branch("asymmetry", &fFeatures.asymmetry, "asymmetry/D");
        fMLTree->Branch("peakChargeRatio", &fFeatures.peak_charge_ratio, "peakChargeRatio/D");
        fMLTree->Branch("kurtosis", &fFeatures.kurtosis, "kurtosis/D");
    }
    
    // Create histograms
    cout << "Creating histograms..." << endl;
    hPhotonScore = new TH1D("hPhotonScore", "ML Photon Score (All);Score;Count", 100, 0, 1);
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons", "ML Score (True Photons);Score;Count", 100, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons", "ML Score (True Hadrons);Score;Count", 100, 0, 1);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence = new TH1D("hConfidence", "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime = new TH1D("hRisetime", "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry = new TH1D("hAsymmetry", "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis = new TH1D("hKurtosis", "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score", 
                              50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 
                                50, 0, 3000, 50, 0, 1);
    
    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 
                                2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");
    
    hTrainingLoss = new TH1D("hTrainingLoss", "Training Loss;Epoch;Loss", 500, 0, 500);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Epoch;Loss", 500, 0, 500);
    hAccuracyHistory = new TH1D("hAccuracyHistory", "Accuracy History;Epoch;Accuracy [%]", 500, 0, 500);
    
    // Register signal handler
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    
    cout << "Initialization complete!" << endl;
    cout << "==========================================" << endl << endl;
    
    INFO("PhotonTriggerML initialized successfully");
    
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    
    // Clear previous ML results
    ClearMLResults();
    
    // Print header every 50 events
    if (fEventCount % 50 == 1) {
        cout << "\n┌────────┬──────────┬─────────┬─────────┬──────────┬─────────┬─────────┐" << endl;
        cout << "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│" << endl;
        cout << "├────────┼──────────┼─────────┼─────────┼──────────┼─────────┼─────────┤" << endl;
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
        
        // Skip events outside energy range
        if (fEnergy < fEnergyMin || fEnergy > fEnergyMax) {
            return eSuccess;
        }
        
        fPrimaryId = shower.GetPrimaryParticle();
        
        switch(fPrimaryId) {
            case 22: fPrimaryType = "photon"; fIsActualPhoton = true; break;
            case 11: case -11: fPrimaryType = "electron"; break;
            case 2212: fPrimaryType = "proton"; break;
            case 1000026056: fPrimaryType = "iron"; break;
            default: fPrimaryType = (fPrimaryId > 1000000000) ? "nucleus" : "unknown";
        }
        
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const utl::CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            utl::Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }
        
        // Print first few events for debugging
        if (fEventCount <= 3) {
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
    
    // Train network periodically with balanced batches
    if (fIsTraining && fEventCount % 10 == 0) {
        // Generate synthetic samples if needed
        if (fPhotonFeatures.size() < 100 && fPhotonFeatures.size() > 10) {
            GenerateSyntheticPhotons();
        }
        
        // Train if we have enough samples
        if (fPhotonFeatures.size() >= 20 && fHadronFeatures.size() >= 20) {
            double val_loss = TrainNetwork();
            
            int epoch = fEventCount / 10;
            hTrainingLoss->SetBinContent(epoch, val_loss);
            
            // Update adaptive threshold periodically
            if (epoch % 10 == 0) {
                UpdateThreshold();
            }
            
            // Calculate current accuracy
            int correct = fTruePositives + fTrueNegatives;
            int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
            if (total > 0) {
                double accuracy = 100.0 * correct / total;
                hAccuracyHistory->SetBinContent(epoch, accuracy);
            }
            
            // Early stopping
            if (val_loss < fBestValidationLoss) {
                fBestValidationLoss = val_loss;
                fEpochsSinceImprovement = 0;
                fNeuralNetwork->SaveWeights("best_" + fWeightsFileName);
            } else {
                fEpochsSinceImprovement++;
                if (fEpochsSinceImprovement > fPatienceEpochs) {
                    cout << "  [Early stopping triggered]" << endl;
                    fIsTraining = false;
                }
            }
        }
    }
    
    // Update metrics display
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        
        // Update confusion matrix
        hConfusionMatrix->SetBinContent(1, 1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2, 1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1, 2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2, 2, fTruePositives);
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
        const utl::CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
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
        
        // Extract features
        fFeatures = ExtractEnhancedFeatures(trace_data);
        
        // Fill feature histograms
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        
        // Update feature statistics if needed
        if (!fNormalizationInitialized && fStationCount > 100) {
            UpdateFeatureStatistics(fFeatures);
            if (fStationCount == 200) {
                fNormalizationInitialized = true;
                cout << "Feature normalization initialized after 200 samples" << endl;
            }
        }
        
        // Normalize features
        std::vector<double> normalized = NormalizeFeatures(fFeatures);
        
        // Get ML prediction
        fPhotonScore = fNeuralNetwork->Predict(normalized, false);
        fConfidence = abs(fPhotonScore - 0.5);
        
        // Store training data with balanced sampling
        if (fIsTraining) {
            bool isValidation = (fStationCount % 10 == 0);
            
            if (isValidation) {
                fValidationFeatures.push_back(normalized);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            } else {
                if (fIsActualPhoton) {
                    fPhotonFeatures.push_back(normalized);
                    if (fPhotonFeatures.size() > kMaxPhotonBuffer) {
                        fPhotonFeatures.pop_front();
                    }
                } else {
                    // Subsample hadrons to maintain balance
                    if (rand() % 100 < 10) {  // Keep 10% of hadrons
                        fHadronFeatures.push_back(normalized);
                        if (fHadronFeatures.size() > kMaxHadronBuffer) {
                            fHadronFeatures.pop_front();
                        }
                    }
                }
            }
        }
        
        fStationCount++;
        bool identifiedAsPhoton = (fPhotonScore > fAdaptiveThreshold);
        
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
        if (fSaveFeatures) {
            fMLTree->Fill();
        }
    }
}

PhotonTriggerML::EnhancedFeatures PhotonTriggerML::ExtractEnhancedFeatures(
    const std::vector<double>& trace, double baseline)
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
    double estimated_baseline = baseline;
    if (baseline <= 0) {
        estimated_baseline = 0;
        for (int i = 0; i < min(100, trace_size); i++) {
            estimated_baseline += trace[i];
        }
        estimated_baseline /= min(100, trace_size);
    }
    
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
    features.peak_time = peak_bin * NS_PER_BIN;
    
    // Calculate rise and fall times
    double peak_10 = 0.1 * peak_value;
    double peak_50 = 0.5 * peak_value;
    double peak_90 = 0.9 * peak_value;
    
    int bin_10_rise = 0, bin_50_rise = 0, bin_90_rise = peak_bin;
    
    // Find rise times
    for (int i = peak_bin; i >= 0; i--) {
        if (signal[i] <= peak_90 && bin_90_rise == peak_bin) bin_90_rise = i;
        if (signal[i] <= peak_50 && bin_50_rise == 0) bin_50_rise = i;
        if (signal[i] <= peak_10) {
            bin_10_rise = i;
            break;
        }
    }
    
    int bin_90_fall = peak_bin, bin_10_fall = trace_size - 1;
    
    // Find fall times
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
    
    // Rise/fall ratio (important for photon discrimination)
    features.rise_fall_ratio = features.risetime_10_90 / (features.falltime_90_10 + 1.0);
    
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
    
    features.pulse_width = abs(bin_half_fall - bin_half_rise) * NS_PER_BIN;
    
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
    features.charge_asymmetry = (early_charge - late_charge) / (early_charge + late_charge + 0.001);
    
    // Smoothness (second derivative RMS)
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
    
    // High frequency content (simplified)
    double high_freq = 0;
    for (int i = 1; i < trace_size - 1; i++) {
        double diff = signal[i+1] - signal[i-1];
        high_freq += diff * diff;
    }
    features.high_freq_content = high_freq / (total_signal * total_signal + 0.001);
    
    // Peak counting
    features.num_peaks = 0;
    double secondary_peak = 0;
    double peak_threshold = 0.15 * peak_value;
    
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

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& features)
{
    std::vector<double> normalized;
    
    // Collect all features
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
        features.secondary_peak_ratio,
        features.rise_fall_ratio,
        features.peak_time,
        features.charge_asymmetry
    };
    
    // Use robust scaling if initialized, otherwise use fixed ranges
    if (fNormalizationInitialized) {
        for (size_t i = 0; i < raw.size(); ++i) {
            double val = (raw[i] - fFeatureMedians[i]) / (fFeatureIQRs[i] + 0.001);
            val = max(-3.0, min(3.0, val));  // Clip to [-3, 3]
            val = (val + 3.0) / 6.0;  // Scale to [0, 1]
            normalized.push_back(val);
        }
    } else {
        // Use fixed normalization ranges
        std::vector<double> mins = {0, 0, 0, 0, -1, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, 0, 0, -1};
        std::vector<double> maxs = {500, 1000, 1000, 1000, 1, 10, 100, 1, 100, 20, 5, 1, 1, 1000, 10, 10, 1, 2, 50000, 1};
        
        for (size_t i = 0; i < raw.size(); ++i) {
            double val = (raw[i] - mins[i]) / (maxs[i] - mins[i] + 0.001);
            val = max(0.0, min(1.0, val));
            normalized.push_back(val);
        }
    }
    
    return normalized;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& features)
{
    static std::vector<std::vector<double>> feature_buffer;
    
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
        features.secondary_peak_ratio,
        features.rise_fall_ratio,
        features.peak_time,
        features.charge_asymmetry
    };
    
    feature_buffer.push_back(raw);
    
    // Calculate medians and IQRs when we have enough samples
    if (feature_buffer.size() == 200) {
        for (size_t i = 0; i < raw.size(); ++i) {
            std::vector<double> feature_values;
            for (const auto& sample : feature_buffer) {
                feature_values.push_back(sample[i]);
            }
            
            std::sort(feature_values.begin(), feature_values.end());
            
            // Calculate median
            size_t n = feature_values.size();
            fFeatureMedians[i] = feature_values[n/2];
            
            // Calculate IQR (75th percentile - 25th percentile)
            double q1 = feature_values[n/4];
            double q3 = feature_values[3*n/4];
            fFeatureIQRs[i] = q3 - q1;
            
            if (fFeatureIQRs[i] < 0.001) fFeatureIQRs[i] = 1.0;
        }
    }
}

double PhotonTriggerML::TrainNetwork()
{
    if (fPhotonFeatures.empty() || fHadronFeatures.empty()) return 1e9;
    
    // Create balanced batch
    std::vector<std::vector<double>> batch_features;
    std::vector<int> batch_labels;
    std::vector<double> batch_weights;
    
    // Calculate class weights based on current imbalance
    double photon_ratio = double(fPhotonFeatures.size()) / 
                         (fPhotonFeatures.size() + fHadronFeatures.size());
    fPhotonWeight = 0.5 / (photon_ratio + 0.001);
    fHadronWeight = 0.5 / (1.0 - photon_ratio + 0.001);
    
    // Sample equal numbers from each class
    int samples_per_class = min(fBatchSize / 2, 
                               min((int)fPhotonFeatures.size(), (int)fHadronFeatures.size()));
    
    // Randomly sample photons
    std::vector<int> photon_indices(fPhotonFeatures.size());
    std::iota(photon_indices.begin(), photon_indices.end(), 0);
    std::random_shuffle(photon_indices.begin(), photon_indices.end());
    
    for (int i = 0; i < samples_per_class; ++i) {
        batch_features.push_back(fPhotonFeatures[photon_indices[i]]);
        batch_labels.push_back(1);
        batch_weights.push_back(fPhotonWeight);
    }
    
    // Randomly sample hadrons
    std::vector<int> hadron_indices(fHadronFeatures.size());
    std::iota(hadron_indices.begin(), hadron_indices.end(), 0);
    std::random_shuffle(hadron_indices.begin(), hadron_indices.end());
    
    for (int i = 0; i < samples_per_class; ++i) {
        batch_features.push_back(fHadronFeatures[hadron_indices[i]]);
        batch_labels.push_back(0);
        batch_weights.push_back(fHadronWeight);
    }
    
    // Shuffle the batch
    std::vector<int> batch_indices(batch_features.size());
    std::iota(batch_indices.begin(), batch_indices.end(), 0);
    std::random_shuffle(batch_indices.begin(), batch_indices.end());
    
    std::vector<std::vector<double>> shuffled_features;
    std::vector<int> shuffled_labels;
    std::vector<double> shuffled_weights;
    
    for (int idx : batch_indices) {
        shuffled_features.push_back(batch_features[idx]);
        shuffled_labels.push_back(batch_labels[idx]);
        shuffled_weights.push_back(batch_weights[idx]);
    }
    
    // Train with current learning rate
    double current_lr = fLearningRate * pow(fLearningRateDecay, fEventCount / 100);
    double train_loss = fNeuralNetwork->Train(shuffled_features, shuffled_labels, 
                                              shuffled_weights, current_lr);
    
    // Calculate validation loss if we have validation data
    double val_loss = train_loss;
    if (!fValidationFeatures.empty()) {
        val_loss = 0;
        int correct = 0;
        
        for (size_t i = 0; i < fValidationFeatures.size(); ++i) {
            double pred = fNeuralNetwork->Predict(fValidationFeatures[i], false);
            int label = fValidationLabels[i];
            
            val_loss -= (label * log(pred + 1e-7) + (1 - label) * log(1 - pred + 1e-7));
            
            bool predicted = (pred > fAdaptiveThreshold);
            if ((predicted && label) || (!predicted && !label)) {
                correct++;
            }
        }
        
        val_loss /= fValidationFeatures.size();
        double val_acc = 100.0 * correct / fValidationFeatures.size();
        
        cout << "  Training - Loss: " << fixed << setprecision(4) << train_loss
             << ", Val Loss: " << val_loss
             << ", Val Acc: " << val_acc << "%"
             << ", LR: " << setprecision(5) << current_lr << endl;
    }
    
    return val_loss;
}

void PhotonTriggerML::GenerateSyntheticPhotons()
{
    if (fPhotonFeatures.size() < 10) return;
    
    // SMOTE-like synthetic sample generation
    int num_synthetic = min(100, (int)fHadronFeatures.size() - (int)fPhotonFeatures.size());
    
    for (int i = 0; i < num_synthetic; ++i) {
        // Randomly select two photon samples
        int idx1 = rand() % fPhotonFeatures.size();
        int idx2 = rand() % fPhotonFeatures.size();
        while (idx2 == idx1 && fPhotonFeatures.size() > 1) {
            idx2 = rand() % fPhotonFeatures.size();
        }
        
        // Create synthetic sample by interpolation
        std::vector<double> synthetic(fPhotonFeatures[idx1].size());
        double alpha = (rand() / double(RAND_MAX)) * 0.5 + 0.25;  // Random between 0.25 and 0.75
        
        for (size_t j = 0; j < synthetic.size(); ++j) {
            synthetic[j] = alpha * fPhotonFeatures[idx1][j] + 
                          (1 - alpha) * fPhotonFeatures[idx2][j];
            
            // Add small noise
            synthetic[j] += (rand() / double(RAND_MAX) - 0.5) * 0.02;
            synthetic[j] = max(0.0, min(1.0, synthetic[j]));
        }
        
        fPhotonFeatures.push_back(synthetic);
    }
    
    cout << "  Generated " << num_synthetic << " synthetic photon samples" << endl;
}

void PhotonTriggerML::UpdateThreshold()
{
    if (fValidationFeatures.empty()) return;
    
    double new_threshold = fNeuralNetwork->CalculateOptimalThreshold(
        fValidationFeatures, fValidationLabels);
    
    // Smooth update
    fAdaptiveThreshold = 0.7 * fAdaptiveThreshold + 0.3 * new_threshold;
    
    cout << "  Updated adaptive threshold to " << fAdaptiveThreshold << endl;
    
    // Log to file
    if (fLogFile.is_open()) {
        fLogFile << "Threshold updated at event " << fEventCount 
                << " to " << fAdaptiveThreshold << endl;
    }
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
                << "% F1: " << f1 
                << "% (TP: " << fTruePositives
                << " FP: " << fFalsePositives
                << " TN: " << fTrueNegatives
                << " FN: " << fFalseNegatives << ")" << endl;
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n==========================================" << endl;
    cout << "PHOTONTRIGGERML FINAL SUMMARY (BALANCED)" << endl;
    cout << "==========================================" << endl;
    
    cout << "Events processed: " << fEventCount << endl;
    cout << "Stations analyzed: " << fStationCount << endl;
    cout << "Final adaptive threshold: " << fAdaptiveThreshold << endl;
    
    // Calculate final metrics
    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total > 0) {
        double accuracy = 100.0 * (fTruePositives + fTrueNegatives) / total;
        double precision = (fTruePositives + fFalsePositives > 0) ? 
                          100.0 * fTruePositives / (fTruePositives + fFalsePositives) : 0;
        double recall = (fTruePositives + fFalseNegatives > 0) ? 
                       100.0 * fTruePositives / (fTruePositives + fFalseNegatives) : 0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;
        double specificity = (fTrueNegatives + fFalsePositives > 0) ?
                            100.0 * fTrueNegatives / (fTrueNegatives + fFalsePositives) : 0;
        
        cout << "\nPERFORMANCE METRICS:" << endl;
        cout << "Accuracy:    " << fixed << setprecision(1) << accuracy << "%" << endl;
        cout << "Precision:   " << precision << "%" << endl;
        cout << "Recall:      " << recall << "%" << endl;
        cout << "Specificity: " << specificity << "%" << endl;
        cout << "F1-Score:    " << f1 << "%" << endl;
        cout << endl;
        
        cout << "CONFUSION MATRIX:" << endl;
        cout << "                Predicted" << endl;
        cout << "             Hadron   Photon" << endl;
        cout << "Actual Hadron  " << setw(6) << fTrueNegatives << "   " << setw(6) << fFalsePositives << endl;
        cout << "       Photon  " << setw(6) << fFalseNegatives << "   " << setw(6) << fTruePositives << endl;
    }
    
    // Save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);
    
    // Save ROOT file
    if (fOutputFile) {
        fOutputFile->cd();
        
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
    
    // Close log file
    if (fLogFile.is_open()) {
        fLogFile.close();
    }
    
    cout << "==========================================" << endl;
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    
    SaveAndDisplaySummary();
    
    return eSuccess;
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
