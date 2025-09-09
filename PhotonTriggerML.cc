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
// Improved Neural Network Implementation with Focal Loss
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.2),  // Moderate dropout
    fIsQuantized(false), fQuantizationScale(127.0)
{
}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize = input_size;
    fHidden1Size = hidden1_size;
    fHidden2Size = hidden2_size;
    
    cout << "Initializing Improved Neural Network: " << input_size << " -> " 
         << hidden1_size << " -> " << hidden2_size << " -> 1" << endl;
    
    // Use fixed seed initially for reproducibility
    std::mt19937 gen(42);
    
    // Xavier initialization for better gradient flow
    std::normal_distribution<> dist(0.0, 1.0);
    
    // Initialize weights with Xavier/He initialization
    fWeights1.resize(hidden1_size, std::vector<double>(input_size));
    double scale1 = sqrt(2.0 / input_size);
    for (int i = 0; i < hidden1_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            fWeights1[i][j] = dist(gen) * scale1;
        }
    }
    
    fWeights2.resize(hidden2_size, std::vector<double>(hidden1_size));
    double scale2 = sqrt(2.0 / hidden1_size);
    for (int i = 0; i < hidden2_size; ++i) {
        for (int j = 0; j < hidden1_size; ++j) {
            fWeights2[i][j] = dist(gen) * scale2;
        }
    }
    
    fWeights3.resize(1, std::vector<double>(hidden2_size));
    double scale3 = sqrt(2.0 / hidden2_size);
    for (int j = 0; j < hidden2_size; ++j) {
        fWeights3[0][j] = dist(gen) * scale3;
    }
    
    // Small random biases
    std::uniform_real_distribution<> bias_dist(-0.01, 0.01);
    fBias1.resize(hidden1_size);
    for (int i = 0; i < hidden1_size; ++i) {
        fBias1[i] = bias_dist(gen);
    }
    
    fBias2.resize(hidden2_size);
    for (int i = 0; i < hidden2_size; ++i) {
        fBias2[i] = bias_dist(gen);
    }
    
    fBias3 = bias_dist(gen);
    
    // Initialize Adam optimizer momentum terms
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
    
    cout << "Neural Network initialized with improved architecture!" << endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& features, bool training)
{
    if (static_cast<int>(features.size()) != fInputSize) {
        return 0.5;
    }
    
    // Forward pass with ReLU activation for hidden layers
    std::vector<double> hidden1(fHidden1Size);
    for (int i = 0; i < fHidden1Size; ++i) {
        double sum = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) {
            sum += fWeights1[i][j] * features[j];
        }
        hidden1[i] = ReLU(sum);
        
        // Apply dropout during training
        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) {
            hidden1[i] = 0;
        } else if (training) {
            hidden1[i] /= (1.0 - fDropoutRate);  // Scale during training
        }
    }
    
    std::vector<double> hidden2(fHidden2Size);
    for (int i = 0; i < fHidden2Size; ++i) {
        double sum = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) {
            sum += fWeights2[i][j] * hidden1[j];
        }
        hidden2[i] = ReLU(sum);
        
        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) {
            hidden2[i] = 0;
        } else if (training) {
            hidden2[i] /= (1.0 - fDropoutRate);
        }
    }
    
    // Output layer with sigmoid
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
    
    // Count classes for focal loss weighting
    // int num_photons = std::count(labels.begin(), labels.end(), 1);
    // int num_hadrons = batch_size - num_photons;  // Removed unused variable
    
    // Focal loss parameters
    double gamma = 2.0;   // Focusing parameter
    double alpha = 0.25;  // Weight for positive class (photons)
    
    // L1 and L2 regularization
    double lambda_l1 = 0.0001;
    double lambda_l2 = 0.001;
    
    // Initialize gradients
    std::vector<std::vector<double>> grad_w1(fHidden1Size, std::vector<double>(fInputSize, 0));
    std::vector<std::vector<double>> grad_w2(fHidden2Size, std::vector<double>(fHidden1Size, 0));
    std::vector<std::vector<double>> grad_w3(1, std::vector<double>(fHidden2Size, 0));
    std::vector<double> grad_b1(fHidden1Size, 0);
    std::vector<double> grad_b2(fHidden2Size, 0);
    double grad_b3 = 0;
    
    // Process each sample
    for (int sample = 0; sample < batch_size; ++sample) {
        const auto& input = features[sample];
        int label = labels[sample];
        
        // Forward pass
        std::vector<double> hidden1(fHidden1Size);
        std::vector<double> hidden1_raw(fHidden1Size);
        std::vector<bool> dropout1(fHidden1Size);
        
        for (int i = 0; i < fHidden1Size; ++i) {
            double sum = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) {
                sum += fWeights1[i][j] * input[j];
            }
            hidden1_raw[i] = sum;
            hidden1[i] = ReLU(sum);
            dropout1[i] = (rand() / double(RAND_MAX)) >= fDropoutRate;
            if (!dropout1[i]) hidden1[i] = 0;
            else hidden1[i] /= (1.0 - fDropoutRate);
        }
        
        std::vector<double> hidden2(fHidden2Size);
        std::vector<double> hidden2_raw(fHidden2Size);
        std::vector<bool> dropout2(fHidden2Size);
        
        for (int i = 0; i < fHidden2Size; ++i) {
            double sum = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) {
                sum += fWeights2[i][j] * hidden1[j];
            }
            hidden2_raw[i] = sum;
            hidden2[i] = ReLU(sum);
            dropout2[i] = (rand() / double(RAND_MAX)) >= fDropoutRate;
            if (!dropout2[i]) hidden2[i] = 0;
            else hidden2[i] /= (1.0 - fDropoutRate);
        }
        
        double output_raw = fBias3;
        for (int j = 0; j < fHidden2Size; ++j) {
            output_raw += fWeights3[0][j] * hidden2[j];
        }
        
        double output = Sigmoid(output_raw);
        
        // Focal loss calculation
        double p = output;
        double pt = label ? p : (1 - p);
        double alpha_t = label ? alpha : (1 - alpha);
        double focal_weight = alpha_t * pow(1 - pt, gamma);
        
        double loss = -focal_weight * (label * log(p + 1e-7) + 
                                       (1 - label) * log(1 - p + 1e-7));
        total_loss += loss;
        
        // Backpropagation with focal loss gradient
        double focal_grad_factor = alpha_t * (
            -gamma * pow(1 - pt, gamma - 1) * log(pt + 1e-7) +
            pow(1 - pt, gamma) / (pt + 1e-7)
        );
        double output_grad = focal_grad_factor * (output - label);
        
        // Output layer gradients
        for (int j = 0; j < fHidden2Size; ++j) {
            grad_w3[0][j] += output_grad * hidden2[j];
        }
        grad_b3 += output_grad;
        
        // Hidden layer 2 gradients
        std::vector<double> hidden2_grad(fHidden2Size);
        for (int j = 0; j < fHidden2Size; ++j) {
            if (dropout2[j]) {
                hidden2_grad[j] = fWeights3[0][j] * output_grad;
                hidden2_grad[j] *= ReLUDerivative(hidden2_raw[j]);
                hidden2_grad[j] /= (1.0 - fDropoutRate);
            }
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
            if (dropout1[j]) {
                hidden1_grad[j] = 0;
                for (int i = 0; i < fHidden2Size; ++i) {
                    hidden1_grad[j] += fWeights2[i][j] * hidden2_grad[i];
                }
                hidden1_grad[j] *= ReLUDerivative(hidden1_raw[j]);
                hidden1_grad[j] /= (1.0 - fDropoutRate);
            }
        }
        
        for (int i = 0; i < fHidden1Size; ++i) {
            for (int j = 0; j < fInputSize; ++j) {
                grad_w1[i][j] += hidden1_grad[i] * input[j];
            }
            grad_b1[i] += hidden1_grad[i];
        }
    }
    
    // Adam optimizer with L1 and L2 regularization
    fTimeStep++;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    
    // Update weights with Adam and regularization
    for (int i = 0; i < fHidden1Size; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            double l1_grad = lambda_l1 * (fWeights1[i][j] > 0 ? 1 : -1);
            double l2_grad = lambda_l2 * fWeights1[i][j];
            double grad = grad_w1[i][j] / batch_size + l1_grad + l2_grad;
            
            fMomentum1_w1[i][j] = beta1 * fMomentum1_w1[i][j] + (1 - beta1) * grad;
            fMomentum2_w1[i][j] = beta2 * fMomentum2_w1[i][j] + (1 - beta2) * grad * grad;
            
            double m_hat = fMomentum1_w1[i][j] / (1 - pow(beta1, fTimeStep));
            double v_hat = fMomentum2_w1[i][j] / (1 - pow(beta2, fTimeStep));
            
            fWeights1[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
        
        double grad = grad_b1[i] / batch_size;
        fMomentum1_b1[i] = beta1 * fMomentum1_b1[i] + (1 - beta1) * grad;
        fMomentum2_b1[i] = beta2 * fMomentum2_b1[i] + (1 - beta2) * grad * grad;
        
        double m_hat = fMomentum1_b1[i] / (1 - pow(beta1, fTimeStep));
        double v_hat = fMomentum2_b1[i] / (1 - pow(beta2, fTimeStep));
        
        fBias1[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
    
    for (int i = 0; i < fHidden2Size; ++i) {
        for (int j = 0; j < fHidden1Size; ++j) {
            double l1_grad = lambda_l1 * (fWeights2[i][j] > 0 ? 1 : -1);
            double l2_grad = lambda_l2 * fWeights2[i][j];
            double grad = grad_w2[i][j] / batch_size + l1_grad + l2_grad;
            
            fMomentum1_w2[i][j] = beta1 * fMomentum1_w2[i][j] + (1 - beta1) * grad;
            fMomentum2_w2[i][j] = beta2 * fMomentum2_w2[i][j] + (1 - beta2) * grad * grad;
            
            double m_hat = fMomentum1_w2[i][j] / (1 - pow(beta1, fTimeStep));
            double v_hat = fMomentum2_w2[i][j] / (1 - pow(beta2, fTimeStep));
            
            fWeights2[i][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
        
        double grad = grad_b2[i] / batch_size;
        fMomentum1_b2[i] = beta1 * fMomentum1_b2[i] + (1 - beta1) * grad;
        fMomentum2_b2[i] = beta2 * fMomentum2_b2[i] + (1 - beta2) * grad * grad;
        
        double m_hat = fMomentum1_b2[i] / (1 - pow(beta1, fTimeStep));
        double v_hat = fMomentum2_b2[i] / (1 - pow(beta2, fTimeStep));
        
        fBias2[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
    
    for (int j = 0; j < fHidden2Size; ++j) {
        double l1_grad = lambda_l1 * (fWeights3[0][j] > 0 ? 1 : -1);
        double l2_grad = lambda_l2 * fWeights3[0][j];
        double grad = grad_w3[0][j] / batch_size + l1_grad + l2_grad;
        
        fMomentum1_w3[0][j] = beta1 * fMomentum1_w3[0][j] + (1 - beta1) * grad;
        fMomentum2_w3[0][j] = beta2 * fMomentum2_w3[0][j] + (1 - beta2) * grad * grad;
        
        double m_hat = fMomentum1_w3[0][j] / (1 - pow(beta1, fTimeStep));
        double v_hat = fMomentum2_w3[0][j] / (1 - pow(beta2, fTimeStep));
        
        fWeights3[0][j] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
    
    double grad = grad_b3 / batch_size;
    fMomentum1_b3 = beta1 * fMomentum1_b3 + (1 - beta1) * grad;
    fMomentum2_b3 = beta2 * fMomentum2_b3 + (1 - beta2) * grad * grad;
    
    double m_hat = fMomentum1_b3 / (1 - pow(beta1, fTimeStep));
    double v_hat = fMomentum2_b3 / (1 - pow(beta2, fTimeStep));
    
    fBias3 -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    
    return total_loss / batch_size;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename);
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
    std::ifstream file(filename);
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
    // Implementation for 8-bit quantization if needed
}

// ============================================================================
// PhotonTriggerML Implementation with Improved Features
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
    fIsTraining(true),
    fTrainingEpochs(200),
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
    fLogFileName("photon_trigger_ml_improved.log"),
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
    fPhotonThreshold(0.6),  // Higher threshold for fewer false positives
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_improved.root"),
    fWeightsFileName("photon_trigger_weights_improved.txt"),
    fLoadPretrainedWeights(false)
{
    fInstance = this;
    
    // Initialize for 19 features (17 original + 2 derived)
    fFeatureMeans.resize(19, 0.0);
    fFeatureStdDevs.resize(19, 1.0);
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Constructor (IMPROVED)" << endl;
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
    INFO("PhotonTriggerML::Init() - Starting initialization (IMPROVED VERSION)");
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Initialization (IMPROVED)" << endl;
    cout << "==========================================" << endl;
    
    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    
    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML Improved Version Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << endl << endl;
    
    // Initialize neural network with larger architecture for 19 features
    cout << "Initializing Improved Neural Network..." << endl;
    fNeuralNetwork->Initialize(19, 16, 10);  // 19 inputs, larger hidden layers
    
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
    fMLTree = new TTree("MLTree", "PhotonTriggerML Improved Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("photonScore", &fPhotonScore, "photonScore/D");
    fMLTree->Branch("confidence", &fConfidence, "confidence/D");
    fMLTree->Branch("primaryId", &fPrimaryId, "primaryId/I");
    fMLTree->Branch("primaryType", &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");
    
    // Feature branches
    fMLTree->Branch("risetime", &fFeatures.risetime_10_90, "risetime/D");
    fMLTree->Branch("pulseWidth", &fFeatures.pulse_width, "pulseWidth/D");
    fMLTree->Branch("asymmetry", &fFeatures.asymmetry, "asymmetry/D");
    fMLTree->Branch("peakChargeRatio", &fFeatures.peak_charge_ratio, "peakChargeRatio/D");
    
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
    hTrainingLoss = new TH1D("hTrainingLoss", "Training Loss;Batch;Loss", 1000, 0, 1000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss", 1000, 0, 1000);
    hAccuracyHistory = new TH1D("hAccuracyHistory", "Accuracy History;Batch;Accuracy [%]", 1000, 0, 1000);
    
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    
    cout << "Initialization complete!" << endl;
    cout << "==========================================" << endl << endl;
    
    INFO("PhotonTriggerML initialized successfully (IMPROVED VERSION)");
    
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
            const utl::CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            utl::Point core = shower.GetSimCore(0);
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
    
    // Train network periodically with improved strategy
    if (fIsTraining && fTrainingFeatures.size() >= 48 && fEventCount % 10 == 0) {
        double val_loss = TrainNetwork();
        
        int batch_num = fEventCount / 10;
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
            cout << "  [New best model saved!]" << endl;
        } else {
            fEpochsSinceImprovement++;
        }
        
        // Early stopping
        if (fEpochsSinceImprovement > 30) {
            cout << "  [Early stopping triggered]" << endl;
            fIsTraining = false;
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
        
        // Add derived physics features
        double muon_likelihood = fFeatures.falltime_90_10 / (fFeatures.risetime_10_90 + 1);
        double em_likelihood = fFeatures.peak_charge_ratio * (1.0 - abs(fFeatures.asymmetry));
        
        // Physics-based discrimination (for reference only, not used in scoring)
        bool physicsPhotonLike = false;
        if (fFeatures.risetime_10_90 < 150 &&  // Sharp rise
            fFeatures.pulse_width < 300 &&      // Narrow pulse
            fFeatures.secondary_peak_ratio < 0.3 && // Single peak
            fFeatures.peak_charge_ratio > 0.15) {   // Concentrated charge
            physicsPhotonLike = true;
        }
        
        // NO physics bias - let the network learn
        double physics_bias = 0.0;
        
        // Fill feature histograms
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        
        // Update feature statistics
        UpdateFeatureStatistics(fFeatures);
        
        // Normalize features including derived ones
        std::vector<double> normalized = NormalizeFeatures(fFeatures);
        
        // Add normalized derived features
        normalized.push_back(muon_likelihood / 10.0);  // Normalize to ~[0,1]
        normalized.push_back(em_likelihood);
        
        // Get ML prediction
        double ml_score = fNeuralNetwork->Predict(normalized, false);
        
        // Apply NO bias
        fPhotonScore = ml_score + physics_bias;
        fPhotonScore = max(0.0, min(1.0, fPhotonScore));  // Clip to [0,1]
        
        fConfidence = abs(fPhotonScore - 0.5);
        
        // Improved training data collection
        bool isValidation = (fStationCount % 10 == 0);
        if (fIsTraining) {
            if (!isValidation) {
                // Reduced oversampling strategy
                if (fIsActualPhoton) {
                    // Minimal oversampling for rare photons
                    for (int copy = 0; copy < 2; copy++) {  // Only 2x
                        std::vector<double> varied = normalized;
                        
                        // Add small random variations
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::normal_distribution<> noise(0, 0.03);
                        
                        for (auto& val : varied) {
                            val += noise(gen);
                            val = max(0.0, min(1.0, val));  // Keep in [0,1]
                        }
                        
                        fTrainingFeatures.push_back(varied);
                        fTrainingLabels.push_back(1);
                    }
                } else if (physicsPhotonLike && !fIsActualPhoton) {
                    // Important false positives - include them
                    fTrainingFeatures.push_back(normalized);
                    fTrainingLabels.push_back(0);
                } else if (rand() % 2 == 0) {
                    // Subsample regular hadrons (50% chance)
                    fTrainingFeatures.push_back(normalized);
                    fTrainingLabels.push_back(0);
                }
            } else {
                // Validation set - no augmentation
                fValidationFeatures.push_back(normalized);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }
        
        fStationCount++;
        bool identifiedAsPhoton = (fPhotonScore > fPhotonThreshold);  // 0.6 threshold
        
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
        
        // Debug output for interesting events
        if ((fIsActualPhoton && identifiedAsPhoton) && fStationCount <= 30) {
            cout << "  TP: Station " << fStationId 
                 << " Score=" << fixed << setprecision(3) << fPhotonScore
                 << " Rise=" << fFeatures.risetime_10_90 << "ns" << endl;
        } else if (fIsActualPhoton && !identifiedAsPhoton && fStationCount <= 30) {
            cout << "  FN: Station " << fStationId 
                 << " Score=" << fixed << setprecision(3) << fPhotonScore
                 << " Rise=" << fFeatures.risetime_10_90 << "ns" << endl;
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
    const std::vector<double>& trace, double /* baseline */)
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
    
    // Add derived features
    double muon_likelihood = features.falltime_90_10 / (features.risetime_10_90 + 1);
    double em_likelihood = features.peak_charge_ratio * (1.0 - abs(features.asymmetry));
    raw.push_back(muon_likelihood);
    raw.push_back(em_likelihood);
    
    static int n = 0;
    n++;
    
    // Online mean and variance calculation (Welford's algorithm)
    for (size_t i = 0; i < raw.size(); ++i) {
        double delta = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += delta / n;
        double delta2 = raw[i] - fFeatureMeans[i];
        if (n > 1) {
            double var = fFeatureStdDevs[i] * fFeatureStdDevs[i] * (n - 2) + delta * delta2;
            fFeatureStdDevs[i] = sqrt(var / (n - 1));
        }
        if (fFeatureStdDevs[i] < 0.001) fFeatureStdDevs[i] = 1.0;
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
    
    // Robust min-max normalization to [0,1]
    std::vector<double> mins = {0, 0, 0, 0, -1, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0};
    std::vector<double> maxs = {500, 1000, 1000, 1000, 1, 10, 100, 1, 100, 20, 5, 1, 1, 1000, 10, 10, 1};
    
    for (size_t i = 0; i < raw.size(); ++i) {
        double val = (raw[i] - mins[i]) / (maxs[i] - mins[i] + 0.001);
        val = max(0.0, min(1.0, val));  // Clip to [0,1]
        normalized.push_back(val);
    }
    
    return normalized;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;
    
    cout << "\n  Training with " << fTrainingFeatures.size() << " samples...";
    
    // Create truly balanced batch with realistic class distribution
    std::vector<std::vector<double>> batch_features;
    std::vector<int> batch_labels;
    
    // Count classes in training data
    // int num_photons_total = std::count(fTrainingLabels.begin(), fTrainingLabels.end(), 1);
    // int num_hadrons_total = fTrainingLabels.size() - num_photons_total;
    
    // Collect indices for each class
    std::vector<int> photon_indices, hadron_indices;
    for (size_t i = 0; i < fTrainingLabels.size(); ++i) {
        if (fTrainingLabels[i] == 1) {
            photon_indices.push_back(i);
        } else {
            hadron_indices.push_back(i);
        }
    }
    
    // Randomly sample from each class
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(photon_indices.begin(), photon_indices.end(), gen);
    std::shuffle(hadron_indices.begin(), hadron_indices.end(), gen);
    
    // Use 1:2 ratio (photon:hadron) to reflect realistic distribution
    int batch_size_photons = min(16, (int)photon_indices.size());
    int batch_size_hadrons = min(32, (int)hadron_indices.size());
    
    for (int i = 0; i < batch_size_photons; ++i) {
        batch_features.push_back(fTrainingFeatures[photon_indices[i]]);
        batch_labels.push_back(1);
    }
    
    for (int i = 0; i < batch_size_hadrons; ++i) {
        batch_features.push_back(fTrainingFeatures[hadron_indices[i]]);
        batch_labels.push_back(0);
    }
    
    // Shuffle the batch
    std::vector<int> batch_indices(batch_features.size());
    std::iota(batch_indices.begin(), batch_indices.end(), 0);
    std::shuffle(batch_indices.begin(), batch_indices.end(), gen);
    
    std::vector<std::vector<double>> shuffled_features;
    std::vector<int> shuffled_labels;
    for (int idx : batch_indices) {
        shuffled_features.push_back(batch_features[idx]);
        shuffled_labels.push_back(batch_labels[idx]);
    }
    
    cout << " (P:" << batch_size_photons << " H:" << batch_size_hadrons << ")";
    
    // Adaptive learning rate with cosine annealing
    double base_lr = 0.005;
    double min_lr = 0.0001;
    double cosine_factor = 0.5 * (1 + cos(M_PI * fTrainingStep / 500.0));
    double learning_rate = min_lr + (base_lr - min_lr) * cosine_factor;
    
    // Train for multiple epochs on this batch
    double total_loss = 0;
    for (int epoch = 0; epoch < 3; epoch++) {  // Reduced epochs per batch
        double loss = fNeuralNetwork->Train(shuffled_features, shuffled_labels, learning_rate);
        total_loss += loss;
    }
    double train_loss = total_loss / 3.0;
    
    cout << " Loss: " << fixed << setprecision(4) << train_loss;
    cout << " LR: " << setprecision(5) << learning_rate;
    
    // Calculate validation loss
    double val_loss = 0;
    if (!fValidationFeatures.empty()) {
        int correct = 0;
        int val_tp = 0, val_fp = 0, val_tn = 0, val_fn = 0;
        
        for (size_t i = 0; i < fValidationFeatures.size(); ++i) {
            double pred = fNeuralNetwork->Predict(fValidationFeatures[i], false);
            int label = fValidationLabels[i];
            
            // Focal loss for validation
            double p = pred;
            double pt = label ? p : (1 - p);
            double alpha = 0.25;
            double gamma = 2.0;
            double alpha_t = label ? alpha : (1 - alpha);
            double focal_weight = alpha_t * pow(1 - pt, gamma);
            val_loss -= focal_weight * (label * log(p + 1e-7) + (1 - label) * log(1 - p + 1e-7));
            
            bool predicted = (pred > fPhotonThreshold);
            if (predicted && label) val_tp++;
            else if (predicted && !label) val_fp++;
            else if (!predicted && label) val_fn++;
            else if (!predicted && !label) val_tn++;
            
            if ((predicted && label) || (!predicted && !label)) {
                correct++;
            }
        }
        
        val_loss /= fValidationFeatures.size();
        double val_acc = 100.0 * correct / fValidationFeatures.size();
        double val_precision = (val_tp + val_fp > 0) ? 100.0 * val_tp / (val_tp + val_fp) : 0;
        double val_recall = (val_tp + val_fn > 0) ? 100.0 * val_tp / (val_tp + val_fn) : 0;
        
        cout << " Val: " << val_loss 
             << " (Acc: " << val_acc << "%"
             << " Prec: " << val_precision << "%"
             << " Rec: " << val_recall << "%)";
        
        int batch_num = fEventCount / 10;
        hValidationLoss->SetBinContent(batch_num, val_loss);
    }
    
    cout << endl;
    
    // Keep limited data size with FIFO strategy
    if (fTrainingFeatures.size() > 3000) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(), 
                               fTrainingFeatures.begin() + 1000);
        fTrainingLabels.erase(fTrainingLabels.begin(), 
                             fTrainingLabels.begin() + 1000);
    }
    
    if (fValidationFeatures.size() > 600) {
        fValidationFeatures.erase(fValidationFeatures.begin(),
                                 fValidationFeatures.begin() + 200);
        fValidationLabels.erase(fValidationLabels.begin(),
                               fValidationLabels.begin() + 200);
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
    double specificity = (fTrueNegatives + fFalsePositives > 0) ?
                        100.0 * fTrueNegatives / (fTrueNegatives + fFalsePositives) : 0;
    
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
        fLogFile << "Specificity: " << specificity << "%" << endl;
        fLogFile << "F1-Score: " << f1 << "%" << endl;
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n==========================================" << endl;
    cout << "PHOTONTRIGGERML FINAL SUMMARY (IMPROVED)" << endl;
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
    
    // Close log file
    if (fLogFile.is_open()) {
        fLogFile.close();
    }
    
    cout << "==========================================" << endl;
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion (IMPROVED VERSION)");
    
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

void PhotonTriggerML::WriteMLAnalysisToLog()
{
    // Optional: Add detailed analysis logging if needed
}
