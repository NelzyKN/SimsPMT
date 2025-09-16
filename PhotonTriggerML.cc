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
#include <TMath.h>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>
#include <ctime>
#include <csignal>
#include <deque>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace det;
using namespace utl;

// Static instance pointer
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

// Static ML results map
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Signal handler
void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        cout << "\n\nInterrupt received. Saving PhotonTriggerML data...\n" << endl;
        if (PhotonTriggerML::fInstance) {
            PhotonTriggerML::fInstance->SaveAndDisplaySummary();
        }
        exit(0);
    }
}

// ============================================================================
// Enhanced Autoencoder for Anomaly Detection
// ============================================================================

PhotonTriggerML::Autoencoder::Autoencoder() :
    fInputSize(0), fLatentSize(4), fInitialized(false), fAnomalyThreshold(0.5)
{
}

void PhotonTriggerML::Autoencoder::Initialize(int input_size)
{
    fInputSize = input_size;
    fLatentSize = 4;  // More aggressive compression for better anomaly detection
    
    // Deep architecture: input -> 32 -> 16 -> 8 -> 4 (latent) -> 8 -> 16 -> 32 -> output
    int hidden1 = 32;
    int hidden2 = 16;
    int hidden3 = 8;
    
    cout << "Initializing Deep Autoencoder: " << input_size 
         << " -> " << hidden1 << " -> " << hidden2 << " -> " << hidden3 
         << " -> " << fLatentSize << " -> ... -> " << input_size << endl;
    
    // Initialize with Xavier/He initialization
    std::mt19937 gen(42);
    
    // Encoder layers
    fEncoderWeights1.resize(hidden1, std::vector<double>(input_size));
    fEncoderBias1.resize(hidden1);
    fEncoderWeights2.resize(hidden2, std::vector<double>(hidden1));
    fEncoderBias2.resize(hidden2);
    fEncoderWeights3.resize(hidden3, std::vector<double>(hidden2));
    fEncoderBias3.resize(hidden3);
    fEncoderWeights4.resize(fLatentSize, std::vector<double>(hidden3));
    fEncoderBias4.resize(fLatentSize);
    
    // Decoder layers (mirror of encoder)
    fDecoderWeights1.resize(hidden3, std::vector<double>(fLatentSize));
    fDecoderBias1.resize(hidden3);
    fDecoderWeights2.resize(hidden2, std::vector<double>(hidden3));
    fDecoderBias2.resize(hidden2);
    fDecoderWeights3.resize(hidden1, std::vector<double>(hidden2));
    fDecoderBias3.resize(hidden1);
    fDecoderWeights4.resize(input_size, std::vector<double>(hidden1));
    fDecoderBias4.resize(input_size);
    
    // Xavier initialization for encoder
    std::normal_distribution<> dist1(0.0, sqrt(2.0 / input_size));
    std::normal_distribution<> dist2(0.0, sqrt(2.0 / hidden1));
    std::normal_distribution<> dist3(0.0, sqrt(2.0 / hidden2));
    std::normal_distribution<> dist4(0.0, sqrt(2.0 / hidden3));
    
    // Initialize encoder weights
    for (int i = 0; i < hidden1; ++i) {
        for (int j = 0; j < input_size; ++j) {
            fEncoderWeights1[i][j] = dist1(gen);
        }
        fEncoderBias1[i] = 0.01;
    }
    
    for (int i = 0; i < hidden2; ++i) {
        for (int j = 0; j < hidden1; ++j) {
            fEncoderWeights2[i][j] = dist2(gen);
        }
        fEncoderBias2[i] = 0.01;
    }
    
    for (int i = 0; i < hidden3; ++i) {
        for (int j = 0; j < hidden2; ++j) {
            fEncoderWeights3[i][j] = dist3(gen);
        }
        fEncoderBias3[i] = 0.01;
    }
    
    for (int i = 0; i < fLatentSize; ++i) {
        for (int j = 0; j < hidden3; ++j) {
            fEncoderWeights4[i][j] = dist4(gen);
        }
        fEncoderBias4[i] = 0.01;
    }
    
    // Initialize decoder weights (similar distributions)
    for (int i = 0; i < hidden3; ++i) {
        for (int j = 0; j < fLatentSize; ++j) {
            fDecoderWeights1[i][j] = dist4(gen);
        }
        fDecoderBias1[i] = 0.01;
    }
    
    for (int i = 0; i < hidden2; ++i) {
        for (int j = 0; j < hidden3; ++j) {
            fDecoderWeights2[i][j] = dist3(gen);
        }
        fDecoderBias2[i] = 0.01;
    }
    
    for (int i = 0; i < hidden1; ++i) {
        for (int j = 0; j < hidden2; ++j) {
            fDecoderWeights3[i][j] = dist2(gen);
        }
        fDecoderBias3[i] = 0.01;
    }
    
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden1; ++j) {
            fDecoderWeights4[i][j] = dist1(gen);
        }
        fDecoderBias4[i] = 0.01;
    }
    
    // Initialize momentum buffers for Adam optimizer
    fEncoderMomentum1.resize(hidden1, std::vector<double>(input_size, 0));
    fEncoderVelocity1.resize(hidden1, std::vector<double>(input_size, 0));
    fEncoderMomentum2.resize(hidden2, std::vector<double>(hidden1, 0));
    fEncoderVelocity2.resize(hidden2, std::vector<double>(hidden1, 0));
    fEncoderMomentum3.resize(hidden3, std::vector<double>(hidden2, 0));
    fEncoderVelocity3.resize(hidden3, std::vector<double>(hidden2, 0));
    fEncoderMomentum4.resize(fLatentSize, std::vector<double>(hidden3, 0));
    fEncoderVelocity4.resize(fLatentSize, std::vector<double>(hidden3, 0));
    
    fDecoderMomentum1.resize(hidden3, std::vector<double>(fLatentSize, 0));
    fDecoderVelocity1.resize(hidden3, std::vector<double>(fLatentSize, 0));
    fDecoderMomentum2.resize(hidden2, std::vector<double>(hidden3, 0));
    fDecoderVelocity2.resize(hidden2, std::vector<double>(hidden3, 0));
    fDecoderMomentum3.resize(hidden1, std::vector<double>(hidden2, 0));
    fDecoderVelocity3.resize(hidden1, std::vector<double>(hidden2, 0));
    fDecoderMomentum4.resize(input_size, std::vector<double>(hidden1, 0));
    fDecoderVelocity4.resize(input_size, std::vector<double>(hidden1, 0));
    
    fInitialized = true;
    fTrainingStep = 0;
}

double PhotonTriggerML::Autoencoder::GetReconstructionError(const std::vector<double>& features)
{
    if (!fInitialized || static_cast<int>(features.size()) != fInputSize) return 1e9;
    
    // Forward pass through encoder
    std::vector<double> hidden1(fEncoderWeights1.size());
    for (size_t i = 0; i < fEncoderWeights1.size(); ++i) {
        double sum = fEncoderBias1[i];
        for (size_t j = 0; j < features.size(); ++j) {
            sum += fEncoderWeights1[i][j] * features[j];
        }
        hidden1[i] = relu(sum);  // ReLU activation
    }
    
    std::vector<double> hidden2(fEncoderWeights2.size());
    for (size_t i = 0; i < fEncoderWeights2.size(); ++i) {
        double sum = fEncoderBias2[i];
        for (size_t j = 0; j < hidden1.size(); ++j) {
            sum += fEncoderWeights2[i][j] * hidden1[j];
        }
        hidden2[i] = relu(sum);
    }
    
    std::vector<double> hidden3(fEncoderWeights3.size());
    for (size_t i = 0; i < fEncoderWeights3.size(); ++i) {
        double sum = fEncoderBias3[i];
        for (size_t j = 0; j < hidden2.size(); ++j) {
            sum += fEncoderWeights3[i][j] * hidden2[j];
        }
        hidden3[i] = relu(sum);
    }
    
    std::vector<double> latent(fLatentSize);
    for (int i = 0; i < fLatentSize; ++i) {
        double sum = fEncoderBias4[i];
        for (size_t j = 0; j < hidden3.size(); ++j) {
            sum += fEncoderWeights4[i][j] * hidden3[j];
        }
        latent[i] = tanh(sum);  // tanh for latent space
    }
    
    // Forward pass through decoder
    std::vector<double> dhidden3(fDecoderWeights1.size());
    for (size_t i = 0; i < fDecoderWeights1.size(); ++i) {
        double sum = fDecoderBias1[i];
        for (int j = 0; j < fLatentSize; ++j) {
            sum += fDecoderWeights1[i][j] * latent[j];
        }
        dhidden3[i] = relu(sum);
    }
    
    std::vector<double> dhidden2(fDecoderWeights2.size());
    for (size_t i = 0; i < fDecoderWeights2.size(); ++i) {
        double sum = fDecoderBias2[i];
        for (size_t j = 0; j < dhidden3.size(); ++j) {
            sum += fDecoderWeights2[i][j] * dhidden3[j];
        }
        dhidden2[i] = relu(sum);
    }
    
    std::vector<double> dhidden1(fDecoderWeights3.size());
    for (size_t i = 0; i < fDecoderWeights3.size(); ++i) {
        double sum = fDecoderBias3[i];
        for (size_t j = 0; j < dhidden2.size(); ++j) {
            sum += fDecoderWeights3[i][j] * dhidden2[j];
        }
        dhidden1[i] = relu(sum);
    }
    
    std::vector<double> reconstructed(fInputSize);
    for (int i = 0; i < fInputSize; ++i) {
        double sum = fDecoderBias4[i];
        for (size_t j = 0; j < dhidden1.size(); ++j) {
            sum += fDecoderWeights4[i][j] * dhidden1[j];
        }
        reconstructed[i] = sigmoid(sum);  // sigmoid for [0,1] output
    }
    
    // Calculate reconstruction error (MSE + L1 for sparsity)
    double mse = 0;
    double l1 = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        double diff = features[i] - reconstructed[i];
        mse += diff * diff;
        l1 += abs(diff);
    }
    
    // Combined error metric
    return sqrt(mse / features.size()) + 0.1 * (l1 / features.size());
}

void PhotonTriggerML::Autoencoder::Train(const std::vector<std::vector<double>>& features,
                                         double learning_rate)
{
    if (!fInitialized || features.empty()) return;
    
    fTrainingStep++;
    
    // Adam optimizer parameters
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    
    // Adjust learning rate with warm-up and decay
    double adjusted_lr = learning_rate;
    if (fTrainingStep < 100) {
        adjusted_lr *= fTrainingStep / 100.0;  // Warm-up
    } else {
        adjusted_lr *= exp(-0.0001 * (fTrainingStep - 100));  // Exponential decay
    }
    
    for (const auto& input : features) {
        if (static_cast<int>(input.size()) != fInputSize) continue;
        
        // Forward pass - encoder
        std::vector<double> hidden1(fEncoderWeights1.size());
        std::vector<double> hidden1_raw(fEncoderWeights1.size());
        for (size_t i = 0; i < fEncoderWeights1.size(); ++i) {
            double sum = fEncoderBias1[i];
            for (size_t j = 0; j < input.size(); ++j) {
                sum += fEncoderWeights1[i][j] * input[j];
            }
            hidden1_raw[i] = sum;
            hidden1[i] = relu(sum);
        }
        
        std::vector<double> hidden2(fEncoderWeights2.size());
        std::vector<double> hidden2_raw(fEncoderWeights2.size());
        for (size_t i = 0; i < fEncoderWeights2.size(); ++i) {
            double sum = fEncoderBias2[i];
            for (size_t j = 0; j < hidden1.size(); ++j) {
                sum += fEncoderWeights2[i][j] * hidden1[j];
            }
            hidden2_raw[i] = sum;
            hidden2[i] = relu(sum);
        }
        
        std::vector<double> hidden3(fEncoderWeights3.size());
        std::vector<double> hidden3_raw(fEncoderWeights3.size());
        for (size_t i = 0; i < fEncoderWeights3.size(); ++i) {
            double sum = fEncoderBias3[i];
            for (size_t j = 0; j < hidden2.size(); ++j) {
                sum += fEncoderWeights3[i][j] * hidden2[j];
            }
            hidden3_raw[i] = sum;
            hidden3[i] = relu(sum);
        }
        
        std::vector<double> latent(fLatentSize);
        std::vector<double> latent_raw(fLatentSize);
        for (int i = 0; i < fLatentSize; ++i) {
            double sum = fEncoderBias4[i];
            for (size_t j = 0; j < hidden3.size(); ++j) {
                sum += fEncoderWeights4[i][j] * hidden3[j];
            }
            latent_raw[i] = sum;
            latent[i] = tanh(sum);
        }
        
        // Forward pass - decoder
        std::vector<double> dhidden3(fDecoderWeights1.size());
        std::vector<double> dhidden3_raw(fDecoderWeights1.size());
        for (size_t i = 0; i < fDecoderWeights1.size(); ++i) {
            double sum = fDecoderBias1[i];
            for (int j = 0; j < fLatentSize; ++j) {
                sum += fDecoderWeights1[i][j] * latent[j];
            }
            dhidden3_raw[i] = sum;
            dhidden3[i] = relu(sum);
        }
        
        std::vector<double> dhidden2(fDecoderWeights2.size());
        std::vector<double> dhidden2_raw(fDecoderWeights2.size());
        for (size_t i = 0; i < fDecoderWeights2.size(); ++i) {
            double sum = fDecoderBias2[i];
            for (size_t j = 0; j < dhidden3.size(); ++j) {
                sum += fDecoderWeights2[i][j] * dhidden3[j];
            }
            dhidden2_raw[i] = sum;
            dhidden2[i] = relu(sum);
        }
        
        std::vector<double> dhidden1(fDecoderWeights3.size());
        std::vector<double> dhidden1_raw(fDecoderWeights3.size());
        for (size_t i = 0; i < fDecoderWeights3.size(); ++i) {
            double sum = fDecoderBias3[i];
            for (size_t j = 0; j < dhidden2.size(); ++j) {
                sum += fDecoderWeights3[i][j] * dhidden2[j];
            }
            dhidden1_raw[i] = sum;
            dhidden1[i] = relu(sum);
        }
        
        std::vector<double> output(fInputSize);
        std::vector<double> output_raw(fInputSize);
        for (int i = 0; i < fInputSize; ++i) {
            double sum = fDecoderBias4[i];
            for (size_t j = 0; j < dhidden1.size(); ++j) {
                sum += fDecoderWeights4[i][j] * dhidden1[j];
            }
            output_raw[i] = sum;
            output[i] = sigmoid(sum);
        }
        
        // Backward pass - compute gradients
        // Output layer gradients
        std::vector<double> output_grad(fInputSize);
        for (int i = 0; i < fInputSize; ++i) {
            double error = output[i] - input[i];
            output_grad[i] = error * sigmoidDerivative(output[i]);
        }
        
        // Decoder layer 3 gradients
        std::vector<double> dhidden1_grad(dhidden1.size(), 0);
        for (size_t j = 0; j < dhidden1.size(); ++j) {
            for (int i = 0; i < fInputSize; ++i) {
                dhidden1_grad[j] += output_grad[i] * fDecoderWeights4[i][j];
            }
            dhidden1_grad[j] *= reluDerivative(dhidden1_raw[j]);
        }
        
        // Decoder layer 2 gradients
        std::vector<double> dhidden2_grad(dhidden2.size(), 0);
        for (size_t j = 0; j < dhidden2.size(); ++j) {
            for (size_t i = 0; i < dhidden1.size(); ++i) {
                dhidden2_grad[j] += dhidden1_grad[i] * fDecoderWeights3[i][j];
            }
            dhidden2_grad[j] *= reluDerivative(dhidden2_raw[j]);
        }
        
        // Decoder layer 1 gradients
        std::vector<double> dhidden3_grad(dhidden3.size(), 0);
        for (size_t j = 0; j < dhidden3.size(); ++j) {
            for (size_t i = 0; i < dhidden2.size(); ++i) {
                dhidden3_grad[j] += dhidden2_grad[i] * fDecoderWeights2[i][j];
            }
            dhidden3_grad[j] *= reluDerivative(dhidden3_raw[j]);
        }
        
        // Latent layer gradients
        std::vector<double> latent_grad(fLatentSize, 0);
        for (int j = 0; j < fLatentSize; ++j) {
            for (size_t i = 0; i < dhidden3.size(); ++i) {
                latent_grad[j] += dhidden3_grad[i] * fDecoderWeights1[i][j];
            }
            latent_grad[j] *= tanhDerivative(latent[j]);
        }
        
        // Encoder layer gradients (continuing backprop)
        std::vector<double> hidden3_grad(hidden3.size(), 0);
        for (size_t j = 0; j < hidden3.size(); ++j) {
            for (int i = 0; i < fLatentSize; ++i) {
                hidden3_grad[j] += latent_grad[i] * fEncoderWeights4[i][j];
            }
            hidden3_grad[j] *= reluDerivative(hidden3_raw[j]);
        }
        
        std::vector<double> hidden2_grad(hidden2.size(), 0);
        for (size_t j = 0; j < hidden2.size(); ++j) {
            for (size_t i = 0; i < hidden3.size(); ++i) {
                hidden2_grad[j] += hidden3_grad[i] * fEncoderWeights3[i][j];
            }
            hidden2_grad[j] *= reluDerivative(hidden2_raw[j]);
        }
        
        std::vector<double> hidden1_grad(hidden1.size(), 0);
        for (size_t j = 0; j < hidden1.size(); ++j) {
            for (size_t i = 0; i < hidden2.size(); ++i) {
                hidden1_grad[j] += hidden2_grad[i] * fEncoderWeights2[i][j];
            }
            hidden1_grad[j] *= reluDerivative(hidden1_raw[j]);
        }
        
        // Update weights using Adam optimizer
        // Decoder weights
        updateWeightsAdam(fDecoderWeights4, output_grad, dhidden1, 
                         fDecoderMomentum4, fDecoderVelocity4, 
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fDecoderBias4, output_grad, fDecoderBiasMomentum4, 
                      fDecoderBiasVelocity4, adjusted_lr, beta1, beta2, epsilon);
        
        updateWeightsAdam(fDecoderWeights3, dhidden1_grad, dhidden2,
                         fDecoderMomentum3, fDecoderVelocity3,
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fDecoderBias3, dhidden1_grad, fDecoderBiasMomentum3,
                      fDecoderBiasVelocity3, adjusted_lr, beta1, beta2, epsilon);
        
        updateWeightsAdam(fDecoderWeights2, dhidden2_grad, dhidden3,
                         fDecoderMomentum2, fDecoderVelocity2,
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fDecoderBias2, dhidden2_grad, fDecoderBiasMomentum2,
                      fDecoderBiasVelocity2, adjusted_lr, beta1, beta2, epsilon);
        
        updateWeightsAdam(fDecoderWeights1, dhidden3_grad, latent,
                         fDecoderMomentum1, fDecoderVelocity1,
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fDecoderBias1, dhidden3_grad, fDecoderBiasMomentum1,
                      fDecoderBiasVelocity1, adjusted_lr, beta1, beta2, epsilon);
        
        // Encoder weights
        updateWeightsAdam(fEncoderWeights4, latent_grad, hidden3,
                         fEncoderMomentum4, fEncoderVelocity4,
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fEncoderBias4, latent_grad, fEncoderBiasMomentum4,
                      fEncoderBiasVelocity4, adjusted_lr, beta1, beta2, epsilon);
        
        updateWeightsAdam(fEncoderWeights3, hidden3_grad, hidden2,
                         fEncoderMomentum3, fEncoderVelocity3,
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fEncoderBias3, hidden3_grad, fEncoderBiasMomentum3,
                      fEncoderBiasVelocity3, adjusted_lr, beta1, beta2, epsilon);
        
        updateWeightsAdam(fEncoderWeights2, hidden2_grad, hidden1,
                         fEncoderMomentum2, fEncoderVelocity2,
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fEncoderBias2, hidden2_grad, fEncoderBiasMomentum2,
                      fEncoderBiasVelocity2, adjusted_lr, beta1, beta2, epsilon);
        
        updateWeightsAdam(fEncoderWeights1, hidden1_grad, input,
                         fEncoderMomentum1, fEncoderVelocity1,
                         adjusted_lr, beta1, beta2, epsilon);
        updateBiasAdam(fEncoderBias1, hidden1_grad, fEncoderBiasMomentum1,
                      fEncoderBiasVelocity1, adjusted_lr, beta1, beta2, epsilon);
    }
}

void PhotonTriggerML::Autoencoder::UpdateThreshold(const std::vector<std::vector<double>>& hadron_features,
                                                   double percentile)
{
    if (hadron_features.empty()) return;
    
    // Calculate reconstruction errors for hadrons
    std::vector<double> errors;
    for (const auto& features : hadron_features) {
        errors.push_back(GetReconstructionError(features));
    }
    
    // Sort errors
    std::sort(errors.begin(), errors.end());
    
    // Set threshold at specified percentile (default was 99th, now adjustable)
    size_t idx = static_cast<size_t>(percentile * errors.size());
    if (idx >= errors.size()) idx = errors.size() - 1;
    fAnomalyThreshold = errors[idx];
    
    // Calculate statistics for monitoring
    double mean = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double sq_sum = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / errors.size() - mean * mean);
    
    cout << "  Anomaly threshold updated:" << endl;
    cout << "    Percentile: " << (percentile * 100) << "%" << endl;
    cout << "    Threshold: " << fAnomalyThreshold << endl;
    cout << "    Mean error: " << mean << " ± " << stdev << endl;
    cout << "    Min/Max: " << errors.front() << " / " << errors.back() << endl;
}

// Helper activation functions
double PhotonTriggerML::Autoencoder::relu(double x) {
    return x > 0 ? x : 0.01 * x;  // Leaky ReLU
}

double PhotonTriggerML::Autoencoder::reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.01;
}

double PhotonTriggerML::Autoencoder::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double PhotonTriggerML::Autoencoder::sigmoidDerivative(double y) {
    return y * (1.0 - y);
}

double PhotonTriggerML::Autoencoder::tanhDerivative(double y) {
    return 1.0 - y * y;
}

// Adam optimizer update functions
void PhotonTriggerML::Autoencoder::updateWeightsAdam(
    std::vector<std::vector<double>>& weights,
    const std::vector<double>& gradients,
    const std::vector<double>& inputs,
    std::vector<std::vector<double>>& momentum,
    std::vector<std::vector<double>>& velocity,
    double lr, double beta1, double beta2, double epsilon)
{
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            double grad = gradients[i] * inputs[j];
            
            // Add L2 regularization
            grad += 0.0001 * weights[i][j];
            
            // Adam updates
            momentum[i][j] = beta1 * momentum[i][j] + (1 - beta1) * grad;
            velocity[i][j] = beta2 * velocity[i][j] + (1 - beta2) * grad * grad;
            
            // Bias correction
            double m_hat = momentum[i][j] / (1 - pow(beta1, fTrainingStep));
            double v_hat = velocity[i][j] / (1 - pow(beta2, fTrainingStep));
            
            weights[i][j] -= lr * m_hat / (sqrt(v_hat) + epsilon);
            
            // Weight clipping to prevent explosion
            if (weights[i][j] > 5.0) weights[i][j] = 5.0;
            if (weights[i][j] < -5.0) weights[i][j] = -5.0;
        }
    }
}

void PhotonTriggerML::Autoencoder::updateBiasAdam(
    std::vector<double>& bias,
    const std::vector<double>& gradients,
    std::vector<double>& momentum,
    std::vector<double>& velocity,
    double lr, double beta1, double beta2, double epsilon)
{
    if (momentum.size() != bias.size()) {
        momentum.resize(bias.size(), 0);
        velocity.resize(bias.size(), 0);
    }
    
    for (size_t i = 0; i < bias.size(); ++i) {
        double grad = gradients[i];
        
        momentum[i] = beta1 * momentum[i] + (1 - beta1) * grad;
        velocity[i] = beta2 * velocity[i] + (1 - beta2) * grad * grad;
        
        double m_hat = momentum[i] / (1 - pow(beta1, fTrainingStep));
        double v_hat = velocity[i] / (1 - pow(beta2, fTrainingStep));
        
        bias[i] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

// ============================================================================
// PhotonTriggerML Implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fAutoencoder(std::make_unique<Autoencoder>()),
    fFeatureCount(0),
    fEventCount(0),
    fStationCount(0),
    fPhotonLikeCount(0),
    fHadronLikeCount(0),
    fEnergy(0),
    fCoreX(0),
    fCoreY(0),
    fPrimaryId(0),
    fPrimaryType("Unknown"),
    fStationId(0),
    fDistance(0),
    fReconstructionError(0),
    fIsAnomaly(false),
    fIsActualPhoton(false),
    fTruePositives(0),
    fFalsePositives(0),
    fTrueNegatives(0),
    fFalseNegatives(0),
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_improved.root"),
    fLogFileName("photon_trigger_improved.log"),
    fTargetPercentile(0.85),  // Start with 85th percentile
    fCurrentThreshold(0.5),
    fDynamicThresholdEnabled(true),
    fValidationFraction(0.2),
    fOutputFile(nullptr),
    fMLTree(nullptr),
    hReconstructionError(nullptr),
    hErrorPhotons(nullptr),
    hErrorHadrons(nullptr),
    hConfusionMatrix(nullptr),
    hThresholdEvolution(nullptr),
    hFeatureImportance(nullptr)
{
    fInstance = this;
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Constructor (IMPROVED)" << endl;
    cout << "Using deep autoencoder with adaptive threshold" << endl;
    cout << "==========================================" << endl;
}

PhotonTriggerML::~PhotonTriggerML()
{
    cout << "PhotonTriggerML Destructor called" << endl;
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization (IMPROVED)");
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Initialization (Improved)" << endl;
    cout << "==========================================" << endl;
    
    // Read configuration
    CentralConfig* cc = CentralConfig::GetInstance();
    Branch topBranch = cc->GetTopBranch("PhotonTriggerML");
    
    if (topBranch.GetChild("EnergyMin")) {
        topBranch.GetChild("EnergyMin").GetData(fEnergyMin);
    }
    if (topBranch.GetChild("EnergyMax")) {
        topBranch.GetChild("EnergyMax").GetData(fEnergyMax);
    }
    if (topBranch.GetChild("OutputFile")) {
        topBranch.GetChild("OutputFile").GetData(fOutputFileName);
    }
    if (topBranch.GetChild("PhotonThreshold")) {
        double threshold;
        topBranch.GetChild("PhotonThreshold").GetData(threshold);
        fTargetPercentile = 1.0 - threshold;  // Convert to percentile
    }
    if (topBranch.GetChild("DynamicThreshold")) {
        topBranch.GetChild("DynamicThreshold").GetData(fDynamicThresholdEnabled);
    }
    
    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    
    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML IMPROVED Detection Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "Target Percentile: " << fTargetPercentile << endl;
    fLogFile << "==========================================" << endl << endl;
    
    // Initialize autoencoder with expanded feature set (25 features)
    cout << "Initializing Deep Autoencoder for Anomaly Detection..." << endl;
    fAutoencoder->Initialize(25);  // Using 25 comprehensive features
    
    // Initialize feature normalization
    fFeatureMeans.resize(25, 0.0);
    fFeatureStds.resize(25, 1.0);
    fFeatureCount = 0;
    
    // Create output file
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
    // Create tree with expanded branches
    fMLTree = new TTree("MLTree", "PhotonTriggerML Improved Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("reconstructionError", &fReconstructionError, "reconstructionError/D");
    fMLTree->Branch("isAnomaly", &fIsAnomaly, "isAnomaly/O");
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");
    fMLTree->Branch("primaryType", &fPrimaryType);
    fMLTree->Branch("anomalyThreshold", &fCurrentThreshold, "anomalyThreshold/D");
    
    // Create histograms
    hReconstructionError = new TH1D("hReconstructionError", "Reconstruction Error;Error;Count", 200, 0, 4);
    hErrorPhotons = new TH1D("hErrorPhotons", "Reconstruction Error (Photons);Error;Count", 200, 0, 4);
    hErrorPhotons->SetLineColor(kBlue);
    hErrorHadrons = new TH1D("hErrorHadrons", "Reconstruction Error (Hadrons);Error;Count", 200, 0, 4);
    hErrorHadrons->SetLineColor(kRed);
    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 
                                2, -0.5, 1.5, 2, -0.5, 1.5);
    
    // Additional monitoring histograms
    hThresholdEvolution = new TH1D("hThresholdEvolution", "Threshold Evolution;Event;Threshold", 
                                   1000, 0, 10000);
    hFeatureImportance = new TH1D("hFeatureImportance", "Feature Importance;Feature;Weight", 
                                  25, 0, 25);
    
    // Register signal handler
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    
    cout << "Initialization complete!" << endl;
    cout << "  Target anomaly percentile: " << (fTargetPercentile * 100) << "%" << endl;
    cout << "  Dynamic threshold: " << (fDynamicThresholdEnabled ? "Enabled" : "Disabled") << endl;
    cout << "==========================================" << endl << endl;
    
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    
    // Clear previous ML results
    ClearMLResults();
    
    // Print header periodically
    if (fEventCount % 50 == 1) {
        cout << "\n┌────────┬──────────┬─────────┬─────────┬──────────┬─────────┬─────────┬──────────┐" << endl;
        cout << "│ Event  │ Stations │ Anomaly%│ Accuracy│ Precision│ Recall  │ F1-Score│ Threshold│" << endl;
        cout << "├────────┼──────────┼─────────┼─────────┼──────────┼─────────┼─────────┼──────────┤" << endl;
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
            case 2212: fPrimaryType = "proton"; break;
            case 1000026056: fPrimaryType = "iron"; break;
            default: fPrimaryType = "hadron";
        }
        
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const utl::CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            utl::Point core = shower.GetSimCore(0);
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
    
    // Sophisticated training strategy
    if (fEventCount % 10 == 0 && !fHadronFeatures.empty()) {
        // DIAGNOSTIC: Print training info
        if (fEventCount % 50 == 0) {
            cout << "\n=== TRAINING DIAGNOSTIC at Event " << fEventCount << " ===" << endl;
            cout << "Hadron Features Buffer: " << fHadronFeatures.size() << " samples" << endl;
            cout << "Photon Features Buffer: " << fPhotonFeatures.size() << " samples" << endl;
        }
        
        // Prepare training batch
        int batch_size = min(256, (int)fHadronFeatures.size());
        std::vector<std::vector<double>> batch;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, fHadronFeatures.size() - 1);
        
        for (int i = 0; i < batch_size; ++i) {
            batch.push_back(fHadronFeatures[dis(gen)]);
        }
        
        // Multiple training iterations
        double base_lr = 0.001;
        double learning_rate = base_lr * exp(-fEventCount / 2000.0);  // Slower decay
        
        // DIAGNOSTIC: Check reconstruction errors before training
        double error_before = 0;
        for (int i = 0; i < min(10, (int)batch.size()); i++) {
            error_before += fAutoencoder->GetReconstructionError(batch[i]);
        }
        error_before /= min(10, (int)batch.size());
        
        for (int iter = 0; iter < 3; ++iter) {
            fAutoencoder->Train(batch, learning_rate);
        }
        
        // DIAGNOSTIC: Check reconstruction errors after training
        double error_after = 0;
        for (int i = 0; i < min(10, (int)batch.size()); i++) {
            error_after += fAutoencoder->GetReconstructionError(batch[i]);
        }
        error_after /= min(10, (int)batch.size());
        
        if (fEventCount % 50 == 0) {
            cout << "Training Effect: " << error_before << " -> " << error_after 
                 << " (LR: " << learning_rate << ")" << endl;
        }
        
        // Update threshold dynamically
        if (fDynamicThresholdEnabled && fEventCount % 50 == 0) {
            double old_threshold = fCurrentThreshold;
            
            // Adjust percentile based on performance
            double recall = (fTruePositives + fFalseNegatives > 0) ? 
                          (double)fTruePositives / (fTruePositives + fFalseNegatives) : 0;
            
            cout << "Current Recall: " << (recall * 100) << "%" << endl;
            cout << "Target Percentile: " << (fTargetPercentile * 100) << "%" << endl;
            
            // More aggressive threshold adjustment
            if (recall < 0.05 && fTargetPercentile > 0.50) {  // If recall < 5%
                fTargetPercentile -= 0.05;  // Bigger adjustment
                cout << "  LOW RECALL: Adjusting percentile down to " << (fTargetPercentile * 100) << "%" << endl;
            } else if (recall < 0.10 && fTargetPercentile > 0.60) {
                fTargetPercentile -= 0.02;
                cout << "  Adjusting percentile down to " << (fTargetPercentile * 100) << "%" << endl;
            } else if (recall > 0.30 && fTargetPercentile < 0.95) {
                fTargetPercentile += 0.01;
                cout << "  Adjusting percentile up to " << (fTargetPercentile * 100) << "%" << endl;
            }
            
            // Calculate error distribution for threshold update
            cout << "Calculating threshold from " << fHadronFeatures.size() << " hadron samples..." << endl;
            std::vector<double> sample_errors;
            for (int i = 0; i < min(1000, (int)fHadronFeatures.size()); i += 10) {
                sample_errors.push_back(fAutoencoder->GetReconstructionError(fHadronFeatures[i]));
            }
            
            if (!sample_errors.empty()) {
                std::sort(sample_errors.begin(), sample_errors.end());
                cout << "Error distribution (hadrons):" << endl;
                cout << "  Min: " << sample_errors.front() << endl;
                cout << "  25%: " << sample_errors[sample_errors.size()/4] << endl;
                cout << "  50%: " << sample_errors[sample_errors.size()/2] << endl;
                cout << "  75%: " << sample_errors[3*sample_errors.size()/4] << endl;
                cout << "  Max: " << sample_errors.back() << endl;
                
                // Also check photon errors if available
                if (!fPhotonFeatures.empty()) {
                    std::vector<double> photon_errors;
                    for (int i = 0; i < min(100, (int)fPhotonFeatures.size()); i++) {
                        photon_errors.push_back(fAutoencoder->GetReconstructionError(fPhotonFeatures[i]));
                    }
                    if (!photon_errors.empty()) {
                        std::sort(photon_errors.begin(), photon_errors.end());
                        cout << "Error distribution (photons):" << endl;
                        cout << "  Min: " << photon_errors.front() << endl;
                        cout << "  50%: " << photon_errors[photon_errors.size()/2] << endl;
                        cout << "  Max: " << photon_errors.back() << endl;
                    }
                }
            }
            
            fAutoencoder->UpdateThreshold(fHadronFeatures, fTargetPercentile);
            fCurrentThreshold = fAutoencoder->GetThreshold();
            
            cout << "THRESHOLD UPDATE: " << old_threshold << " -> " << fCurrentThreshold << endl;
            
            hThresholdEvolution->SetBinContent(fEventCount/10, fCurrentThreshold);
        }
    }
    
    // Display metrics
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
        
        // Extract comprehensive features (25 features)
        std::vector<double> features = ExtractComprehensiveFeatures(trace_data);
        
        // Update normalization statistics
        UpdateFeatureStatistics(features);
        
        // Normalize features
        std::vector<double> normalized_features = NormalizeFeatures(features);
        
        // Get reconstruction error from autoencoder
        fReconstructionError = fAutoencoder->GetReconstructionError(normalized_features);
        fCurrentThreshold = fAutoencoder->GetThreshold();
        
        // Determine if it's an anomaly (potential photon)
        fIsAnomaly = (fReconstructionError > fCurrentThreshold);
        
        // DIAGNOSTIC: Track reconstruction errors by particle type
        static int photon_samples = 0;
        static int hadron_samples = 0;
        static double photon_error_sum = 0;
        static double hadron_error_sum = 0;
        static double photon_error_min = 1e9;
        static double photon_error_max = 0;
        static double hadron_error_min = 1e9;
        static double hadron_error_max = 0;
        
        if (fIsActualPhoton) {
            photon_samples++;
            photon_error_sum += fReconstructionError;
            photon_error_min = min(photon_error_min, fReconstructionError);
            photon_error_max = max(photon_error_max, fReconstructionError);
        } else {
            hadron_samples++;
            hadron_error_sum += fReconstructionError;
            hadron_error_min = min(hadron_error_min, fReconstructionError);
            hadron_error_max = max(hadron_error_max, fReconstructionError);
        }
        
        // Print detailed diagnostics every 1000 stations
        if (fStationCount % 1000 == 0 && fStationCount > 0) {
            cout << "\n=== DIAGNOSTIC at Station " << fStationCount << " ===" << endl;
            cout << "Current Threshold: " << fCurrentThreshold << endl;
            
            if (photon_samples > 0) {
                cout << "PHOTON Stats (" << photon_samples << " samples):" << endl;
                cout << "  Avg Error: " << photon_error_sum / photon_samples << endl;
                cout << "  Min/Max: " << photon_error_min << " / " << photon_error_max << endl;
                cout << "  Above Threshold: " << (photon_error_sum / photon_samples > fCurrentThreshold ? "YES" : "NO") << endl;
            }
            
            if (hadron_samples > 0) {
                cout << "HADRON Stats (" << hadron_samples << " samples):" << endl;
                cout << "  Avg Error: " << hadron_error_sum / hadron_samples << endl;
                cout << "  Min/Max: " << hadron_error_min << " / " << hadron_error_max << endl;
            }
            
            // Print feature statistics for current sample
            cout << "\nCurrent Sample (" << fPrimaryType << "):" << endl;
            cout << "  Reconstruction Error: " << fReconstructionError << endl;
            cout << "  Is Anomaly: " << (fIsAnomaly ? "YES" : "NO") << endl;
            cout << "  Key Features (first 5):" << endl;
            for (int i = 0; i < min(5, (int)features.size()); i++) {
                cout << "    F" << i << ": " << features[i] 
                     << " (norm: " << normalized_features[i] << ")" << endl;
            }
            
            // Check if autoencoder weights are updating
            static double last_threshold = 0;
            if (abs(fCurrentThreshold - last_threshold) > 0.001) {
                cout << "  THRESHOLD CHANGED: " << last_threshold << " -> " << fCurrentThreshold << endl;
                last_threshold = fCurrentThreshold;
            }
        }
        
        // Store ML result for PMTTraceModule compatibility
        MLResult mlResult;
        mlResult.photonScore = fReconstructionError;
        mlResult.identifiedAsPhoton = fIsAnomaly;
        mlResult.isActualPhoton = fIsActualPhoton;
        mlResult.primaryType = fPrimaryType;
        mlResult.confidence = fIsAnomaly ? 
            min(1.0, fReconstructionError / (fCurrentThreshold + 0.001)) : 
            max(0.0, 1.0 - fReconstructionError / (fCurrentThreshold + 0.001));
        mlResult.features = ExtractCompatibilityFeatures(trace_data);
        mlResult.vemCharge = mlResult.features.total_charge;
        
        // Store in static map
        fMLResultsMap[fStationId] = mlResult;
        
        // Store features for training
        if (!fIsActualPhoton) {
            fHadronFeatures.push_back(normalized_features);
            if (fHadronFeatures.size() > 20000) {  // Larger buffer
                // Keep only recent samples
                fHadronFeatures.erase(fHadronFeatures.begin(), 
                                     fHadronFeatures.begin() + 10000);
            }
        } else {
            // Store photon features separately for validation
            fPhotonFeatures.push_back(normalized_features);
            if (fPhotonFeatures.size() > 5000) {
                fPhotonFeatures.erase(fPhotonFeatures.begin(),
                                     fPhotonFeatures.begin() + 2500);
            }
        }
        
        fStationCount++;
        
        // Update counters
        if (fIsAnomaly) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++;
            else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++;
            else fFalseNegatives++;
        }
        
        // Fill histograms
        hReconstructionError->Fill(fReconstructionError);
        
        if (fIsActualPhoton) {
            hErrorPhotons->Fill(fReconstructionError);
        } else {
            hErrorHadrons->Fill(fReconstructionError);
        }
        
        // Fill tree
        fMLTree->Fill();
    }
}

std::vector<double> PhotonTriggerML::ExtractComprehensiveFeatures(const std::vector<double>& trace)
{
    std::vector<double> features;
    const double baseline = 50.0;
    
    // Safety check for trace size
    if (trace.empty() || trace.size() < 100) {
        // Return default features if trace is too small
        features.resize(25, 0.0);
        return features;
    }
    
    // Find peak and basic statistics
    int peak_bin = 0;
    double peak_value = 0;
    double total_signal = 0;
    double total_squared = 0;
    int first_bin = -1, last_bin = -1;
    
    for (size_t i = 0; i < trace.size(); i++) {
        double val = trace[i] - baseline;
        if (val > peak_value) {
            peak_value = val;
            peak_bin = i;
        }
        if (val > 0) {
            total_signal += val;
            total_squared += val * val;
            if (first_bin < 0) first_bin = i;
            last_bin = i;
        }
    }
    
    // If no signal found, return default features
    if (total_signal <= 0 || peak_value <= 0) {
        features.resize(25, 0.0);
        return features;
    }
    
    // 1-3. Peak characteristics
    features.push_back(peak_value / 1000.0);  // Normalized peak
    features.push_back(total_signal / 10000.0);  // Normalized total charge
    features.push_back(static_cast<double>(peak_bin) / static_cast<double>(trace.size()));  // Normalized peak position
    
    // 4-6. Time characteristics
    int rise_10 = peak_bin, rise_90 = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (i < (int)trace.size()) {
            double val = trace[i] - baseline;
            if (val < 0.9 * peak_value && rise_90 == peak_bin) rise_90 = i;
            if (val < 0.1 * peak_value) {
                rise_10 = i;
                break;
            }
        }
    }
    
    int fall_90 = peak_bin, fall_10 = peak_bin;
    for (size_t i = peak_bin; i < trace.size(); i++) {
        double val = trace[i] - baseline;
        if (val < 0.9 * peak_value && fall_90 == peak_bin) fall_90 = i;
        if (val < 0.1 * peak_value) {
            fall_10 = i;
            break;
        }
    }
    
    double rise_time = abs(rise_90 - rise_10) / 100.0;
    double fall_time = abs(fall_10 - fall_90) / 100.0;
    double rise_fall_sum = rise_time + fall_time + 0.001;  // Prevent division by zero
    double asymmetry = (fall_time - rise_time) / rise_fall_sum;
    
    features.push_back(rise_time);  // Rise time
    features.push_back(fall_time);  // Fall time
    features.push_back(asymmetry);  // Asymmetry
    
    // 7-9. FWHM and signal duration
    int half_rise = rise_10, half_fall = fall_10;
    for (int i = rise_10; i <= peak_bin && i < (int)trace.size(); i++) {
        if (trace[i] - baseline >= 0.5 * peak_value) {
            half_rise = i;
            break;
        }
    }
    for (int i = peak_bin; i < (int)trace.size(); i++) {
        if (trace[i] - baseline <= 0.5 * peak_value) {
            half_fall = i;
            break;
        }
    }
    
    features.push_back((half_fall - half_rise) / 100.0);  // FWHM
    features.push_back((last_bin > first_bin) ? (last_bin - first_bin) / 1000.0 : 0.0);  // Signal duration
    double rms = (total_signal > 0) ? sqrt(total_squared / total_signal) / 100.0 : 0.0;
    features.push_back(rms);  // RMS
    
    // 10-14. Charge distribution in time windows
    double early_charge = 0, peak_charge = 0, late_charge = 0;
    double very_early = 0, very_late = 0;
    
    int window_size = trace.size() / 8;  // Dynamic window sizing
    
    // Very early window
    for (int i = 0; i < min(window_size, (int)trace.size()); i++) {
        double val = trace[i] - baseline;
        if (val > 0) very_early += val;
    }
    
    // Early window
    for (int i = window_size; i < min(3 * window_size, (int)trace.size()); i++) {
        double val = trace[i] - baseline;
        if (val > 0) early_charge += val;
    }
    
    // Peak window
    int peak_start = max(0, peak_bin - 100);
    int peak_end = min((int)trace.size(), peak_bin + 100);
    for (int i = peak_start; i < peak_end; i++) {
        double val = trace[i] - baseline;
        if (val > 0) peak_charge += val;
    }
    
    // Late window
    for (int i = 5 * window_size; i < min(7 * window_size, (int)trace.size()); i++) {
        double val = trace[i] - baseline;
        if (val > 0) late_charge += val;
    }
    
    // Very late window
    for (int i = max(0, (int)trace.size() - window_size); i < (int)trace.size(); i++) {
        double val = trace[i] - baseline;
        if (val > 0) very_late += val;
    }
    
    // Add small value to total_signal to prevent division by zero
    double safe_total = total_signal + 0.001;
    features.push_back(very_early / safe_total);
    features.push_back(early_charge / safe_total);
    features.push_back(peak_charge / safe_total);
    features.push_back(late_charge / safe_total);
    features.push_back(very_late / safe_total);
    
    // 15-17. Statistical moments
    double mean_time = 0;
    for (size_t i = 0; i < trace.size(); i++) {
        double val = max(0.0, trace[i] - baseline);
        mean_time += i * val;
    }
    mean_time = (total_signal > 0) ? mean_time / total_signal : static_cast<double>(peak_bin);
    
    double variance = 0, skewness = 0, kurtosis = 0;
    for (size_t i = 0; i < trace.size(); i++) {
        double val = max(0.0, trace[i] - baseline);
        double diff = i - mean_time;
        double weight = (total_signal > 0) ? val / total_signal : 0.0;
        variance += diff * diff * weight;
        skewness += diff * diff * diff * weight;
        kurtosis += diff * diff * diff * diff * weight;
    }
    
    double std_dev = sqrt(variance + 0.001);  // Add small value to prevent zero
    features.push_back(variance / 10000.0);
    features.push_back((std_dev > 0.001) ? skewness / (std_dev * std_dev * std_dev) : 0.0);
    features.push_back((variance > 0.001) ? (kurtosis / (variance * variance)) - 3.0 : 0.0);
    
    // 18-20. Signal quality metrics
    double smoothness = 0;
    int smooth_count = 0;
    for (int i = max(1, peak_bin - 100); i < min(2047, peak_bin + 100); i++) {
        double second_deriv = trace[i+1] - 2*trace[i] + trace[i-1];
        smoothness += abs(second_deriv);
        smooth_count++;
    }
    features.push_back(smoothness / (smooth_count * 100.0 + 1));
    
    // Peak sharpness
    double peak_sharpness = peak_value / (half_fall - half_rise + 1);
    features.push_back(peak_sharpness / 100.0);
    
    // Signal-to-noise ratio estimate
    double noise_estimate = 0;
    for (int i = 0; i < 100; i++) {
        noise_estimate += abs(trace[i] - baseline);
    }
    noise_estimate /= 100.0;
    features.push_back(peak_value / (noise_estimate + 1) / 100.0);
    
    // 21-25. Frequency domain features (simplified)
    // Calculate dominant frequency components
    double low_freq = 0, mid_freq = 0, high_freq = 0;
    
    // Low frequency: changes over >100 bins
    for (int i = 0; i < 19; i++) {  // Changed from 20 to 19 to prevent overflow
        int start1 = i * 100;
        int end1 = min((i + 1) * 100, (int)trace.size());
        int start2 = (i + 1) * 100;
        int end2 = min((i + 2) * 100, (int)trace.size());
        
        if (start2 >= (int)trace.size()) break;
        
        double sum1 = 0, sum2 = 0;
        for (int j = start1; j < end1; j++) {
            sum1 += trace[j] - baseline;
        }
        for (int j = start2; j < end2; j++) {
            sum2 += trace[j] - baseline;
        }
        low_freq += abs(sum2 - sum1);
    }
    
    // Mid frequency: changes over 10-50 bins
    int mid_start = max(0, peak_bin - 200);
    int mid_end = min((int)trace.size() - 50, peak_bin + 200);
    
    for (int i = mid_start; i < mid_end; i += 10) {
        if (i + 50 >= (int)trace.size()) break;
        
        double sum1 = 0, sum2 = 0;
        for (int j = 0; j < 25 && (i + j) < (int)trace.size(); j++) {
            sum1 += trace[i + j] - baseline;
        }
        for (int j = 25; j < 50 && (i + j) < (int)trace.size(); j++) {
            sum2 += trace[i + j] - baseline;
        }
        mid_freq += abs(sum2 - sum1);
    }
    
    // High frequency: bin-to-bin changes
    int high_start = max(0, peak_bin - 100);
    int high_end = min((int)trace.size() - 1, peak_bin + 100);
    
    for (int i = high_start; i < high_end; i++) {
        high_freq += abs(trace[i + 1] - trace[i]);
    }
    
    features.push_back(low_freq / 10000.0);
    features.push_back(mid_freq / 1000.0);
    features.push_back(high_freq / 1000.0);
    
    // Ratio of peak to total (protect against division by zero)
    features.push_back((total_signal > 0) ? peak_value / total_signal : 1.0);
    
    // Number of peaks (simplified)
    int n_peaks = 0;
    for (int i = 10; i < (int)trace.size() - 10; i++) {
        if (trace[i] > trace[i-1] && trace[i] > trace[i+1] && 
            trace[i] - baseline > 0.1 * peak_value) {
            n_peaks++;
        }
    }
    features.push_back(n_peaks / 10.0);
    
    return features;
}

void PhotonTriggerML::UpdateFeatureStatistics(const std::vector<double>& features)
{
    if (features.size() != fFeatureMeans.size()) return;
    
    fFeatureCount++;
    
    // Update running mean and variance (Welford's algorithm)
    for (size_t i = 0; i < features.size(); i++) {
        double delta = features[i] - fFeatureMeans[i];
        fFeatureMeans[i] += delta / fFeatureCount;
        double delta2 = features[i] - fFeatureMeans[i];
        fFeatureStds[i] += delta * delta2;
    }
    
    // Update standard deviations periodically
    if (fFeatureCount % 100 == 0) {
        for (size_t i = 0; i < fFeatureStds.size(); i++) {
            fFeatureStds[i] = sqrt(fFeatureStds[i] / (fFeatureCount - 1));
            if (fFeatureStds[i] < 0.001) fFeatureStds[i] = 1.0;  // Avoid division by zero
        }
    }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const std::vector<double>& features)
{
    std::vector<double> normalized(features.size());
    
    if (fFeatureCount < 100) {
        // Not enough statistics, use simple normalization
        for (size_t i = 0; i < features.size(); i++) {
            normalized[i] = max(0.0, min(1.0, features[i]));
        }
    } else {
        // Z-score normalization then sigmoid to [0,1]
        for (size_t i = 0; i < features.size(); i++) {
            double z = (features[i] - fFeatureMeans[i]) / (fFeatureStds[i] + 0.001);
            normalized[i] = 1.0 / (1.0 + exp(-z/2.0));  // Sigmoid with reduced slope
        }
    }
    
    return normalized;
}

PhotonTriggerML::MLResult::Features PhotonTriggerML::ExtractCompatibilityFeatures(const std::vector<double>& trace)
{
    MLResult::Features features;
    const double ADC_PER_VEM = 180.0;
    
    // Find peak
    int peak_bin = 0;
    double peak_value = 0;
    double total_signal = 0;
    
    for (size_t i = 0; i < trace.size(); i++) {
        double val = trace[i] - 50;
        if (val > peak_value) {
            peak_value = val;
            peak_bin = i;
        }
        if (val > 0) total_signal += val;
    }
    
    features.total_charge = total_signal / ADC_PER_VEM;
    features.peak_charge_ratio = peak_value / (total_signal + 1);
    
    // Rise time
    int rise_start = peak_bin, rise_end = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (trace[i] - 50 < 0.1 * peak_value) {
            rise_start = i;
            break;
        }
        if (trace[i] - 50 < 0.9 * peak_value) {
            rise_end = i;
        }
    }
    features.risetime_10_90 = (rise_end - rise_start) * 25.0;
    
    // Pulse width (FWHM)
    int half_rise = rise_start, half_fall = peak_bin;
    for (int i = rise_start; i <= peak_bin; i++) {
        if (trace[i] - 50 >= 0.5 * peak_value) {
            half_rise = i;
            break;
        }
    }
    for (size_t i = peak_bin; i < trace.size(); i++) {
        if (trace[i] - 50 <= 0.5 * peak_value) {
            half_fall = i;
            break;
        }
    }
    features.pulse_width = (half_fall - half_rise) * 25.0;
    
    // Asymmetry
    int fall_start = peak_bin, fall_end = peak_bin;
    for (size_t i = peak_bin; i < trace.size(); i++) {
        if (trace[i] - 50 < 0.9 * peak_value) {
            fall_start = i;
        }
        if (trace[i] - 50 < 0.1 * peak_value) {
            fall_end = i;
            break;
        }
    }
    double rise = rise_end - rise_start;
    double fall = fall_end - fall_start;
    features.asymmetry = (fall - rise) / (fall + rise + 1);
    
    // Kurtosis
    double mean_time = 0;
    for (size_t i = 0; i < trace.size(); i++) {
        mean_time += i * max(0.0, trace[i] - 50);
    }
    mean_time /= (total_signal + 1);
    
    double variance = 0;
    double kurtosis = 0;
    for (size_t i = 0; i < trace.size(); i++) {
        double diff = i - mean_time;
        double weight = max(0.0, trace[i] - 50) / (total_signal + 1);
        variance += diff * diff * weight;
        kurtosis += diff * diff * diff * diff * weight;
    }
    features.kurtosis = (variance > 0) ? kurtosis / (variance * variance) - 3.0 : 0;
    
    return features;
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
    
    double anomaly_rate = (fPhotonLikeCount + fHadronLikeCount > 0) ?
                         100.0 * fPhotonLikeCount / (fPhotonLikeCount + fHadronLikeCount) : 0;
    
    // Display in table format
    cout << "│ " << setw(6) << fEventCount 
         << " │ " << setw(8) << fStationCount
         << " │ " << fixed << setprecision(1) << setw(7) << anomaly_rate << "%"
         << " │ " << setw(7) << accuracy << "%"
         << " │ " << setw(8) << precision << "%"
         << " │ " << setw(7) << recall << "%"
         << " │ " << setw(7) << f1 << "%"
         << " │ " << setprecision(3) << setw(8) << fCurrentThreshold << "│" << endl;
    
    // Log to file
    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount 
                << " - Acc: " << accuracy 
                << "% Prec: " << precision
                << "% Rec: " << recall
                << "% F1: " << f1 
                << " Threshold: " << fCurrentThreshold << endl;
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n╔═══════════════════════════════════════════╗" << endl;
    cout << "║  PHOTONTRIGGERML FINAL SUMMARY (IMPROVED) ║" << endl;
    cout << "╚═══════════════════════════════════════════╝" << endl;
    
    // Basic statistics
    cout << "\n┌─── Event Statistics ──────────────────────┐" << endl;
    cout << "│ Events processed:     " << setw(15) << fEventCount << " │" << endl;
    cout << "│ Stations analyzed:    " << setw(15) << fStationCount << " │" << endl;
    cout << "│ Final threshold:      " << setw(15) << fixed << setprecision(4) 
         << fCurrentThreshold << " │" << endl;
    cout << "│ Target percentile:    " << setw(14) << (fTargetPercentile * 100) << "% │" << endl;
    cout << "└───────────────────────────────────────────┘" << endl;
    
    // Calculate detailed metrics
    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    
    if (total > 0) {
        double accuracy = 100.0 * (fTruePositives + fTrueNegatives) / total;
        double precision = (fTruePositives + fFalsePositives > 0) ? 
                          100.0 * fTruePositives / (fTruePositives + fFalsePositives) : 0;
        double recall = (fTruePositives + fFalseNegatives > 0) ? 
                       100.0 * fTruePositives / (fTruePositives + fFalseNegatives) : 0;
        double specificity = (fTrueNegatives + fFalsePositives > 0) ?
                           100.0 * fTrueNegatives / (fTrueNegatives + fFalsePositives) : 0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;
        double mcc_num = (fTruePositives * fTrueNegatives) - (fFalsePositives * fFalseNegatives);
        double mcc_den = sqrt((double)(fTruePositives + fFalsePositives) * 
                             (fTruePositives + fFalseNegatives) * 
                             (fTrueNegatives + fFalsePositives) * 
                             (fTrueNegatives + fFalseNegatives));
        double mcc = (mcc_den > 0) ? mcc_num / mcc_den : 0;
        
        cout << "\n┌─── Performance Metrics ───────────────────┐" << endl;
        cout << "│ Accuracy:      " << setw(21) << fixed << setprecision(2) 
             << accuracy << "% │" << endl;
        cout << "│ Precision:     " << setw(21) << precision << "% │" << endl;
        cout << "│ Recall:        " << setw(21) << recall << "% │" << endl;
        cout << "│ Specificity:   " << setw(21) << specificity << "% │" << endl;
        cout << "│ F1-Score:      " << setw(21) << f1 << "% │" << endl;
        cout << "│ MCC:           " << setw(21) << fixed << setprecision(4) 
             << mcc << "   │" << endl;
        cout << "└───────────────────────────────────────────┘" << endl;
        
        // Confusion Matrix
        cout << "\n┌─── Confusion Matrix ──────────────────────┐" << endl;
        cout << "│                  PREDICTED                │" << endl;
        cout << "│              Normal    Anomaly            │" << endl;
        cout << "│   ┌─────────┬─────────┬─────────┐         │" << endl;
        cout << "│ A │ Normal  │" << setw(8) << fTrueNegatives 
             << " │" << setw(8) << fFalsePositives << " │         │" << endl;
        cout << "│ C ├─────────┼─────────┼─────────┤         │" << endl;
        cout << "│ T │ Photon  │" << setw(8) << fFalseNegatives 
             << " │" << setw(8) << fTruePositives << " │         │" << endl;
        cout << "│   └─────────┴─────────┴─────────┘         │" << endl;
        cout << "└───────────────────────────────────────────┘" << endl;
        
        // Class distribution
        int actual_photons = fTruePositives + fFalseNegatives;
        int actual_hadrons = fTrueNegatives + fFalsePositives;
        
        cout << "\n┌─── Detection Analysis ────────────────────┐" << endl;
        if (actual_photons > 0) {
            double photon_detection_rate = 100.0 * fTruePositives / actual_photons;
            cout << "│ Photon Detection Rate: " << setw(14) << fixed << setprecision(2) 
                 << photon_detection_rate << "% │" << endl;
        }
        if (actual_hadrons > 0) {
            double hadron_rejection_rate = 100.0 * fTrueNegatives / actual_hadrons;
            double false_alarm_rate = 100.0 * fFalsePositives / actual_hadrons;
            cout << "│ Hadron Rejection Rate: " << setw(14) 
                 << hadron_rejection_rate << "% │" << endl;
            cout << "│ False Alarm Rate:      " << setw(14) 
                 << false_alarm_rate << "% │" << endl;
        }
        cout << "└───────────────────────────────────────────┘" << endl;
        
        // Performance assessment
        cout << "\n┌─── Performance Assessment ────────────────┐" << endl;
        if (recall < 10) {
            cout << "│ ⚠ Low recall - consider lowering         │" << endl;
            cout << "│   the anomaly threshold percentile       │" << endl;
        } else if (recall < 25) {
            cout << "│ ✓ Moderate recall achieved               │" << endl;
            cout << "│   Further tuning may improve performance │" << endl;
        } else {
            cout << "│ ✓ Good recall achieved!                  │" << endl;
        }
        
        if (precision < 20) {
            cout << "│ ⚠ High false positive rate               │" << endl;
            cout << "│   Consider raising threshold             │" << endl;
        } else if (precision < 40) {
            cout << "│ ✓ Acceptable false positive rate         │" << endl;
        } else {
            cout << "│ ✓ Excellent precision!                   │" << endl;
        }
        cout << "└───────────────────────────────────────────┘" << endl;
    }
    
    // Training statistics
    if (!fHadronFeatures.empty()) {
        cout << "\n┌─── Training Statistics ───────────────────┐" << endl;
        cout << "│ Hadron samples:     " << setw(18) << fHadronFeatures.size() << " │" << endl;
        cout << "│ Photon samples:     " << setw(18) << fPhotonFeatures.size() << " │" << endl;
        cout << "│ Feature dimensions: " << setw(18) << 25 << " │" << endl;
        cout << "│ Network depth:      " << setw(18) << "4 layers" << " │" << endl;
        cout << "└───────────────────────────────────────────┘" << endl;
    }
    
    // Save ROOT file
    if (fOutputFile) {
        fOutputFile->cd();
        
        if (fMLTree) {
            fMLTree->Write();
            cout << "\n✓ Wrote " << fMLTree->GetEntries() << " entries to tree" << endl;
        }
        
        // Write histograms
        if (hReconstructionError) hReconstructionError->Write();
        if (hErrorPhotons) hErrorPhotons->Write();
        if (hErrorHadrons) hErrorHadrons->Write();
        if (hConfusionMatrix) hConfusionMatrix->Write();
        if (hThresholdEvolution) hThresholdEvolution->Write();
        if (hFeatureImportance) hFeatureImportance->Write();
        
        cout << "✓ Histograms written to " << fOutputFileName << endl;
        
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = nullptr;
    }
    
    // Close log file
    if (fLogFile.is_open()) {
        fLogFile << "\n=== FINAL SUMMARY ===" << endl;
        fLogFile << "Events: " << fEventCount << endl;
        fLogFile << "Stations: " << fStationCount << endl;
        fLogFile << "TP: " << fTruePositives << " FP: " << fFalsePositives << endl;
        fLogFile << "TN: " << fTrueNegatives << " FN: " << fFalseNegatives << endl;
        fLogFile << "Final Threshold: " << fCurrentThreshold << endl;
        fLogFile.close();
    }
    
    cout << "\n╔═══════════════════════════════════════════╗" << endl;
    cout << "║           ANALYSIS COMPLETE               ║" << endl;
    cout << "╚═══════════════════════════════════════════╝" << endl;
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

// Static methods for PMTTraceModule compatibility
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
