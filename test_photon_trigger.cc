/**
 * test_photon_trigger.cc
 * Simple test program to verify PhotonTriggerML module compilation and basic functionality
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "PhotonTriggerML.h"

using namespace std;

// Generate a simulated FADC trace
vector<double> GenerateSimulatedTrace(bool is_photon, double energy_eV = 1e18) {
    vector<double> trace(2048, 50.0);  // Baseline at 50 ADC
    
    // Random number generator
    default_random_engine gen(is_photon ? 42 : 84);
    normal_distribution<double> noise(0, 2);
    
    // Peak parameters
    int peak_position = 650 + (is_photon ? 0 : 50);
    double peak_amplitude = 200 + (energy_eV / 1e18) * 100;
    
    // Generate main peak
    double sigma = is_photon ? 30 : 50;  // Photons have narrower peaks
    for (int i = 0; i < 2048; i++) {
        // Gaussian peak
        double signal = peak_amplitude * exp(-pow(i - peak_position, 2) / (2 * sigma * sigma));
        
        // Add muon peaks for hadrons
        if (!is_photon && i > peak_position + 100) {
            signal += 50 * exp(-pow(i - peak_position - 150, 2) / (2 * 25 * 25));
            signal += 30 * exp(-pow(i - peak_position - 300, 2) / (2 * 30 * 30));
        }
        
        trace[i] += signal + noise(gen);
    }
    
    return trace;
}

// Test feature extraction
void TestFeatureExtraction() {
    cout << "\n=== Testing Feature Extraction ===" << endl;
    
    TraceFeatureExtractor extractor;
    
    // Test with photon-like trace
    auto photon_trace = GenerateSimulatedTrace(true, 5e18);
    auto photon_features = extractor.ExtractFeatures(photon_trace);
    
    cout << "Photon-like trace features:" << endl;
    cout << "  Peak amplitude: " << photon_features.peak_amplitude << " VEM" << endl;
    cout << "  Total charge: " << photon_features.total_charge << " VEM" << endl;
    cout << "  Rise time (10-90%): " << photon_features.risetime_10_90 << " ns" << endl;
    cout << "  FWHM: " << photon_features.fwhm << " ns" << endl;
    cout << "  Smoothness: " << photon_features.smoothness << endl;
    cout << "  Early/Late ratio: " << photon_features.early_to_late_ratio << endl;
    cout << "  Number of peaks: " << photon_features.num_peaks << endl;
    
    // Test with hadron-like trace
    auto hadron_trace = GenerateSimulatedTrace(false, 5e18);
    auto hadron_features = extractor.ExtractFeatures(hadron_trace);
    
    cout << "\nHadron-like trace features:" << endl;
    cout << "  Peak amplitude: " << hadron_features.peak_amplitude << " VEM" << endl;
    cout << "  Total charge: " << hadron_features.total_charge << " VEM" << endl;
    cout << "  Rise time (10-90%): " << hadron_features.risetime_10_90 << " ns" << endl;
    cout << "  FWHM: " << hadron_features.fwhm << " ns" << endl;
    cout << "  Smoothness: " << hadron_features.smoothness << endl;
    cout << "  Early/Late ratio: " << hadron_features.early_to_late_ratio << endl;
    cout << "  Number of peaks: " << hadron_features.num_peaks << endl;
}

// Test neural network discrimination
void TestNeuralNetwork() {
    cout << "\n=== Testing Neural Network ===" << endl;
    
    PhotonDiscriminatorNN nn;
    TraceFeatureExtractor extractor;
    
    // Test multiple traces
    int n_photons_correct = 0;
    int n_hadrons_correct = 0;
    int n_tests = 100;
    
    for (int i = 0; i < n_tests; i++) {
        // Test photon
        auto photon_trace = GenerateSimulatedTrace(true, 1e18 + i * 1e16);
        auto photon_features = extractor.ExtractFeatures(photon_trace);
        float photon_score = nn.Predict(photon_features.GetFeatureVector());
        if (photon_score > 0.5) n_photons_correct++;
        
        // Test hadron
        auto hadron_trace = GenerateSimulatedTrace(false, 1e18 + i * 1e16);
        auto hadron_features = extractor.ExtractFeatures(hadron_trace);
        float hadron_score = nn.Predict(hadron_features.GetFeatureVector());
        if (hadron_score < 0.5) n_hadrons_correct++;
    }
    
    cout << "Photon identification rate: " << (100.0 * n_photons_correct / n_tests) << "%" << endl;
    cout << "Hadron rejection rate: " << (100.0 * n_hadrons_correct / n_tests) << "%" << endl;
    
    // Detailed test on single traces
    auto photon_trace = GenerateSimulatedTrace(true, 5e18);
    auto photon_features = extractor.ExtractFeatures(photon_trace);
    float photon_score = nn.Predict(photon_features.GetFeatureVector());
    
    auto hadron_trace = GenerateSimulatedTrace(false, 5e18);
    auto hadron_features = extractor.ExtractFeatures(hadron_trace);
    float hadron_score = nn.Predict(hadron_features.GetFeatureVector());
    
    cout << "\nDetailed NN outputs:" << endl;
    cout << "  Photon trace -> NN output: " << photon_score 
         << " (expect > 0.5)" << endl;
    cout << "  Hadron trace -> NN output: " << hadron_score 
         << " (expect < 0.5)" << endl;
    
    // Test threshold behavior
    cout << "\nThreshold tests:" << endl;
    for (float threshold : {0.3, 0.5, 0.7}) {
        bool photon_trigger = nn.IsPhotonLike(photon_features.GetFeatureVector(), threshold);
        bool hadron_trigger = nn.IsPhotonLike(hadron_features.GetFeatureVector(), threshold);
        cout << "  Threshold " << threshold << ": "
             << "Photon=" << (photon_trigger ? "TRIGGER" : "reject") << ", "
             << "Hadron=" << (hadron_trigger ? "TRIGGER" : "reject") << endl;
    }
}

// Test performance metrics
void TestPerformanceMetrics() {
    cout << "\n=== Testing Performance Metrics ===" << endl;
    
    PhotonDiscriminatorNN nn;
    TraceFeatureExtractor extractor;
    
    // Measure processing time
    auto photon_trace = GenerateSimulatedTrace(true, 5e18);
    
    clock_t start = clock();
    for (int i = 0; i < 1000; i++) {
        auto features = extractor.ExtractFeatures(photon_trace);
        float score = nn.Predict(features.GetFeatureVector());
        (void)score;  // Suppress unused warning
    }
    clock_t end = clock();
    
    double time_per_trace = (double)(end - start) / CLOCKS_PER_SEC / 1000.0;
    cout << "Processing time per trace: " << time_per_trace * 1e6 << " microseconds" << endl;
    
    if (time_per_trace < 1e-6) {
        cout << "✓ Meets real-time requirement (< 1 microsecond)" << endl;
    } else {
        cout << "⚠ May need optimization for real-time operation" << endl;
    }
    
    // Memory usage estimate (simplified)
    size_t feature_size = sizeof(TraceFeatureExtractor::Features);
    size_t nn_size = sizeof(PhotonDiscriminatorNN);
    cout << "\nMemory usage:" << endl;
    cout << "  Feature structure: " << feature_size << " bytes" << endl;
    cout << "  Neural network: " << nn_size << " bytes" << endl;
    cout << "  Total per channel: " << (feature_size + nn_size) << " bytes" << endl;
}

// Main test program
int main(int argc, char* argv[]) {
    cout << "========================================" << endl;
    cout << "PhotonTriggerML Module Test Program" << endl;
    cout << "========================================" << endl;
    
    try {
        // Run tests
        TestFeatureExtraction();
        TestNeuralNetwork();
        TestPerformanceMetrics();
        
        cout << "\n========================================" << endl;
        cout << "✓ All tests completed successfully!" << endl;
        cout << "========================================" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
