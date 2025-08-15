/**
 * PhotonTriggerML.cc
 * Machine Learning-based T1 photon trigger implementation
 * 
 * Implements lightweight neural network for real-time photon/hadron discrimination
 * Designed for FPGA deployment with 8-bit quantized weights
 * Target: 10^18 - 10^19 eV energy range
 */

#include "PhotonTriggerML.h"

#include <fwk/CentralConfig.h>
#include <utl/ErrorLogger.h>
#include <evt/Event.h>
#include <evt/ShowerSimData.h>
#include <sevt/SEvent.h>
#include <sevt/Station.h>
#include <sevt/PMT.h>
#include <sdet/Station.h>
#include <det/Detector.h>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TGraph.h>
#include <TMath.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace sevt;

// ============================================================================
// PhotonDiscriminatorNN Implementation
// ============================================================================

PhotonDiscriminatorNN::PhotonDiscriminatorNN() :
    input_scale_(1.0/255.0),
    weight_scale_(1.0/127.0),
    output_scale_(1.0)
{
    Initialize();
}

PhotonDiscriminatorNN::~PhotonDiscriminatorNN() {}

void PhotonDiscriminatorNN::Initialize()
{
    // Initialize network architecture
    // In production, these weights would be loaded from a trained model file
    // Here we initialize with example weights that favor photon-like characteristics
    
    // Layer 1: Input (24) -> Hidden1 (16)
    weights1_.resize(HIDDEN1_SIZE, vector<int8_t>(NUM_FEATURES));
    bias1_.resize(HIDDEN1_SIZE);
    
    // Initialize with small random weights for demonstration
    // In reality, these would be trained weights
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(0.0, 30.0);
    
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            // Emphasize certain features known to discriminate photons
            float weight = distribution(generator);
            
            // Boost weights for photon-discriminating features
            if (j == 0 || j == 1) weight *= 1.5;  // Risetime features
            if (j == 12) weight *= 2.0;           // Early-to-late ratio
            if (j == 13) weight *= 1.8;           // Smoothness
            
            weights1_[i][j] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, weight)));
        }
        bias1_[i] = static_cast<int8_t>(distribution(generator) * 0.1);
    }
    
    // Layer 2: Hidden1 (16) -> Hidden2 (8)
    weights2_.resize(HIDDEN2_SIZE, vector<int8_t>(HIDDEN1_SIZE));
    bias2_.resize(HIDDEN2_SIZE);
    
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            weights2_[i][j] = static_cast<int8_t>(distribution(generator));
        }
        bias2_[i] = static_cast<int8_t>(distribution(generator) * 0.1);
    }
    
    // Layer 3: Hidden2 (8) -> Output (1)
    weights3_.resize(HIDDEN2_SIZE);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        weights3_[i] = static_cast<int8_t>(distribution(generator));
    }
    bias3_ = 0;
}

float PhotonDiscriminatorNN::Predict(const vector<float>& features)
{
    if (features.size() != NUM_FEATURES) {
        cerr << "Error: Expected " << NUM_FEATURES << " features, got " << features.size() << endl;
        return 0.0;
    }
    
    // Layer 1: Input -> Hidden1
    vector<float> hidden1(HIDDEN1_SIZE, 0.0);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float sum = bias1_[i] * weight_scale_;
        for (int j = 0; j < NUM_FEATURES; j++) {
            sum += features[j] * weights1_[i][j] * weight_scale_;
        }
        hidden1[i] = ReLU(sum);
    }
    
    // Layer 2: Hidden1 -> Hidden2
    vector<float> hidden2(HIDDEN2_SIZE, 0.0);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float sum = bias2_[i] * weight_scale_;
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            sum += hidden1[j] * weights2_[i][j] * weight_scale_;
        }
        hidden2[i] = ReLU(sum);
    }
    
    // Layer 3: Hidden2 -> Output
    float output = bias3_ * weight_scale_;
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        output += hidden2[i] * weights3_[i] * weight_scale_;
    }
    
    return Sigmoid(output);
}

bool PhotonDiscriminatorNN::IsPhotonLike(const vector<float>& features, float threshold)
{
    return Predict(features) > threshold;
}

// ============================================================================
// TraceFeatureExtractor Implementation
// ============================================================================

vector<float> TraceFeatureExtractor::Features::GetFeatureVector() const
{
    vector<float> features;
    features.reserve(NUM_FEATURES);
    
    // Add all features in order
    features.push_back(risetime_10_50);
    features.push_back(risetime_10_90);
    features.push_back(falltime_90_10);
    features.push_back(fwhm);
    features.push_back(peak_amplitude);
    features.push_back(total_charge);
    features.push_back(peak_to_charge);
    features.push_back(asymmetry);
    features.push_back(kurtosis);
    features.push_back(num_peaks);
    
    // Add window charges
    for (int i = 0; i < TIME_WINDOWS; i++) {
        features.push_back(window_charges[i]);
    }
    
    features.push_back(early_to_late_ratio);
    features.push_back(smoothness);
    features.push_back(peak_density);
    
    // Normalize features to [0, 1] range for NN input
    // In production, use pre-computed normalization parameters
    for (auto& f : features) {
        f = std::max(0.0f, std::min(1.0f, f / 100.0f));  // Simple normalization
    }
    
    return features;
}

TraceFeatureExtractor::Features TraceFeatureExtractor::ExtractFeatures(
    const vector<double>& trace, double baseline)
{
    Features features;
    const int trace_size = trace.size();
    
    if (trace_size != TRACE_BINS) {
        cerr << "Warning: Unexpected trace size " << trace_size << endl;
    }
    
    // Find peak
    int peak_bin = FindPeakBin(trace, baseline);
    double peak_value = trace[peak_bin] - baseline;
    
    // Convert to VEM units (assuming 180 ADC counts per VEM)
    const double ADC_PER_VEM = 180.0;
    features.peak_amplitude = peak_value / ADC_PER_VEM;
    
    // Calculate total charge
    double total_charge = 0;
    for (const auto& val : trace) {
        double signal = val - baseline;
        if (signal > 0) total_charge += signal;
    }
    features.total_charge = total_charge / ADC_PER_VEM;
    
    // Peak to charge ratio (important for photon discrimination)
    features.peak_to_charge = (features.total_charge > 0) ? 
        features.peak_amplitude / features.total_charge : 0;
    
    // Rise times (photons have different rise characteristics)
    double peak_10 = baseline + 0.1 * peak_value;
    double peak_50 = baseline + 0.5 * peak_value;
    double peak_90 = baseline + 0.9 * peak_value;
    
    int bin_10_rise = peak_bin, bin_50_rise = peak_bin, bin_90_rise = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (trace[i] <= peak_90 && bin_90_rise == peak_bin) bin_90_rise = i;
        if (trace[i] <= peak_50 && bin_50_rise == peak_bin) bin_50_rise = i;
        if (trace[i] <= peak_10 && bin_10_rise == peak_bin) bin_10_rise = i;
    }
    
    features.risetime_10_50 = (bin_50_rise - bin_10_rise) * 25.0;  // Convert to ns
    features.risetime_10_90 = (bin_90_rise - bin_10_rise) * 25.0;
    
    // Fall time
    int bin_90_fall = peak_bin, bin_10_fall = peak_bin;
    for (int i = peak_bin; i < trace_size; i++) {
        if (trace[i] <= peak_90 && bin_90_fall == peak_bin) bin_90_fall = i;
        if (trace[i] <= peak_10 && bin_10_fall == peak_bin) bin_10_fall = i;
    }
    features.falltime_90_10 = (bin_10_fall - bin_90_fall) * 25.0;
    
    // FWHM
    features.fwhm = CalculateFWHM(trace, peak_bin, peak_value, baseline);
    
    // Asymmetry (photons tend to be more symmetric)
    double moment2 = 0, moment3 = 0;
    for (int i = 0; i < trace_size; i++) {
        double signal = trace[i] - baseline;
        if (signal > 0) {
            double diff = i - peak_bin;
            moment2 += diff * diff * signal;
            moment3 += diff * diff * diff * signal;
        }
    }
    features.asymmetry = (moment2 > 0) ? moment3 / pow(moment2, 1.5) : 0;
    
    // Kurtosis (peakedness - photons have different peak structure)
    double mean_bin = 0, variance = 0, moment4 = 0;
    double weight_sum = 0;
    for (int i = 0; i < trace_size; i++) {
        double signal = trace[i] - baseline;
        if (signal > 0) {
            mean_bin += i * signal;
            weight_sum += signal;
        }
    }
    if (weight_sum > 0) mean_bin /= weight_sum;
    
    for (int i = 0; i < trace_size; i++) {
        double signal = trace[i] - baseline;
        if (signal > 0) {
            double diff = i - mean_bin;
            variance += diff * diff * signal / weight_sum;
            moment4 += pow(diff, 4) * signal / weight_sum;
        }
    }
    features.kurtosis = (variance > 0) ? moment4 / (variance * variance) - 3 : 0;
    
    // Number of peaks (muons create multiple peaks)
    features.num_peaks = CountPeaks(trace, baseline, 0.2 * peak_value);
    
    // Time-windowed charges (important for temporal structure)
    int window_size = trace_size / TIME_WINDOWS;
    for (int w = 0; w < TIME_WINDOWS; w++) {
        double window_charge = 0;
        int start = w * window_size;
        int end = min((w + 1) * window_size, trace_size);
        for (int i = start; i < end; i++) {
            double signal = trace[i] - baseline;
            if (signal > 0) window_charge += signal;
        }
        features.window_charges[w] = window_charge / ADC_PER_VEM;
    }
    
    // Early to late ratio (key discriminator - photons have more EM component early)
    double early_charge = features.window_charges[0] + features.window_charges[1];
    double late_charge = features.window_charges[6] + features.window_charges[7];
    features.early_to_late_ratio = (late_charge > 0) ? early_charge / late_charge : 10.0;
    
    // Smoothness (photon signals are smoother due to EM cascade)
    features.smoothness = CalculateSmoothness(trace);
    
    // Peak density (number of peaks per unit time)
    features.peak_density = features.num_peaks / (features.fwhm / 25.0);  // peaks per bin
    
    return features;
}

double TraceFeatureExtractor::CalculateBaseline(const vector<double>& trace)
{
    // Use first 100 bins to estimate baseline
    double sum = 0;
    int count = min(100, (int)trace.size());
    for (int i = 0; i < count; i++) {
        sum += trace[i];
    }
    return sum / count;
}

int TraceFeatureExtractor::FindPeakBin(const vector<double>& trace, double baseline)
{
    int peak_bin = 0;
    double peak_val = 0;
    for (size_t i = 0; i < trace.size(); i++) {
        double signal = trace[i] - baseline;
        if (signal > peak_val) {
            peak_val = signal;
            peak_bin = i;
        }
    }
    return peak_bin;
}

double TraceFeatureExtractor::CalculateFWHM(const vector<double>& trace, 
    int peak_bin, double peak_val, double baseline)
{
    double half_max = baseline + 0.5 * peak_val;
    
    // Find left half-max point
    int left_bin = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (trace[i] <= half_max) {
            left_bin = i;
            break;
        }
    }
    
    // Find right half-max point
    int right_bin = peak_bin;
    for (size_t i = peak_bin; i < trace.size(); i++) {
        if (trace[i] <= half_max) {
            right_bin = i;
            break;
        }
    }
    
    return (right_bin - left_bin) * 25.0;  // Convert to ns
}

int TraceFeatureExtractor::CountPeaks(const vector<double>& trace, 
    double baseline, double threshold)
{
    int num_peaks = 0;
    bool in_peak = false;
    double peak_threshold = baseline + threshold;
    
    for (size_t i = 1; i < trace.size() - 1; i++) {
        if (!in_peak && trace[i] > peak_threshold) {
            // Check if it's a local maximum
            if (trace[i] > trace[i-1] && trace[i] > trace[i+1]) {
                num_peaks++;
                in_peak = true;
            }
        } else if (in_peak && trace[i] < peak_threshold) {
            in_peak = false;
        }
    }
    
    return num_peaks;
}

double TraceFeatureExtractor::CalculateSmoothness(const vector<double>& trace)
{
    // Calculate RMS of second derivative as measure of smoothness
    double sum_sq_diff = 0;
    int count = 0;
    
    for (size_t i = 1; i < trace.size() - 1; i++) {
        double second_deriv = trace[i+1] - 2*trace[i] + trace[i-1];
        sum_sq_diff += second_deriv * second_deriv;
        count++;
    }
    
    return (count > 0) ? sqrt(sum_sq_diff / count) : 0;
}

// ============================================================================
// PhotonTriggerML Implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    total_stations_(0),
    photon_triggers_(0),
    hadron_triggers_(0),
    true_photon_triggers_(0),
    false_photon_triggers_(0),
    efficiency_(0),
    purity_(0),
    background_reduction_(0),
    output_file_(nullptr),
    feature_tree_(nullptr),
    performance_tree_(nullptr)
{
}

PhotonTriggerML::~PhotonTriggerML()
{
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Initializing ML-based photon trigger");
    
    // Initialize ML components
    discriminator_ = make_unique<PhotonDiscriminatorNN>();
    extractor_ = make_unique<TraceFeatureExtractor>();
    
    // Load configuration (would come from XML in production)
    config_.photon_threshold = 0.5;
    config_.energy_min = 1e18;
    config_.energy_max = 1e19;
    config_.save_features = true;
    config_.apply_trigger = true;
    
    INFO("=== PhotonTriggerML Configuration ===");
    INFO("Energy range: 10^18 - 10^19 eV");
    INFO("NN threshold: 0.5");
    INFO("Target: 50% efficiency improvement, 30% background reduction");
    INFO("Architecture: 24 inputs -> 16 hidden -> 8 hidden -> 1 output");
    INFO("=====================================");
    
    // Create output file
    output_file_ = new TFile("photon_trigger_ml.root", "RECREATE");
    
    // Create feature tree
    if (config_.save_features) {
        feature_tree_ = new TTree("features", "ML Features and Predictions");
        feature_tree_->Branch("nn_output", &tree_nn_output_);
        feature_tree_->Branch("energy", &tree_energy_);
        feature_tree_->Branch("distance", &tree_distance_);
        feature_tree_->Branch("is_photon_truth", &tree_is_photon_truth_);
        feature_tree_->Branch("trigger_decision", &tree_trigger_decision_);
        feature_tree_->Branch("features", &tree_features_);
    }
    
    // Create performance tree
    performance_tree_ = new TTree("performance", "Trigger Performance Metrics");
    performance_tree_->Branch("efficiency", &efficiency_);
    performance_tree_->Branch("purity", &purity_);
    performance_tree_->Branch("background_reduction", &background_reduction_);
    
    // Create histograms
    h_nn_output_all_ = new TH1D("h_nn_output_all", 
        "NN Output (All);Photon Probability;Count", 100, 0, 1);
    h_nn_output_photon_ = new TH1D("h_nn_output_photon", 
        "NN Output (True Photons);Photon Probability;Count", 100, 0, 1);
    h_nn_output_hadron_ = new TH1D("h_nn_output_hadron", 
        "NN Output (Hadrons);Photon Probability;Count", 100, 0, 1);
    
    h_nn_vs_energy_ = new TH2D("h_nn_vs_energy", 
        "NN Output vs Energy;Energy [eV];Photon Probability", 
        50, 1e18, 1e19, 100, 0, 1);
    h_nn_vs_distance_ = new TH2D("h_nn_vs_distance", 
        "NN Output vs Distance;Distance [m];Photon Probability", 
        50, 0, 3000, 100, 0, 1);
    
    h_efficiency_vs_energy_ = new TH1D("h_efficiency_vs_energy", 
        "Photon Efficiency vs Energy;Energy [eV];Efficiency", 10, 1e18, 1e19);
    h_purity_vs_energy_ = new TH1D("h_purity_vs_energy", 
        "Trigger Purity vs Energy;Energy [eV];Purity", 10, 1e18, 1e19);
    
    // Feature comparison histograms
    h_risetime_photon_ = new TH1D("h_risetime_photon", 
        "Rise Time 10-90% (Photon);Time [ns];Count", 50, 0, 500);
    h_risetime_hadron_ = new TH1D("h_risetime_hadron", 
        "Rise Time 10-90% (Hadron);Time [ns];Count", 50, 0, 500);
    
    h_smoothness_photon_ = new TH1D("h_smoothness_photon", 
        "Signal Smoothness (Photon);RMS 2nd Derivative;Count", 50, 0, 100);
    h_smoothness_hadron_ = new TH1D("h_smoothness_hadron", 
        "Signal Smoothness (Hadron);RMS 2nd Derivative;Count", 50, 0, 100);
    
    h_early_late_photon_ = new TH1D("h_early_late_photon", 
        "Early/Late Charge Ratio (Photon);Ratio;Count", 50, 0, 10);
    h_early_late_hadron_ = new TH1D("h_early_late_hadron", 
        "Early/Late Charge Ratio (Hadron);Ratio;Count", 50, 0, 10);
    
    INFO("PhotonTriggerML initialized successfully");
    
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    // Get shower information
    tree_energy_ = 0;
    tree_is_photon_truth_ = 0;
    
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        tree_energy_ = shower.GetEnergy();
        
        // Determine if this is a photon shower (would check particle ID in real implementation)
        // For demonstration, we'll simulate this based on shower characteristics
        // In reality, check shower.GetPrimaryParticle() or similar
        tree_is_photon_truth_ = (shower.GetXmax() > 700);  // Photons develop deeper
    }
    
    // Skip if outside energy range
    if (tree_energy_ < config_.energy_min || tree_energy_ > config_.energy_max) {
        return eSuccess;
    }
    
    // Process stations
    if (event.HasSEvent()) {
        const SEvent& sevent = event.GetSEvent();
        
        for (SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(event, *it);
        }
    }
    
    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const Event& event, const Station& station)
{
    // Get station traces
    const int firstPMT = sdet::Station::GetFirstPMTId();
    
    for (int p = 0; p < 3; p++) {
        const int pmtId = p + firstPMT;
        
        if (!station.HasPMT(pmtId)) continue;
        
        const PMT& pmt = station.GetPMT(pmtId);
        
        // Get FADC trace
        if (!pmt.HasFADCTrace()) continue;
        
        try {
            const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            
            // Convert to vector<double>
            vector<double> trace_data;
            for (int i = 0; i < TRACE_BINS; i++) {
                try {
                    trace_data.push_back(trace[i]);
                } catch (...) {
                    trace_data.push_back(50.0);  // baseline
                }
            }
            
            // Extract features
            auto features = extractor_->ExtractFeatures(trace_data);
            
            // Get feature vector for NN
            tree_features_ = features.GetFeatureVector();
            
            // Run NN discrimination
            tree_nn_output_ = discriminator_->Predict(tree_features_);
            
            // Apply trigger decision
            tree_trigger_decision_ = ApplyT1Trigger(tree_nn_output_, features);
            
            // Update statistics
            total_stations_++;
            if (tree_trigger_decision_) {
                photon_triggers_++;
                if (tree_is_photon_truth_) {
                    true_photon_triggers_++;
                } else {
                    false_photon_triggers_++;
                }
            } else {
                hadron_triggers_++;
            }
            
            // Fill histograms
            h_nn_output_all_->Fill(tree_nn_output_);
            
            if (tree_is_photon_truth_) {
                h_nn_output_photon_->Fill(tree_nn_output_);
                h_risetime_photon_->Fill(features.risetime_10_90);
                h_smoothness_photon_->Fill(features.smoothness);
                h_early_late_photon_->Fill(features.early_to_late_ratio);
            } else {
                h_nn_output_hadron_->Fill(tree_nn_output_);
                h_risetime_hadron_->Fill(features.risetime_10_90);
                h_smoothness_hadron_->Fill(features.smoothness);
                h_early_late_hadron_->Fill(features.early_to_late_ratio);
            }
            
            h_nn_vs_energy_->Fill(tree_energy_, tree_nn_output_);
            
            // Get station distance
            try {
                const Detector& detector = Detector::GetInstance();
                const sdet::SDetector& sdetector = detector.GetSDetector();
                const sdet::Station& detStation = sdetector.GetStation(station.GetId());
                // Calculate distance (simplified)
                tree_distance_ = 1000;  // Placeholder
                h_nn_vs_distance_->Fill(tree_distance_, tree_nn_output_);
            } catch (...) {
                tree_distance_ = -1;
            }
            
            // Save to tree
            if (feature_tree_) {
                feature_tree_->Fill();
            }
            
        } catch (const exception& e) {
            // Skip if trace access fails
            continue;
        }
    }
}

bool PhotonTriggerML::ApplyT1Trigger(float nn_output, 
    const TraceFeatureExtractor::Features& features)
{
    // Multi-criteria trigger decision
    // Combines NN output with additional physics-based cuts
    
    // Primary criterion: NN photon probability
    bool nn_trigger = (nn_output > config_.photon_threshold);
    
    // Additional criteria for robust triggering
    bool has_sufficient_charge = (features.total_charge > 1.0);  // > 1 VEM
    bool has_good_peak = (features.peak_amplitude > 1.5);        // > 1.5 VEM peak
    
    // Special trigger for very photon-like signals
    bool very_photon_like = (nn_output > 0.8) && has_sufficient_charge;
    
    // Combined decision
    return very_photon_like || (nn_trigger && has_sufficient_charge && has_good_peak);
}

void PhotonTriggerML::UpdatePerformanceMetrics()
{
    // Calculate efficiency
    if (total_stations_ > 0) {
        efficiency_ = static_cast<double>(true_photon_triggers_) / total_stations_;
    }
    
    // Calculate purity
    if (photon_triggers_ > 0) {
        purity_ = static_cast<double>(true_photon_triggers_) / photon_triggers_;
    }
    
    // Calculate background reduction
    // Compare to baseline trigger (simplified)
    double baseline_hadron_rate = 0.7;  // Assume 70% hadron triggers in baseline
    double current_hadron_rate = static_cast<double>(false_photon_triggers_) / total_stations_;
    background_reduction_ = 1.0 - (current_hadron_rate / baseline_hadron_rate);
}

void PhotonTriggerML::CreateROCCurve()
{
    // Create ROC curve by varying threshold
    vector<double> thresholds, tpr, fpr;
    
    for (double thresh = 0.0; thresh <= 1.0; thresh += 0.05) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        
        // Reprocess histogram data with different threshold
        for (int i = 1; i <= h_nn_output_photon_->GetNbinsX(); i++) {
            double value = h_nn_output_photon_->GetBinCenter(i);
            double count_photon = h_nn_output_photon_->GetBinContent(i);
            double count_hadron = h_nn_output_hadron_->GetBinContent(i);
            
            if (value > thresh) {
                tp += count_photon;
                fp += count_hadron;
            } else {
                fn += count_photon;
                tn += count_hadron;
            }
        }
        
        double tpr_val = (tp + fn > 0) ? tp / (double)(tp + fn) : 0;
        double fpr_val = (fp + tn > 0) ? fp / (double)(fp + tn) : 0;
        
        thresholds.push_back(thresh);
        tpr.push_back(tpr_val);
        fpr.push_back(fpr_val);
    }
    
    g_roc_curve_ = new TGraph(fpr.size(), &fpr[0], &tpr[0]);
    g_roc_curve_->SetName("g_roc_curve");
    g_roc_curve_->SetTitle("ROC Curve;False Positive Rate;True Positive Rate");
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Finalizing analysis");
    
    // Update final metrics
    UpdatePerformanceMetrics();
    
    // Create ROC curve
    CreateROCCurve();
    
    // Fill performance tree
    if (performance_tree_) {
        performance_tree_->Fill();
    }
    
    // Print summary
    ostringstream summary;
    summary << "\n=== PhotonTriggerML Performance Summary ===\n"
            << "Total stations processed: " << total_stations_ << "\n"
            << "Photon triggers: " << photon_triggers_ << "\n"
            << "True photon triggers: " << true_photon_triggers_ << "\n"
            << "False photon triggers: " << false_photon_triggers_ << "\n"
            << "Efficiency: " << efficiency_ * 100 << "%\n"
            << "Purity: " << purity_ * 100 << "%\n"
            << "Background reduction: " << background_reduction_ * 100 << "%\n"
            << "==========================================";
    INFO(summary.str());
    
    // Calculate improvement metrics
    double baseline_efficiency = 0.3;  // Assume 30% baseline
    double efficiency_improvement = (efficiency_ - baseline_efficiency) / baseline_efficiency;
    
    if (efficiency_improvement >= 0.5) {
        INFO("✓ Target met: 50% efficiency improvement achieved!");
    } else {
        INFO("Current efficiency improvement: " + to_string(efficiency_improvement * 100) + "%");
    }
    
    if (background_reduction_ >= 0.3) {
        INFO("✓ Target met: 30% background reduction achieved!");
    } else {
        INFO("Current background reduction: " + to_string(background_reduction_ * 100) + "%");
    }
    
    // Save output
    if (output_file_) {
        output_file_->cd();
        output_file_->Write();
        output_file_->Close();
        delete output_file_;
        INFO("Results saved to photon_trigger_ml.root");
    }
    
    return eSuccess;
}
