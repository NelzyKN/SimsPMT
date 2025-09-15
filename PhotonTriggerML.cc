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
// Autoencoder for Anomaly Detection
// ============================================================================

PhotonTriggerML::Autoencoder::Autoencoder() :
    fInputSize(0), fLatentSize(8), fInitialized(false), fAnomalyThreshold(0.5)
{
}

void PhotonTriggerML::Autoencoder::Initialize(int input_size)
{
    fInputSize = input_size;
    fLatentSize = 8;  // Compressed representation
    
    cout << "Initializing Autoencoder: " << input_size 
         << " -> " << fLatentSize << " -> " << input_size << endl;
    
    // Initialize encoder weights (input -> latent)
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, sqrt(2.0 / input_size));
    
    fEncoderWeights.resize(fLatentSize, std::vector<double>(input_size));
    fEncoderBias.resize(fLatentSize);
    
    for (int i = 0; i < fLatentSize; ++i) {
        for (int j = 0; j < input_size; ++j) {
            fEncoderWeights[i][j] = dist(gen);
        }
        fEncoderBias[i] = 0.01;
    }
    
    // Initialize decoder weights (latent -> output)
    fDecoderWeights.resize(input_size, std::vector<double>(fLatentSize));
    fDecoderBias.resize(input_size);
    
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < fLatentSize; ++j) {
            fDecoderWeights[i][j] = dist(gen);
        }
        fDecoderBias[i] = 0.01;
    }
    
    // Initialize optimizer states
    fEncoderMomentum.resize(fLatentSize, std::vector<double>(input_size, 0));
    fDecoderMomentum.resize(input_size, std::vector<double>(fLatentSize, 0));
    
    fInitialized = true;
}

double PhotonTriggerML::Autoencoder::GetReconstructionError(const std::vector<double>& features)
{
    if (!fInitialized || static_cast<int>(features.size()) != fInputSize) return 1e9;
    
    // Encode to latent space
    std::vector<double> latent(fLatentSize);
    for (int i = 0; i < fLatentSize; ++i) {
        double sum = fEncoderBias[i];
        for (size_t j = 0; j < features.size(); ++j) {
            sum += fEncoderWeights[i][j] * features[j];
        }
        latent[i] = tanh(sum);  // Use tanh activation
    }
    
    // Decode back to feature space
    std::vector<double> reconstructed(fInputSize);
    for (int i = 0; i < fInputSize; ++i) {
        double sum = fDecoderBias[i];
        for (int j = 0; j < fLatentSize; ++j) {
            sum += fDecoderWeights[i][j] * latent[j];
        }
        reconstructed[i] = 1.0 / (1.0 + exp(-sum));  // Sigmoid for [0,1] output
    }
    
    // Calculate reconstruction error (MSE)
    double error = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        double diff = features[i] - reconstructed[i];
        error += diff * diff;
    }
    
    return sqrt(error / features.size());  // RMSE
}

void PhotonTriggerML::Autoencoder::Train(const std::vector<std::vector<double>>& features,
                                         double learning_rate)
{
    if (!fInitialized || features.empty()) return;
    
    const double momentum = 0.9;
    
    for (const auto& input : features) {
        if (static_cast<int>(input.size()) != fInputSize) continue;
        
        // Forward pass: encode
        std::vector<double> latent(fLatentSize);
        std::vector<double> latent_raw(fLatentSize);
        
        for (int i = 0; i < fLatentSize; ++i) {
            double sum = fEncoderBias[i];
            for (size_t j = 0; j < input.size(); ++j) {
                sum += fEncoderWeights[i][j] * input[j];
            }
            latent_raw[i] = sum;
            latent[i] = tanh(sum);
        }
        
        // Forward pass: decode
        std::vector<double> output(fInputSize);
        std::vector<double> output_raw(fInputSize);
        
        for (int i = 0; i < fInputSize; ++i) {
            double sum = fDecoderBias[i];
            for (int j = 0; j < fLatentSize; ++j) {
                sum += fDecoderWeights[i][j] * latent[j];
            }
            output_raw[i] = sum;
            output[i] = 1.0 / (1.0 + exp(-sum));
        }
        
        // Backward pass: compute gradients
        // Output layer gradients
        std::vector<double> output_grad(fInputSize);
        for (int i = 0; i < fInputSize; ++i) {
            double error = output[i] - input[i];
            output_grad[i] = error * output[i] * (1 - output[i]);  // Sigmoid derivative
        }
        
        // Update decoder weights
        for (int i = 0; i < fInputSize; ++i) {
            for (int j = 0; j < fLatentSize; ++j) {
                double grad = output_grad[i] * latent[j];
                fDecoderMomentum[i][j] = momentum * fDecoderMomentum[i][j] + (1 - momentum) * grad;
                fDecoderWeights[i][j] -= learning_rate * fDecoderMomentum[i][j];
            }
            fDecoderBias[i] -= learning_rate * output_grad[i];
        }
        
        // Hidden layer gradients
        std::vector<double> latent_grad(fLatentSize, 0);
        for (int j = 0; j < fLatentSize; ++j) {
            for (int i = 0; i < fInputSize; ++i) {
                latent_grad[j] += output_grad[i] * fDecoderWeights[i][j];
            }
            latent_grad[j] *= (1 - latent[j] * latent[j]);  // tanh derivative
        }
        
        // Update encoder weights
        for (int i = 0; i < fLatentSize; ++i) {
            for (size_t j = 0; j < input.size(); ++j) {
                double grad = latent_grad[i] * input[j];
                fEncoderMomentum[i][j] = momentum * fEncoderMomentum[i][j] + (1 - momentum) * grad;
                fEncoderWeights[i][j] -= learning_rate * fEncoderMomentum[i][j];
            }
            fEncoderBias[i] -= learning_rate * latent_grad[i];
        }
    }
}

void PhotonTriggerML::Autoencoder::UpdateThreshold(const std::vector<std::vector<double>>& hadron_features)
{
    if (hadron_features.empty()) return;
    
    // Calculate reconstruction errors for hadrons
    std::vector<double> errors;
    for (const auto& features : hadron_features) {
        errors.push_back(GetReconstructionError(features));
    }
    
    // Sort errors
    std::sort(errors.begin(), errors.end());
    
    // Set threshold at 99th percentile of hadron reconstruction errors
    // Anything with higher error is likely a photon (anomaly)
    size_t idx = static_cast<size_t>(0.99 * errors.size());
    if (idx >= errors.size()) idx = errors.size() - 1;
    fAnomalyThreshold = errors[idx];
    
    cout << "  Anomaly threshold set to: " << fAnomalyThreshold 
         << " (99th percentile of hadron errors)" << endl;
}

// ============================================================================
// PhotonTriggerML Implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fAutoencoder(std::make_unique<Autoencoder>()),
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
    fOutputFileName("photon_trigger_anomaly.root"),
    fLogFileName("photon_trigger_anomaly.log"),
    fOutputFile(nullptr),
    fMLTree(nullptr),
    hReconstructionError(nullptr),
    hErrorPhotons(nullptr),
    hErrorHadrons(nullptr),
    hConfusionMatrix(nullptr)
{
    fInstance = this;
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Constructor (ANOMALY)" << endl;
    cout << "Using autoencoder-based anomaly detection" << endl;
    cout << "==========================================" << endl;
}

PhotonTriggerML::~PhotonTriggerML()
{
    cout << "PhotonTriggerML Destructor called" << endl;
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization (ANOMALY)");
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Initialization" << endl;
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
    
    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    
    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML ANOMALY Detection Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << endl << endl;
    
    // Initialize autoencoder with reduced feature set (10 most important features)
    cout << "Initializing Autoencoder for Anomaly Detection..." << endl;
    fAutoencoder->Initialize(10);  // Using 10 key features
    
    // Create output file
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
    // Create tree
    fMLTree = new TTree("MLTree", "PhotonTriggerML Anomaly Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("reconstructionError", &fReconstructionError, "reconstructionError/D");
    fMLTree->Branch("isAnomaly", &fIsAnomaly, "isAnomaly/O");
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");
    fMLTree->Branch("primaryType", &fPrimaryType);
    
    // Create histograms
    hReconstructionError = new TH1D("hReconstructionError", "Reconstruction Error;Error;Count", 100, 0, 2);
    hErrorPhotons = new TH1D("hErrorPhotons", "Reconstruction Error (Photons);Error;Count", 100, 0, 2);
    hErrorPhotons->SetLineColor(kBlue);
    hErrorHadrons = new TH1D("hErrorHadrons", "Reconstruction Error (Hadrons);Error;Count", 100, 0, 2);
    hErrorHadrons->SetLineColor(kRed);
    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 
                                2, -0.5, 1.5, 2, -0.5, 1.5);
    
    // Register signal handler
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    
    cout << "Initialization complete!" << endl;
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
        cout << "\n┌────────┬──────────┬─────────┬─────────┬──────────┬─────────┬─────────┐" << endl;
        cout << "│ Event  │ Stations │ Anomaly%│ Accuracy│ Precision│ Recall  │ F1-Score│" << endl;
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
    
    // Train autoencoder periodically (only on hadrons to learn "normal" patterns)
    if (fEventCount % 20 == 0 && !fHadronFeatures.empty()) {
        // Sample a batch of hadron features
        int batch_size = min(64, (int)fHadronFeatures.size());
        std::vector<std::vector<double>> batch;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, fHadronFeatures.size() - 1);
        
        for (int i = 0; i < batch_size; ++i) {
            batch.push_back(fHadronFeatures[dis(gen)]);
        }
        
        // Train autoencoder
        double learning_rate = 0.001 * exp(-fEventCount / 1000.0);  // Decay
        fAutoencoder->Train(batch, learning_rate);
        
        // Update anomaly threshold
        if (fEventCount % 100 == 0) {
            fAutoencoder->UpdateThreshold(fHadronFeatures);
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
        
        // Extract simplified features (10 key features)
        std::vector<double> features = ExtractSimplifiedFeatures(trace_data);
        
        // Normalize features to [0, 1]
        for (auto& f : features) {
            f = max(0.0, min(1.0, f));
        }
        
        // Get reconstruction error from autoencoder
        fReconstructionError = fAutoencoder->GetReconstructionError(features);
        
        // Determine if it's an anomaly (potential photon)
        fIsAnomaly = (fReconstructionError > fAutoencoder->GetThreshold());
        
        // Store ML result for PMTTraceModule compatibility
        MLResult mlResult;
        mlResult.photonScore = fReconstructionError;
        mlResult.identifiedAsPhoton = fIsAnomaly;
        mlResult.isActualPhoton = fIsActualPhoton;
        mlResult.primaryType = fPrimaryType;
        mlResult.confidence = fIsAnomaly ? 
            fReconstructionError / (fAutoencoder->GetThreshold() + 0.001) : 
            (fAutoencoder->GetThreshold() - fReconstructionError) / (fAutoencoder->GetThreshold() + 0.001);
        mlResult.features = ExtractCompatibilityFeatures(trace_data);
        mlResult.vemCharge = mlResult.features.total_charge;
        
        // Store in static map
        fMLResultsMap[fStationId] = mlResult;
        
        // Store features for training
        if (!fIsActualPhoton) {
            fHadronFeatures.push_back(features);
            if (fHadronFeatures.size() > 10000) {
                fHadronFeatures.erase(fHadronFeatures.begin());
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

std::vector<double> PhotonTriggerML::ExtractSimplifiedFeatures(const std::vector<double>& trace)
{
    std::vector<double> features;
    
    // Find peak
    int peak_bin = 0;
    double peak_value = 0;
    double total_signal = 0;
    
    for (size_t i = 0; i < trace.size(); i++) {
        double val = trace[i] - 50;  // Simple baseline subtraction
        if (val > peak_value) {
            peak_value = val;
            peak_bin = i;
        }
        if (val > 0) total_signal += val;
    }
    
    // 1. Peak amplitude (normalized)
    features.push_back(peak_value / 1000.0);
    
    // 2. Total charge (normalized)
    features.push_back(total_signal / 10000.0);
    
    // 3. Peak position (normalized)
    features.push_back(peak_bin / 2048.0);
    
    // 4. Rise time (10% to 90%)
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
    features.push_back((rise_end - rise_start) / 100.0);
    
    // 5. Fall time (90% to 10%)
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
    features.push_back((fall_end - fall_start) / 100.0);
    
    // 6. Asymmetry
    double rise = rise_end - rise_start;
    double fall = fall_end - fall_start;
    features.push_back((fall - rise) / (fall + rise + 1));
    
    // 7. Early fraction
    double early_charge = 0;
    for (int i = 0; i < 512; i++) {
        double val = trace[i] - 50;
        if (val > 0) early_charge += val;
    }
    features.push_back(early_charge / (total_signal + 1));
    
    // 8. Late fraction
    double late_charge = 0;
    for (int i = 1536; i < 2048; i++) {
        double val = trace[i] - 50;
        if (val > 0) late_charge += val;
    }
    features.push_back(late_charge / (total_signal + 1));
    
    // 9. Peak to total ratio
    features.push_back(peak_value / (total_signal + 1));
    
    // 10. Signal smoothness (second derivative)
    double smoothness = 0;
    int count = 0;
    for (int i = peak_bin - 50; i < peak_bin + 50 && i > 1 && i < 2046; i++) {
        if (i < 0 || i >= 2048) continue;
        double second_deriv = trace[i+1] - 2*trace[i] + trace[i-1];
        smoothness += abs(second_deriv);
        count++;
    }
    features.push_back(smoothness / (count * 100.0 + 1));
    
    return features;
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
        double val = trace[i] - 50;  // Simple baseline subtraction
        if (val > peak_value) {
            peak_value = val;
            peak_bin = i;
        }
        if (val > 0) total_signal += val;
    }
    
    features.total_charge = total_signal / ADC_PER_VEM;
    features.peak_charge_ratio = peak_value / (total_signal + 1);
    
    // Rise time (10% to 90%)
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
    features.risetime_10_90 = (rise_end - rise_start) * 25.0; // 25 ns per bin
    
    // Fall time
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
    
    // Pulse width (FWHM)
    int half_rise = rise_start, half_fall = fall_end;
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
    double rise = rise_end - rise_start;
    double fall = fall_end - fall_start;
    features.asymmetry = (fall - rise) / (fall + rise + 1);
    
    // Kurtosis (simplified)
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
         << " │ " << setw(7) << f1 << "%│" << endl;
    
    // Log to file
    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount 
                << " - Acc: " << accuracy 
                << "% Prec: " << precision
                << "% Rec: " << recall
                << "% F1: " << f1 << endl;
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n╔══════════════════════════════════════════╗" << endl;
    cout << "║  PHOTONTRIGGERML FINAL SUMMARY (ANOMALY) ║" << endl;
    cout << "╚══════════════════════════════════════════╝" << endl;
    
    // Basic statistics
    cout << "\n┌─── Event Statistics ─────────────────────┐" << endl;
    cout << "│ Events processed:     " << setw(15) << fEventCount << " │" << endl;
    cout << "│ Stations analyzed:    " << setw(15) << fStationCount << " │" << endl;
    cout << "│ Anomaly threshold:    " << setw(15) << fixed << setprecision(4) 
         << fAutoencoder->GetThreshold() << " │" << endl;
    cout << "└──────────────────────────────────────────┘" << endl;
    
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
        
        cout << "\n┌─── Performance Metrics ──────────────────┐" << endl;
        cout << "│ Accuracy:      " << setw(21) << fixed << setprecision(2) 
             << accuracy << "% │" << endl;
        cout << "│ Precision:     " << setw(21) << precision << "% │" << endl;
        cout << "│ Recall:        " << setw(21) << recall << "% │" << endl;
        cout << "│ Specificity:   " << setw(21) << specificity << "% │" << endl;
        cout << "│ F1-Score:      " << setw(21) << f1 << "% │" << endl;
        cout << "│ MCC:           " << setw(21) << fixed << setprecision(4) 
             << mcc << "   │" << endl;
        cout << "└──────────────────────────────────────────┘" << endl;
        
        // Confusion Matrix with better formatting
        cout << "\n┌─── Confusion Matrix ─────────────────────┐" << endl;
        cout << "│                  PREDICTED                │" << endl;
        cout << "│              Normal    Anomaly            │" << endl;
        cout << "│   ┌────────┬─────────┬─────────┐         │" << endl;
        cout << "│ A │ Normal │" << setw(8) << fTrueNegatives 
             << " │" << setw(8) << fFalsePositives << " │         │" << endl;
        cout << "│ C ├────────┼─────────┼─────────┤         │" << endl;
        cout << "│ T │ Photon │" << setw(8) << fFalseNegatives 
             << " │" << setw(8) << fTruePositives << " │         │" << endl;
        cout << "│   └────────┴─────────┴─────────┘         │" << endl;
        cout << "└──────────────────────────────────────────┘" << endl;
        
        // Class distribution analysis
        int actual_photons = fTruePositives + fFalseNegatives;
        int actual_hadrons = fTrueNegatives + fFalsePositives;
        int predicted_photons = fTruePositives + fFalsePositives;
        int predicted_hadrons = fTrueNegatives + fFalseNegatives;
        
        cout << "\n┌─── Class Distribution ───────────────────┐" << endl;
        cout << "│ Actual Photons:    " << setw(10) << actual_photons 
             << " (" << fixed << setprecision(1) << setw(5) 
             << 100.0 * actual_photons / total << "%) │" << endl;
        cout << "│ Actual Hadrons:    " << setw(10) << actual_hadrons 
             << " (" << setw(5) << 100.0 * actual_hadrons / total << "%) │" << endl;
        cout << "│ Predicted Anomalies:" << setw(9) << predicted_photons 
             << " (" << setw(5) << 100.0 * predicted_photons / total << "%) │" << endl;
        cout << "│ Predicted Normal:  " << setw(10) << predicted_hadrons 
             << " (" << setw(5) << 100.0 * predicted_hadrons / total << "%) │" << endl;
        cout << "└──────────────────────────────────────────┘" << endl;
        
        // Error analysis
        if (actual_photons > 0 && actual_hadrons > 0) {
            double photon_detection_rate = 100.0 * fTruePositives / actual_photons;
            double hadron_rejection_rate = 100.0 * fTrueNegatives / actual_hadrons;
            double false_alarm_rate = 100.0 * fFalsePositives / actual_hadrons;
            
            cout << "\n┌─── Detection Analysis ───────────────────┐" << endl;
            cout << "│ Photon Detection Rate: " << setw(14) << fixed << setprecision(2) 
                 << photon_detection_rate << "% │" << endl;
            cout << "│ Hadron Rejection Rate: " << setw(14) 
                 << hadron_rejection_rate << "% │" << endl;
            cout << "│ False Alarm Rate:      " << setw(14) 
                 << false_alarm_rate << "% │" << endl;
            cout << "└──────────────────────────────────────────┘" << endl;
        }
        
        // Performance assessment
        cout << "\n┌─── Performance Assessment ───────────────┐" << endl;
        if (recall < 5) {
            cout << "│ ⚠ CRITICAL: Very low photon detection!   │" << endl;
            cout << "│   - Threshold may be too high            │" << endl;
            cout << "│   - Consider more training epochs        │" << endl;
        } else if (recall < 20) {
            cout << "│ ⚠ WARNING: Low photon detection rate     │" << endl;
            cout << "│   - May need threshold adjustment        │" << endl;
        } else if (recall < 50) {
            cout << "│ ✓ FAIR: Moderate photon detection        │" << endl;
            cout << "│   - Performance could be improved        │" << endl;
        } else {
            cout << "│ ✓ GOOD: Reasonable photon detection      │" << endl;
        }
        
        if (precision < 10) {
            cout << "│ ⚠ WARNING: Very high false positive rate │" << endl;
        } else if (precision < 30) {
            cout << "│ ⚠ CAUTION: High false positive rate      │" << endl;
        } else {
            cout << "│ ✓ GOOD: Acceptable false positive rate   │" << endl;
        }
        cout << "└──────────────────────────────────────────┘" << endl;
    } else {
        cout << "\n⚠ No events processed - unable to calculate metrics" << endl;
    }
    
    // Training statistics
    if (!fHadronFeatures.empty()) {
        cout << "\n┌─── Training Statistics ──────────────────┐" << endl;
        cout << "│ Hadron samples collected: " << setw(11) << fHadronFeatures.size() << " │" << endl;
        cout << "│ Training epochs:          " << setw(11) << (fEventCount / 20) << " │" << endl;
        cout << "└──────────────────────────────────────────┘" << endl;
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
        
        cout << "✓ Histograms written to " << fOutputFileName << endl;
        
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = nullptr;
    }
    
    // Close log file
    if (fLogFile.is_open()) {
        // Write final summary to log
        fLogFile << "\n=== FINAL SUMMARY ===" << endl;
        fLogFile << "Events: " << fEventCount << endl;
        fLogFile << "Stations: " << fStationCount << endl;
        fLogFile << "TP: " << fTruePositives << " FP: " << fFalsePositives << endl;
        fLogFile << "TN: " << fTrueNegatives << " FN: " << fFalseNegatives << endl;
        fLogFile.close();
    }
    
    cout << "\n╔══════════════════════════════════════════╗" << endl;
    cout << "║         ANALYSIS COMPLETE                ║" << endl;
    cout << "╚══════════════════════════════════════════╝" << endl;
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
