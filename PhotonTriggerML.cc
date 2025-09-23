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
#include <TProfile.h>
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
// SimpleNeuralNet Implementation
// ============================================================================

PhotonTriggerML::SimpleNeuralNet::SimpleNeuralNet() :
    fInputSize(0), fHidden1Size(32), fHidden2Size(16), fInitialized(false)
{
}

void PhotonTriggerML::SimpleNeuralNet::Initialize(int input_size)
{
    fInputSize = input_size;
    fHidden1Size = 32;
    fHidden2Size = 16;
    
    // Initialize with Xavier initialization
    std::mt19937 gen(42);
    std::normal_distribution<> dist1(0.0, sqrt(2.0 / input_size));
    std::normal_distribution<> dist2(0.0, sqrt(2.0 / fHidden1Size));
    std::normal_distribution<> dist3(0.0, sqrt(2.0 / fHidden2Size));
    
    // Layer 1: input -> hidden1
    fWeights1.resize(fHidden1Size, std::vector<double>(input_size));
    fBias1.resize(fHidden1Size);
    for (int i = 0; i < fHidden1Size; i++) {
        for (int j = 0; j < input_size; j++) {
            fWeights1[i][j] = dist1(gen);
        }
        fBias1[i] = 0.01;
    }
    
    // Layer 2: hidden1 -> hidden2
    fWeights2.resize(fHidden2Size, std::vector<double>(fHidden1Size));
    fBias2.resize(fHidden2Size);
    for (int i = 0; i < fHidden2Size; i++) {
        for (int j = 0; j < fHidden1Size; j++) {
            fWeights2[i][j] = dist2(gen);
        }
        fBias2[i] = 0.01;
    }
    
    // Layer 3: hidden2 -> output
    fWeights3.resize(1, std::vector<double>(fHidden2Size));
    fBias3.resize(1);
    for (int j = 0; j < fHidden2Size; j++) {
        fWeights3[0][j] = dist3(gen);
    }
    fBias3[0] = 0.0;
    
    fInitialized = true;
}

double PhotonTriggerML::SimpleNeuralNet::Predict(const std::vector<double>& features)
{
    if (!fInitialized) return 0.5;
    
    // Layer 1
    std::vector<double> h1(fHidden1Size);
    for (int i = 0; i < fHidden1Size; i++) {
        double sum = fBias1[i];
        for (size_t j = 0; j < features.size(); j++) {
            sum += fWeights1[i][j] * features[j];
        }
        h1[i] = relu(sum);
    }
    
    // Layer 2
    std::vector<double> h2(fHidden2Size);
    for (int i = 0; i < fHidden2Size; i++) {
        double sum = fBias2[i];
        for (int j = 0; j < fHidden1Size; j++) {
            sum += fWeights2[i][j] * h1[j];
        }
        h2[i] = relu(sum);
    }
    
    // Output layer
    double output = fBias3[0];
    for (int j = 0; j < fHidden2Size; j++) {
        output += fWeights3[0][j] * h2[j];
    }
    
    return sigmoid(output);
}

void PhotonTriggerML::SimpleNeuralNet::Train(const std::vector<double>& features, 
                                             bool is_photon, double lr)
{
    if (!fInitialized) return;
    
    // Forward pass
    std::vector<double> h1(fHidden1Size);
    std::vector<double> h1_raw(fHidden1Size);
    for (int i = 0; i < fHidden1Size; i++) {
        double sum = fBias1[i];
        for (size_t j = 0; j < features.size(); j++) {
            sum += fWeights1[i][j] * features[j];
        }
        h1_raw[i] = sum;
        h1[i] = relu(sum);
    }
    
    std::vector<double> h2(fHidden2Size);
    std::vector<double> h2_raw(fHidden2Size);
    for (int i = 0; i < fHidden2Size; i++) {
        double sum = fBias2[i];
        for (int j = 0; j < fHidden1Size; j++) {
            sum += fWeights2[i][j] * h1[j];
        }
        h2_raw[i] = sum;
        h2[i] = relu(sum);
    }
    
    double output_raw = fBias3[0];
    for (int j = 0; j < fHidden2Size; j++) {
        output_raw += fWeights3[0][j] * h2[j];
    }
    double output = sigmoid(output_raw);
    
    // Backward pass
    double target = is_photon ? 1.0 : 0.0;
    double output_grad = (output - target);
    
    // Layer 3 gradients
    std::vector<double> h2_grad(fHidden2Size);
    for (int j = 0; j < fHidden2Size; j++) {
        h2_grad[j] = output_grad * fWeights3[0][j] * reluDerivative(h2_raw[j]);
    }
    
    // Layer 2 gradients
    std::vector<double> h1_grad(fHidden1Size);
    for (int j = 0; j < fHidden1Size; j++) {
        double sum = 0;
        for (int i = 0; i < fHidden2Size; i++) {
            sum += h2_grad[i] * fWeights2[i][j];
        }
        h1_grad[j] = sum * reluDerivative(h1_raw[j]);
    }
    
    // Update weights
    // Layer 3
    fBias3[0] -= lr * output_grad;
    for (int j = 0; j < fHidden2Size; j++) {
        fWeights3[0][j] -= lr * output_grad * h2[j];
    }
    
    // Layer 2
    for (int i = 0; i < fHidden2Size; i++) {
        fBias2[i] -= lr * h2_grad[i];
        for (int j = 0; j < fHidden1Size; j++) {
            fWeights2[i][j] -= lr * h2_grad[i] * h1[j];
        }
    }
    
    // Layer 1
    for (int i = 0; i < fHidden1Size; i++) {
        fBias1[i] -= lr * h1_grad[i];
        for (size_t j = 0; j < features.size(); j++) {
            fWeights1[i][j] -= lr * h1_grad[i] * features[j];
        }
    }
}

// ============================================================================
// EnsembleClassifier Implementation
// ============================================================================

PhotonTriggerML::EnsembleClassifier::EnsembleClassifier() :
    fMuonWeight(0.5), fTimeWeight(0.3), fShapeWeight(0.2), fThreshold(0.5)
{
}

void PhotonTriggerML::EnsembleClassifier::Initialize()
{
    // Weights are physics-motivated
    fMuonWeight = 0.5;   // Muon content is the strongest discriminator
    fTimeWeight = 0.3;   // Time profile differences
    fShapeWeight = 0.2;  // Overall pulse shape
    fThreshold = 0.5;
}

double PhotonTriggerML::EnsembleClassifier::MuonContentScore(const PhysicsFeatures& f)
{
    // Photons have less muon content (lower very_late_fraction)
    double muon_score = 1.0 - f.very_late_fraction * 5.0;  // Scale factor
    muon_score -= f.muon_bump_score * 2.0;
    muon_score -= f.late_to_early_ratio * 1.5;
    
    // Sigmoid to [0,1]
    return 1.0 / (1.0 + exp(-muon_score));
}

double PhotonTriggerML::EnsembleClassifier::TimeProfileScore(const PhysicsFeatures& f)
{
    // Photons have characteristic time profiles
    double time_score = 0.0;
    
    // Photons typically have moderate rise times
    time_score += exp(-pow(f.rise_time_norm - 0.5, 2) / 0.1);
    
    // Less asymmetric pulses
    time_score -= abs(f.pulse_asymmetry) * 0.5;
    
    // More concentrated in time
    time_score -= f.time_rms_norm * 0.3;
    
    return 1.0 / (1.0 + exp(-time_score));
}

double PhotonTriggerML::EnsembleClassifier::ShapeScore(const PhysicsFeatures& f)
{
    double shape_score = 0.0;
    
    // Smoother signals (less noise)
    shape_score -= f.smoothness * 0.5;
    
    // Fewer sub-peaks
    shape_score -= f.n_sub_peaks * 0.8;
    
    // Different kurtosis
    shape_score += exp(-pow(f.time_kurtosis, 2) / 2.0);
    
    return 1.0 / (1.0 + exp(-shape_score));
}

double PhotonTriggerML::EnsembleClassifier::Predict(const PhysicsFeatures& features)
{
    double muon = MuonContentScore(features);
    double time = TimeProfileScore(features);
    double shape = ShapeScore(features);
    
    return fMuonWeight * muon + fTimeWeight * time + fShapeWeight * shape;
}

void PhotonTriggerML::EnsembleClassifier::AddTrainingSample(const PhysicsFeatures& features, 
                                                            bool is_photon)
{
    fTrainingSamples.push_back({features, is_photon});
    
    // Keep buffer size manageable
    if (fTrainingSamples.size() > 10000) {
        fTrainingSamples.erase(fTrainingSamples.begin(), 
                              fTrainingSamples.begin() + 5000);
    }
}

void PhotonTriggerML::EnsembleClassifier::UpdateThreshold(double target_efficiency)
{
    if (fPhotonScores.empty()) return;
    
    std::sort(fPhotonScores.begin(), fPhotonScores.end());
    int idx = (int)((1.0 - target_efficiency) * fPhotonScores.size());
    if (idx >= 0 && idx < (int)fPhotonScores.size()) {
        fThreshold = fPhotonScores[idx];
    }
}

// ============================================================================
// PhotonTriggerML Implementation
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fNeuralNet(std::make_unique<SimpleNeuralNet>()),
    fEnsemble(std::make_unique<EnsembleClassifier>()),
    fTrainingBuffer(10000),
    fEventCount(0),
    fStationCount(0),
    fPhotonEventCount(0),
    fHadronEventCount(0),
    fEnergy(0),
    fCoreX(0),
    fCoreY(0),
    fPrimaryId(0),
    fPrimaryType("Unknown"),
    fIsActualPhoton(false),
    fStationId(0),
    fDistance(0),
    fTruePositives(0),
    fFalsePositives(0),
    fTrueNegatives(0),
    fFalseNegatives(0),
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fPhotonThreshold(0.5),
    fTargetEfficiency(0.7),
    fUseEnsemble(true),
    fUseDynamicThreshold(true),
    fSaveFeatures(true),
    fOutputFileName("photon_trigger_ml.root"),
    fWeightsFileName("photon_trigger_weights.dat"),
    fLogFileName("photon_trigger_ml.log"),
    fOutputFile(nullptr),
    fMLTree(nullptr),
    fFeatureTree(nullptr),
    fPhotonScore(0),
    fPredictedAsPhoton(false)
{
    fInstance = this;
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Constructor (SUPERVISED)" << endl;
    cout << "Using physics-based features and ensemble" << endl;
    cout << "==========================================" << endl;
}

PhotonTriggerML::~PhotonTriggerML()
{
    cout << "PhotonTriggerML Destructor called" << endl;
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization (SUPERVISED)");
    
    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Initialization (Supervised)" << endl;
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
        topBranch.GetChild("PhotonThreshold").GetData(fPhotonThreshold);
    }
    if (topBranch.GetChild("UseEnsemble")) {
        topBranch.GetChild("UseEnsemble").GetData(fUseEnsemble);
    }
    if (topBranch.GetChild("DynamicThreshold")) {
        topBranch.GetChild("DynamicThreshold").GetData(fUseDynamicThreshold);
    }
    if (topBranch.GetChild("SaveFeatures")) {
        topBranch.GetChild("SaveFeatures").GetData(fSaveFeatures);
    }
    
    // Open log file
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    
    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML SUPERVISED Detection Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "Target Efficiency: " << fTargetEfficiency << endl;
    fLogFile << "Use Ensemble: " << fUseEnsemble << endl;
    fLogFile << "==========================================" << endl << endl;
    
    // Initialize classifiers
    cout << "Initializing classifiers..." << endl;
    fNeuralNet->Initialize(18);  // 18 physics features
    fEnsemble->Initialize();
    
    // Create output file
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
    // Create trees
    fMLTree = new TTree("MLTree", "PhotonTriggerML Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("photonScore", &fPhotonScore, "photonScore/D");
    fMLTree->Branch("predictedAsPhoton", &fPredictedAsPhoton, "predictedAsPhoton/O");
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");
    fMLTree->Branch("primaryType", &fPrimaryType);
    
    if (fSaveFeatures) {
        fFeatureTree = new TTree("FeatureTree", "Physics Features");
        fFeatureTree->Branch("features", &fCurrentFeatures, 
            "rise_time_norm/D:fall_time_norm/D:pulse_asymmetry/D:fwhm_norm/D:"
            "very_early_fraction/D:early_fraction/D:peak_fraction/D:"
            "late_fraction/D:very_late_fraction/D:time_rms_norm/D:"
            "time_skewness/D:time_kurtosis/D:smoothness/D:"
            "muon_bump_score/D:n_sub_peaks/D:late_to_early_ratio/D:snr_estimate/D");
        fFeatureTree->Branch("isPhoton", &fIsActualPhoton, "isPhoton/O");
    }
    
    // Initialize histograms
    InitializeHistograms();
    
    // Register signal handler
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    
    cout << "Initialization complete!" << endl;
    cout << "  Classifier: " << (fUseEnsemble ? "Ensemble" : "Neural Network") << endl;
    cout << "  Dynamic Threshold: " << (fUseDynamicThreshold ? "Enabled" : "Disabled") << endl;
    cout << "  Initial Threshold: " << fPhotonThreshold << endl;
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
        cout << "\n┌────────┬──────────┬──────────┬─────────┬──────────┬─────────┬─────────┐" << endl;
        cout << "│ Event  │ Stations │ Photons  │ Hadrons │ Precision│ Recall  │ F1-Score│" << endl;
        cout << "├────────┼──────────┼──────────┼─────────┼──────────┼─────────┼─────────┤" << endl;
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
            case 22: 
                fPrimaryType = "photon"; 
                fIsActualPhoton = true;
                fPhotonEventCount++;
                break;
            case 2212: 
                fPrimaryType = "proton";
                fHadronEventCount++;
                break;
            case 1000026056: 
                fPrimaryType = "iron";
                fHadronEventCount++;
                break;
            default: 
                fPrimaryType = "hadron";
                fHadronEventCount++;
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
    
    // Train classifiers periodically with balanced batches
    if (fEventCount % 20 == 0 && fTrainingBuffer.PhotonCount() > 10) {
        TrainClassifiers();
    }
    
    // Update dynamic threshold
    if (fUseDynamicThreshold && fEventCount % 50 == 0) {
        UpdateDynamicThreshold();
    }
    
    // Display metrics
    if (fEventCount % 10 == 0) {
        int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total > 0) {
            double precision = (fTruePositives + fFalsePositives > 0) ? 
                             100.0 * fTruePositives / (fTruePositives + fFalsePositives) : 0;
            double recall = (fTruePositives + fFalseNegatives > 0) ? 
                          100.0 * fTruePositives / (fTruePositives + fFalseNegatives) : 0;
            double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;
            
            cout << "│ " << setw(6) << fEventCount 
                 << " │ " << setw(8) << fStationCount
                 << " │ " << setw(8) << fPhotonEventCount
                 << " │ " << setw(7) << fHadronEventCount
                 << " │ " << fixed << setprecision(1) << setw(7) << precision << "%"
                 << " │ " << setw(6) << recall << "%"
                 << " │ " << setw(6) << f1 << "%│" << endl;
            
            // Log to file
            if (fLogFile.is_open()) {
                fLogFile << "Event " << fEventCount 
                        << " - Prec: " << precision 
                        << "% Rec: " << recall
                        << "% F1: " << f1 
                        << " Threshold: " << fPhotonThreshold << endl;
            }
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
        
        // Extract physics features
        fCurrentFeatures = ExtractPhysicsFeatures(trace_data);
        
        // Convert to vector for neural net
        std::vector<double> feature_vec = {
            fCurrentFeatures.rise_time_norm,
            fCurrentFeatures.fall_time_norm,
            fCurrentFeatures.pulse_asymmetry,
            fCurrentFeatures.fwhm_norm,
            fCurrentFeatures.very_early_fraction,
            fCurrentFeatures.early_fraction,
            fCurrentFeatures.peak_fraction,
            fCurrentFeatures.late_fraction,
            fCurrentFeatures.very_late_fraction,
            fCurrentFeatures.time_rms_norm,
            fCurrentFeatures.time_skewness,
            fCurrentFeatures.time_kurtosis,
            fCurrentFeatures.smoothness,
            fCurrentFeatures.muon_bump_score,
            fCurrentFeatures.n_sub_peaks,
            fCurrentFeatures.late_to_early_ratio,
            fCurrentFeatures.snr_estimate
        };
        
        // Get prediction
        if (fUseEnsemble) {
            fPhotonScore = fEnsemble->Predict(fCurrentFeatures);
        } else {
            fPhotonScore = fNeuralNet->Predict(feature_vec);
        }
        
        fPredictedAsPhoton = (fPhotonScore > fPhotonThreshold);
        
        // Store for training
        fTrainingBuffer.Add(feature_vec, fIsActualPhoton);
        fEnsemble->AddTrainingSample(fCurrentFeatures, fIsActualPhoton);
        
        // Update performance metrics
        UpdatePerformanceMetrics(fIsActualPhoton, fPredictedAsPhoton);
        
        // Store ML result for PMTTraceModule
        MLResult mlResult;
        mlResult.photonScore = fPhotonScore;
        mlResult.identifiedAsPhoton = fPredictedAsPhoton;
        mlResult.isActualPhoton = fIsActualPhoton;
        mlResult.primaryType = fPrimaryType;
        mlResult.confidence = fPredictedAsPhoton ? fPhotonScore : (1.0 - fPhotonScore);
        
        // Fill compatibility features
        const double ADC_PER_VEM = 180.0;
        double total_charge = 0;
        for (double val : trace_data) {
            if (val > 50) total_charge += (val - 50);
        }
        mlResult.vemCharge = total_charge / ADC_PER_VEM;
        mlResult.features.total_charge = total_charge / ADC_PER_VEM;
        mlResult.features.risetime_10_90 = fCurrentFeatures.rise_time_norm * 100;
        mlResult.features.pulse_width = fCurrentFeatures.fwhm_norm * 100;
        mlResult.features.asymmetry = fCurrentFeatures.pulse_asymmetry;
        mlResult.features.peak_charge_ratio = fCurrentFeatures.peak_fraction;
        mlResult.features.kurtosis = fCurrentFeatures.time_kurtosis;
        mlResult.features.muon_fraction = fCurrentFeatures.very_late_fraction;
        
        fMLResultsMap[fStationId] = mlResult;
        
        fStationCount++;
        
        // Fill histograms
        FillHistograms();
        
        // Fill trees
        fMLTree->Fill();
        if (fSaveFeatures && fFeatureTree) {
            fFeatureTree->Fill();
        }
    }
}

PhotonTriggerML::PhysicsFeatures PhotonTriggerML::ExtractPhysicsFeatures(const std::vector<double>& trace)
{
    PhysicsFeatures features;
    const double baseline = 50.0;
    const int trace_size = trace.size();
    
    // Find peak and basic statistics
    int peak_bin = 0;
    double peak_value = 0;
    double total_signal = 0;
    
    for (int i = 0; i < trace_size; i++) {
        double val = trace[i] - baseline;
        if (val > peak_value) {
            peak_value = val;
            peak_bin = i;
        }
        if (val > 0) total_signal += val;
    }
    
    if (peak_value <= 0 || total_signal <= 0) {
        return features;  // Return zeros
    }
    
    // Normalize trace by peak to remove energy dependence
    std::vector<double> norm_trace(trace_size);
    for (int i = 0; i < trace_size; i++) {
        norm_trace[i] = (trace[i] - baseline) / peak_value;
    }
    
    // 1. Rise time (10% to 90%) - NORMALIZED
    int rise_10 = peak_bin, rise_90 = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (norm_trace[i] < 0.9 && rise_90 == peak_bin) rise_90 = i;
        if (norm_trace[i] < 0.1) {
            rise_10 = i;
            break;
        }
    }
    features.rise_time_norm = (rise_90 - rise_10) / 100.0;
    
    // 2. Fall time (90% to 10%) - NORMALIZED
    int fall_90 = peak_bin, fall_10 = peak_bin;
    for (int i = peak_bin; i < trace_size; i++) {
        if (norm_trace[i] < 0.9 && fall_90 == peak_bin) fall_90 = i;
        if (norm_trace[i] < 0.1) {
            fall_10 = i;
            break;
        }
    }
    features.fall_time_norm = (fall_10 - fall_90) / 100.0;
    
    // 3. Asymmetry
    double rise_time = rise_90 - rise_10;
    double fall_time = fall_10 - fall_90;
    features.pulse_asymmetry = (fall_time - rise_time) / (fall_time + rise_time + 0.001);
    
    // 4. FWHM
    int half_rise = rise_10, half_fall = fall_10;
    for (int i = rise_10; i <= peak_bin; i++) {
        if (norm_trace[i] >= 0.5) {
            half_rise = i;
            break;
        }
    }
    for (int i = peak_bin; i < trace_size; i++) {
        if (norm_trace[i] <= 0.5) {
            half_fall = i;
            break;
        }
    }
    features.fwhm_norm = (half_fall - half_rise) / 100.0;
    
    // 5-9. Charge distribution (CRITICAL FOR MUON DETECTION)
    double very_early = 0, early = 0, peak_region = 0, late = 0, very_late = 0;
    
    for (int i = 0; i < trace_size; i++) {
        if (norm_trace[i] > 0) {
            int rel_pos = i - peak_bin;
            
            if (rel_pos < -500) very_early += norm_trace[i];
            else if (rel_pos < -200) early += norm_trace[i];
            else if (rel_pos < 200) peak_region += norm_trace[i];
            else if (rel_pos < 500) late += norm_trace[i];
            else very_late += norm_trace[i];  // MUONS APPEAR HERE
        }
    }
    
    double charge_sum = very_early + early + peak_region + late + very_late + 0.001;
    features.very_early_fraction = very_early / charge_sum;
    features.early_fraction = early / charge_sum;
    features.peak_fraction = peak_region / charge_sum;
    features.late_fraction = late / charge_sum;
    features.very_late_fraction = very_late / charge_sum;  // KEY DISCRIMINATOR
    
    // 10. Late to early ratio (muon indicator)
    features.late_to_early_ratio = very_late / (early + 0.001);
    
    // 11-13. Statistical moments
    double mean_time = 0;
    for (int i = 0; i < trace_size; i++) {
        if (norm_trace[i] > 0) {
            mean_time += i * norm_trace[i] / charge_sum;
        }
    }
    
    double variance = 0, skewness = 0, kurtosis = 0;
    for (int i = 0; i < trace_size; i++) {
        if (norm_trace[i] > 0) {
            double diff = i - mean_time;
            double weight = norm_trace[i] / charge_sum;
            variance += diff * diff * weight;
            skewness += diff * diff * diff * weight;
            kurtosis += diff * diff * diff * diff * weight;
        }
    }
    
    double std_dev = sqrt(variance + 0.001);
    features.time_rms_norm = std_dev / 1000.0;
    features.time_skewness = skewness / (std_dev * std_dev * std_dev + 0.001);
    features.time_kurtosis = (kurtosis / (variance * variance + 0.001)) - 3.0;
    
    // 14. Smoothness
    double smoothness = 0;
    int smooth_count = 0;
    for (int i = max(1, peak_bin - 100); i < min(trace_size - 1, peak_bin + 100); i++) {
        double second_deriv = norm_trace[i+1] - 2*norm_trace[i] + norm_trace[i-1];
        smoothness += abs(second_deriv);
        smooth_count++;
    }
    features.smoothness = smoothness / (smooth_count + 1);
    
    // 15. Muon bump detection
    double muon_bump = 0;
    for (int i = peak_bin + 600; i < min(trace_size - 50, peak_bin + 1500); i++) {
        // Look for secondary bump characteristic of muons
        if (norm_trace[i] > norm_trace[i-1] && 
            norm_trace[i] > norm_trace[i+1] && 
            norm_trace[i] > 0.05) {
            muon_bump += norm_trace[i];
        }
    }
    features.muon_bump_score = muon_bump;
    
    // 16. Number of sub-peaks
    int n_peaks = 0;
    for (int i = 10; i < trace_size - 10; i++) {
        if (abs(i - peak_bin) < 100) continue;  // Skip main peak
        if (norm_trace[i] > norm_trace[i-1] && 
            norm_trace[i] > norm_trace[i+1] && 
            norm_trace[i] > 0.1) {
            n_peaks++;
        }
    }
    features.n_sub_peaks = n_peaks;
    
    // 17. Signal-to-noise
    double noise_rms = 0;
    for (int i = 0; i < 100; i++) {
        noise_rms += (trace[i] - baseline) * (trace[i] - baseline);
    }
    noise_rms = sqrt(noise_rms / 100.0);
    features.snr_estimate = peak_value / (noise_rms + 1.0);
    
    return features;
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

void PhotonTriggerML::UpdatePerformanceMetrics(bool is_photon, bool predicted_photon)
{
    if (is_photon && predicted_photon) {
        fTruePositives++;
    } else if (is_photon && !predicted_photon) {
        fFalseNegatives++;
    } else if (!is_photon && predicted_photon) {
        fFalsePositives++;
    } else {
        fTrueNegatives++;
    }
    
    // Update moving averages
    if (fTruePositives + fFalsePositives > 0) {
        double precision = (double)fTruePositives / (fTruePositives + fFalsePositives);
        fRecentPrecisions.push_back(precision);
        if (fRecentPrecisions.size() > fPerformanceWindow) {
            fRecentPrecisions.pop_front();
        }
    }
    
    if (fTruePositives + fFalseNegatives > 0) {
        double recall = (double)fTruePositives / (fTruePositives + fFalseNegatives);
        fRecentRecalls.push_back(recall);
        if (fRecentRecalls.size() > fPerformanceWindow) {
            fRecentRecalls.pop_front();
        }
    }
}

void PhotonTriggerML::TrainClassifiers()
{
    // Create balanced batch for training
    std::vector<std::pair<std::vector<double>, bool>> batch;
    
    size_t photon_count = 0;
    size_t hadron_count = 0;
    size_t batch_size = 64;
    
    for (const auto& sample : fTrainingBuffer.samples) {
        if (sample.second && photon_count < batch_size/2) {
            batch.push_back(sample);
            photon_count++;
        } else if (!sample.second && hadron_count < batch_size/2) {
            batch.push_back(sample);
            hadron_count++;
        }
        
        if (batch.size() >= batch_size) break;
    }
    
    // Shuffle batch
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(batch.begin(), batch.end(), gen);
    
    // Train neural network
    double learning_rate = 0.001 * exp(-fEventCount / 5000.0);  // Decay
    for (const auto& [features, is_photon] : batch) {
        fNeuralNet->Train(features, is_photon, learning_rate);
    }
}

void PhotonTriggerML::UpdateDynamicThreshold()
{
    // Collect recent scores
    std::vector<double> photon_scores;
    std::vector<double> hadron_scores;
    
    for (const auto& [features, is_photon] : fTrainingBuffer.samples) {
        double score = fUseEnsemble ? 
            fEnsemble->Predict(ExtractPhysicsFeatures(features)) :
            fNeuralNet->Predict(features);
            
        if (is_photon) {
            photon_scores.push_back(score);
        } else {
            hadron_scores.push_back(score);
        }
    }
    
    if (!photon_scores.empty()) {
        double new_threshold = CalculateOptimalThreshold(fTargetEfficiency);
        
        // Smooth transition
        fPhotonThreshold = 0.9 * fPhotonThreshold + 0.1 * new_threshold;
        
        cout << "  Threshold updated: " << fPhotonThreshold << endl;
    }
}

double PhotonTriggerML::CalculateOptimalThreshold(double target_efficiency)
{
    std::vector<double> photon_scores;
    
    for (const auto& [features, is_photon] : fTrainingBuffer.samples) {
        if (is_photon) {
            double score = fUseEnsemble ? 
                fEnsemble->Predict(ExtractPhysicsFeatures(features)) :
                fNeuralNet->Predict(features);
            photon_scores.push_back(score);
        }
    }
    
    if (photon_scores.empty()) return 0.5;
    
    std::sort(photon_scores.begin(), photon_scores.end());
    int idx = (int)((1.0 - target_efficiency) * photon_scores.size());
    
    return photon_scores[max(0, min(idx, (int)photon_scores.size() - 1))];
}

void PhotonTriggerML::InitializeHistograms()
{
    // Score distributions
    hPhotonScore = new TH1D("hPhotonScore", "Photon Score;Score;Count", 100, 0, 1);
    hPhotonScoreTrue = new TH1D("hPhotonScoreTrue", "Score (True Photons);Score;Count", 100, 0, 1);
    hPhotonScoreFalse = new TH1D("hPhotonScoreFalse", "Score (Hadrons);Score;Count", 100, 0, 1);
    
    // 2D correlations
    hScoreVsEnergy = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score", 
                              50, log10(fEnergyMin), log10(fEnergyMax), 50, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 
                                50, 0, 3000, 50, 0, 1);
    
    // Muon fraction (key discriminator)
    hMuonFraction = new TH1D("hMuonFraction", "Muon Fraction;Very Late Fraction;Count", 50, 0, 0.5);
    hMuonFractionPhoton = new TH1D("hMuonFractionPhoton", "Muon Fraction (Photons);Very Late Fraction;Count", 
                                   50, 0, 0.5);
    hMuonFractionHadron = new TH1D("hMuonFractionHadron", "Muon Fraction (Hadrons);Very Late Fraction;Count", 
                                   50, 0, 0.5);
    
    // Performance vs energy
    hEfficiencyVsEnergy = new TProfile("hEfficiencyVsEnergy", "Efficiency vs Energy;log10(E/eV);Efficiency", 
                                       20, log10(fEnergyMin), log10(fEnergyMax));
    hPurityVsEnergy = new TProfile("hPurityVsEnergy", "Purity vs Energy;log10(E/eV);Purity", 
                                   20, log10(fEnergyMin), log10(fEnergyMax));
    
    // Confusion matrix
    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 
                                2, -0.5, 1.5, 2, -0.5, 1.5);
    
    // Feature importance
    hFeatureImportance = new TH1D("hFeatureImportance", "Feature Importance;Feature;Weight", 18, 0, 18);
    hFeatureImportance->GetXaxis()->SetBinLabel(1, "Rise Time");
    hFeatureImportance->GetXaxis()->SetBinLabel(2, "Fall Time");
    hFeatureImportance->GetXaxis()->SetBinLabel(3, "Asymmetry");
    hFeatureImportance->GetXaxis()->SetBinLabel(4, "FWHM");
    hFeatureImportance->GetXaxis()->SetBinLabel(5, "Very Early");
    hFeatureImportance->GetXaxis()->SetBinLabel(6, "Early");
    hFeatureImportance->GetXaxis()->SetBinLabel(7, "Peak");
    hFeatureImportance->GetXaxis()->SetBinLabel(8, "Late");
    hFeatureImportance->GetXaxis()->SetBinLabel(9, "Very Late");
    hFeatureImportance->GetXaxis()->SetBinLabel(10, "Time RMS");
    hFeatureImportance->GetXaxis()->SetBinLabel(11, "Skewness");
    hFeatureImportance->GetXaxis()->SetBinLabel(12, "Kurtosis");
    hFeatureImportance->GetXaxis()->SetBinLabel(13, "Smoothness");
    hFeatureImportance->GetXaxis()->SetBinLabel(14, "Muon Bump");
    hFeatureImportance->GetXaxis()->SetBinLabel(15, "N Peaks");
    hFeatureImportance->GetXaxis()->SetBinLabel(16, "Late/Early");
    hFeatureImportance->GetXaxis()->SetBinLabel(17, "SNR");
    
    // Set colors
    hPhotonScoreTrue->SetLineColor(kBlue);
    hPhotonScoreFalse->SetLineColor(kRed);
    hMuonFractionPhoton->SetLineColor(kBlue);
    hMuonFractionHadron->SetLineColor(kRed);
}

void PhotonTriggerML::FillHistograms()
{
    hPhotonScore->Fill(fPhotonScore);
    
    if (fIsActualPhoton) {
        hPhotonScoreTrue->Fill(fPhotonScore);
        hMuonFractionPhoton->Fill(fCurrentFeatures.very_late_fraction);
    } else {
        hPhotonScoreFalse->Fill(fPhotonScore);
        hMuonFractionHadron->Fill(fCurrentFeatures.very_late_fraction);
    }
    
    hMuonFraction->Fill(fCurrentFeatures.very_late_fraction);
    
    if (fEnergy > 0) {
        hScoreVsEnergy->Fill(log10(fEnergy), fPhotonScore);
        hEfficiencyVsEnergy->Fill(log10(fEnergy), fIsActualPhoton && fPredictedAsPhoton);
        hPurityVsEnergy->Fill(log10(fEnergy), fPredictedAsPhoton && fIsActualPhoton);
    }
    
    if (fDistance > 0) {
        hScoreVsDistance->Fill(fDistance, fPhotonScore);
    }
    
    // Update confusion matrix
    int actual = fIsActualPhoton ? 1 : 0;
    int predicted = fPredictedAsPhoton ? 1 : 0;
    hConfusionMatrix->Fill(predicted, actual);
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n╔════════════════════════════════════════════╗" << endl;
    cout << "║  PHOTONTRIGGERML FINAL SUMMARY (SUPERVISED) ║" << endl;
    cout << "╚════════════════════════════════════════════╝" << endl;
    
    // Basic statistics
    cout << "\n┌─── Event Statistics ─────────────────────────┐" << endl;
    cout << "│ Events processed:     " << setw(15) << fEventCount << " │" << endl;
    cout << "│ Stations analyzed:    " << setw(15) << fStationCount << " │" << endl;
    cout << "│ Photon events:        " << setw(15) << fPhotonEventCount << " │" << endl;
    cout << "│ Hadron events:        " << setw(15) << fHadronEventCount << " │" << endl;
    cout << "│ Final threshold:      " << setw(15) << fixed << setprecision(4) 
         << fPhotonThreshold << " │" << endl;
    cout << "└──────────────────────────────────────────────┘" << endl;
    
    // Performance metrics
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
        
        cout << "\n┌─── Performance Metrics ──────────────────────┐" << endl;
        cout << "│ Accuracy:      " << setw(21) << fixed << setprecision(2) 
             << accuracy << "% │" << endl;
        cout << "│ Precision:     " << setw(21) << precision << "% │" << endl;
        cout << "│ Recall:        " << setw(21) << recall << "% │" << endl;
        cout << "│ Specificity:   " << setw(21) << specificity << "% │" << endl;
        cout << "│ F1-Score:      " << setw(21) << f1 << "% │" << endl;
        cout << "└──────────────────────────────────────────────┘" << endl;
        
        // Confusion Matrix
        cout << "\n┌─── Confusion Matrix ─────────────────────────┐" << endl;
        cout << "│                  PREDICTED                  │" << endl;
        cout << "│              Hadron    Photon               │" << endl;
        cout << "│   ┌─────────┬─────────┬─────────┐           │" << endl;
        cout << "│ A │ Hadron  │" << setw(8) << fTrueNegatives 
             << " │" << setw(8) << fFalsePositives << " │           │" << endl;
        cout << "│ C ├─────────┼─────────┼─────────┤           │" << endl;
        cout << "│ T │ Photon  │" << setw(8) << fFalseNegatives 
             << " │" << setw(8) << fTruePositives << " │           │" << endl;
        cout << "│   └─────────┴─────────┴─────────┘           │" << endl;
        cout << "└──────────────────────────────────────────────┘" << endl;
    }
    
    // Save ROOT file
    if (fOutputFile) {
        fOutputFile->cd();
        
        if (fMLTree) {
            fMLTree->Write();
            cout << "\n✓ Wrote " << fMLTree->GetEntries() << " entries to ML tree" << endl;
        }
        
        if (fFeatureTree) {
            fFeatureTree->Write();
            cout << "✓ Wrote " << fFeatureTree->GetEntries() << " entries to feature tree" << endl;
        }
        
        // Write histograms
        hPhotonScore->Write();
        hPhotonScoreTrue->Write();
        hPhotonScoreFalse->Write();
        hScoreVsEnergy->Write();
        hScoreVsDistance->Write();
        hMuonFraction->Write();
        hMuonFractionPhoton->Write();
        hMuonFractionHadron->Write();
        hEfficiencyVsEnergy->Write();
        hPurityVsEnergy->Write();
        hConfusionMatrix->Write();
        hFeatureImportance->Write();
        
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
        fLogFile << "Final Threshold: " << fPhotonThreshold << endl;
        fLogFile.close();
    }
    
    cout << "\n╔════════════════════════════════════════════╗" << endl;
    cout << "║           ANALYSIS COMPLETE                 ║" << endl;
    cout << "╚════════════════════════════════════════════╝" << endl;
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    SaveAndDisplaySummary();
    return eSuccess;
}

// Static methods for inter-module communication
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
