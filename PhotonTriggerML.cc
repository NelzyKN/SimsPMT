//I want to kms
#include "PhotonTriggerML.h"

#include <fwk/CentralConfig.h>
#include <utl/ErrorLogger.h>
#include <evt/Event.h>
#include <evt/ShowerSimData.h>
#include <sevt/SEvent.h>
#include <sevt/Station.h>
#include <sevt/PMT.h>
#include <sdet/Station.h>
#include <sdet/PMTConstants.h>
#include <det/Detector.h>
#include <sdet/SDetector.h>
#include <utl/CoordinateSystemPtr.h>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace sevt;
using namespace det;
using namespace sdet;
using namespace utl;

// Constructor
PhotonTriggerML::PhotonTriggerML() :
    fEventCount(0),
    fStationCount(0),
    fPhotonLikeCount(0),
    fHadronLikeCount(0),
    fEnergy(0),
    fCoreX(0),
    fCoreY(0),
    fOutputFile(nullptr),
    fMLTree(nullptr),
    fPhotonScore(0),
    fDistance(0),
    fStationId(0)
{
}

// Destructor
PhotonTriggerML::~PhotonTriggerML()
{
}

// Init method
VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Initializing simplified ML photon trigger");
    
    INFO("=== PhotonTriggerML Configuration ===");
    INFO("Energy range: 10^18 - 10^19 eV");
    INFO("Target: 50% efficiency improvement");
    INFO("Simple scoring algorithm (no neural network)");
    INFO("=====================================");
    
    // Create output file
    fOutputFile = new TFile("photon_trigger_ml.root", "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
    // Create tree
    fMLTree = new TTree("MLTree", "Photon Trigger ML Analysis");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("photonScore", &fPhotonScore, "photonScore/D");
    fMLTree->Branch("risetime", &fFeatures.risetime, "risetime/D");
    fMLTree->Branch("smoothness", &fFeatures.smoothness, "smoothness/D");
    fMLTree->Branch("early_late_ratio", &fFeatures.early_late_ratio, "early_late_ratio/D");
    fMLTree->Branch("total_charge", &fFeatures.total_charge, "total_charge/D");
    fMLTree->Branch("num_peaks", &fFeatures.num_peaks, "num_peaks/I");
    
    // Create histograms
    hPhotonScore = new TH1D("hPhotonScore", "Photon Score;Score;Count", 100, 0, 1);
    hRisetime = new TH1D("hRisetime", "Rise Time;Time [ns];Count", 50, 0, 500);
    hSmoothness = new TH1D("hSmoothness", "Signal Smoothness;RMS;Count", 50, 0, 100);
    hEarlyLateRatio = new TH1D("hEarlyLateRatio", "Early/Late Ratio;Ratio;Count", 50, 0, 10);
    hTotalCharge = new TH1D("hTotalCharge", "Total Charge;VEM;Count", 100, 0, 100);
    hScoreVsEnergy = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score", 
                              50, 1e18, 1e19, 100, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 
                                50, 0, 3000, 100, 0, 1);
    
    INFO("PhotonTriggerML initialized successfully");
    
    return eSuccess;
}

// Run method
VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    
    if (fEventCount % 100 == 0) {
        ostringstream msg;
        msg << "Processing event " << fEventCount;
        INFO(msg.str());
    }
    
    // Get shower information
    fEnergy = 0;
    fCoreX = 0;
    fCoreY = 0;
    
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        
        // Get core position
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }
    }
    
    // Skip if outside energy range
    if (fEnergy < 1e18 || fEnergy > 1e19) {
        return eSuccess;
    }
    
    // Process stations
    if (event.HasSEvent()) {
        const SEvent& sevent = event.GetSEvent();
        
        for (SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }
    
    return eSuccess;
}

// Process a single station
void PhotonTriggerML::ProcessStation(const Station& station)
{
    fStationId = station.GetId();
    
    // Skip if no simulation data
    if (!station.HasSimData()) {
        return;
    }
    
    // Get station position and calculate distance
    try {
        const Detector& detector = Detector::GetInstance();
        const SDetector& sdetector = detector.GetSDetector();
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
        
        const PMT& pmt = station.GetPMT(pmtId);
        
        // Get FADC trace
        if (!pmt.HasFADCTrace()) continue;
        
        try {
            const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            
            // Convert to vector<double>
            vector<double> trace_data;
            for (int i = 0; i < 2048; i++) {
                try {
                    trace_data.push_back(trace[i]);
                } catch (...) {
                    trace_data.push_back(50.0);  // baseline
                }
            }
            
            // Extract features
            fFeatures = ExtractFeatures(trace_data);
            
            // Calculate photon score
            fPhotonScore = CalculatePhotonScore(fFeatures);
            
            // Update counters
            fStationCount++;
            if (fPhotonScore > 0.5) {
                fPhotonLikeCount++;
            } else {
                fHadronLikeCount++;
            }
            
            // Fill histograms
            hPhotonScore->Fill(fPhotonScore);
            hRisetime->Fill(fFeatures.risetime);
            hSmoothness->Fill(fFeatures.smoothness);
            hEarlyLateRatio->Fill(fFeatures.early_late_ratio);
            hTotalCharge->Fill(fFeatures.total_charge);
            hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
            if (fDistance > 0) {
                hScoreVsDistance->Fill(fDistance, fPhotonScore);
            }
            
            // Fill tree
            fMLTree->Fill();
            
        } catch (...) {
            // Skip if trace access fails
            continue;
        }
    }
}

// Extract features from trace
PhotonTriggerML::Features PhotonTriggerML::ExtractFeatures(const vector<double>& trace, double baseline)
{
    Features features;
    const int trace_size = trace.size();
    const double ADC_PER_VEM = 180.0;
    
    // Find peak
    int peak_bin = 0;
    double peak_value = 0;
    for (int i = 0; i < trace_size; i++) {
        double signal = trace[i] - baseline;
        if (signal > peak_value) {
            peak_value = signal;
            peak_bin = i;
        }
    }
    
    features.peak_amplitude = peak_value / ADC_PER_VEM;
    
    // Calculate total charge
    double total_charge = 0;
    for (const auto& val : trace) {
        double signal = val - baseline;
        if (signal > 0) total_charge += signal;
    }
    features.total_charge = total_charge / ADC_PER_VEM;
    
    // Peak to charge ratio
    features.peak_charge_ratio = (features.total_charge > 0) ? 
        features.peak_amplitude / features.total_charge : 0;
    
    // Rise time (10% to 90%)
    double peak_10 = baseline + 0.1 * peak_value;
    double peak_90 = baseline + 0.9 * peak_value;
    
    int bin_10_rise = peak_bin;
    int bin_90_rise = peak_bin;
    for (int i = peak_bin; i >= 0; i--) {
        if (trace[i] <= peak_90 && bin_90_rise == peak_bin) bin_90_rise = i;
        if (trace[i] <= peak_10 && bin_10_rise == peak_bin) bin_10_rise = i;
    }
    features.risetime = (bin_90_rise - bin_10_rise) * 25.0;  // Convert to ns
    
    // Fall time
    int bin_90_fall = peak_bin;
    int bin_10_fall = peak_bin;
    for (int i = peak_bin; i < trace_size; i++) {
        if (trace[i] <= peak_90 && bin_90_fall == peak_bin) bin_90_fall = i;
        if (trace[i] <= peak_10 && bin_10_fall == peak_bin) bin_10_fall = i;
    }
    features.falltime = (bin_10_fall - bin_90_fall) * 25.0;
    
    // Count peaks
    features.num_peaks = 0;
    double peak_threshold = baseline + 0.2 * peak_value;
    bool in_peak = false;
    for (int i = 1; i < trace_size - 1; i++) {
        if (!in_peak && trace[i] > peak_threshold) {
            if (trace[i] > trace[i-1] && trace[i] > trace[i+1]) {
                features.num_peaks++;
                in_peak = true;
            }
        } else if (in_peak && trace[i] < peak_threshold) {
            in_peak = false;
        }
    }
    
    // Early to late ratio
    double early_charge = 0, late_charge = 0;
    int mid_point = trace_size / 2;
    for (int i = 0; i < mid_point; i++) {
        double signal = trace[i] - baseline;
        if (signal > 0) early_charge += signal;
    }
    for (int i = mid_point; i < trace_size; i++) {
        double signal = trace[i] - baseline;
        if (signal > 0) late_charge += signal;
    }
    features.early_late_ratio = (late_charge > 0) ? early_charge / late_charge : 10.0;
    
    // Smoothness (RMS of second derivative)
    double sum_sq_diff = 0;
    int count = 0;
    for (int i = 1; i < trace_size - 1; i++) {
        double second_deriv = trace[i+1] - 2*trace[i] + trace[i-1];
        sum_sq_diff += second_deriv * second_deriv;
        count++;
    }
    features.smoothness = (count > 0) ? sqrt(sum_sq_diff / count) : 0;
    
    return features;
}

// Calculate photon score (simple weighted combination)
double PhotonTriggerML::CalculatePhotonScore(const Features& features)
{
    // Simple scoring based on known photon characteristics
    // Photons have: shorter rise times, smoother signals, higher early/late ratio
    
    double score = 0;
    
    // Rise time score (photons have shorter rise times)
    if (features.risetime < 200) score += 0.2;
    if (features.risetime < 150) score += 0.1;
    
    // Smoothness score (photons are smoother)
    if (features.smoothness < 20) score += 0.2;
    if (features.smoothness < 10) score += 0.1;
    
    // Early/late ratio (photons have more early light)
    if (features.early_late_ratio > 2) score += 0.15;
    if (features.early_late_ratio > 3) score += 0.1;
    
    // Few peaks (photons have fewer muon peaks)
    if (features.num_peaks <= 2) score += 0.15;
    if (features.num_peaks == 1) score += 0.1;
    
    // Normalize to 0-1
    score = min(1.0, max(0.0, score));
    
    return score;
}

// Finish method
VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Finalizing analysis");
    
    // Calculate efficiency
    double photon_fraction = (fStationCount > 0) ? 
        (double)fPhotonLikeCount / fStationCount : 0;
    
    // Print summary
    ostringstream summary;
    summary << "\n=== PhotonTriggerML Summary ===\n"
            << "Total events: " << fEventCount << "\n"
            << "Total stations: " << fStationCount << "\n"
            << "Photon-like: " << fPhotonLikeCount << " (" 
            << photon_fraction * 100 << "%)\n"
            << "Hadron-like: " << fHadronLikeCount << "\n"
            << "===============================";
    INFO(summary.str());
    
    // Save output
    if (fOutputFile) {
        fOutputFile->cd();
        fOutputFile->Write();
        fOutputFile->Close();
        delete fOutputFile;
        INFO("Results saved to photon_trigger_ml.root");
    }
    
    return eSuccess;
}
