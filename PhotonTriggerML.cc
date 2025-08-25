/**
 * PhotonTriggerML.cc
 * ML-based photon trigger implementation for PhD research
 * Implements sophisticated discrimination between photon and hadronic air showers
 * 
 * Author: Khoa Nguyen
 * Supervisor: David F. Nitz
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
#include <TMath.h>

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <csignal>
#include <iomanip>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace det;
using namespace utl;

// Static instance pointer for signal handler
PhotonTriggerML* PhotonTriggerML::fInstance = 0;

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

// Constructor
PhotonTriggerML::PhotonTriggerML() :
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
    fDistance(0),
    fStationId(0),
    fIsActualPhoton(false),
    fOutputFile(nullptr),
    fMLTree(nullptr),
    hPhotonScore(nullptr),
    hRisetime(nullptr),
    hSmoothness(nullptr),
    hEarlyLateRatio(nullptr),
    hTotalCharge(nullptr),
    hScoreVsEnergy(nullptr),
    hScoreVsDistance(nullptr),
    hPhotonScorePhotons(nullptr),
    hPhotonScoreHadrons(nullptr),
    hROCCurve(nullptr),
    fTruePositives(0),
    fFalsePositives(0),
    fTrueNegatives(0),
    fFalseNegatives(0)
{
    // Features struct will be initialized with default values
    fInstance = this;  // Set static instance pointer
}

// Destructor
PhotonTriggerML::~PhotonTriggerML()
{
    // Cleanup is handled in Finish() method
    // ROOT objects are deleted when file is closed
}

// Init method
VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Initializing ML-based photon trigger");
    
    INFO("=== PhotonTriggerML Configuration ===");
    INFO("Energy range: 10^18 - 10^19 eV");
    INFO("Target: 50% efficiency improvement");
    INFO("ML discriminator: Lightweight neural network");
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
    fMLTree->Branch("primaryId", &fPrimaryId, "primaryId/I");
    fMLTree->Branch("primaryType", &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");
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
                              50, 1e16, 1e20, 100, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 
                                50, 0, 3000, 100, 0, 1);
    
    // Add performance analysis histograms
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons", "Photon Score (True Photons);Score;Count", 100, 0, 1);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons", "Photon Score (True Hadrons);Score;Count", 100, 0, 1);
    hROCCurve = new TH2D("hROCCurve", "ROC Curve;False Positive Rate;True Positive Rate", 100, 0, 1, 100, 0, 1);
    
    // Set up signal handlers
    signal(SIGINT, PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);
    INFO("Signal handlers installed - use Ctrl+C to interrupt and save partial results");
    
    INFO("PhotonTriggerML initialized successfully");
    INFO("Waiting for events...");
    
    return eSuccess;
}

// Run method
VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    
    // More frequent status updates initially
    if (fEventCount <= 20 || fEventCount % 50 == 0) {
        ostringstream msg;
        msg << "PhotonTriggerML::Run() - Event " << fEventCount 
            << " | Stations processed: " << fStationCount 
            << " | Photon-like: " << fPhotonLikeCount;
        INFO(msg.str());
    }
    
    // Get shower information
    fEnergy = 0;
    fCoreX = 0;
    fCoreY = 0;
    fPrimaryId = 0;
    fPrimaryType = "Unknown";
    fIsActualPhoton = false;
    
    bool hasShower = false;
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        hasShower = true;
        
        // Get primary particle information
        fPrimaryId = shower.GetPrimaryParticle();
        
        // Determine particle type from ID
        // PDG codes: photon=22, electron=11, proton=2212, iron=1000026056
        switch(fPrimaryId) {
            case 22:
                fPrimaryType = "photon";
                fIsActualPhoton = true;
                break;
            case 11:
            case -11:
                fPrimaryType = "electron";
                break;
            case 2212:
                fPrimaryType = "proton";
                break;
            case 1000026056:
                fPrimaryType = "iron";
                break;
            case 14:
            case -14:
                fPrimaryType = "muon";
                break;
            default:
                if (fPrimaryId > 1000000000) {
                    fPrimaryType = "nucleus";
                } else {
                    fPrimaryType = "unknown";
                }
                break;
        }
        
        // Debug: Log energy and particle type for first few events
        if (fEventCount <= 10) {
            ostringstream msg;
            msg << "  Event " << fEventCount 
                << " | Primary: " << fPrimaryType << " (ID=" << fPrimaryId << ")"
                << " | Energy: " << fEnergy/1e18 << " EeV"
                << " | IsPhoton: " << (fIsActualPhoton ? "YES" : "NO");
            INFO(msg.str());
        }
        
        // Get core position
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }
    } else {
        if (fEventCount <= 10) {
            INFO("  Event has no SimShower data");
        }
    }
    
    // Process stations
    int stationsInEvent = 0;
    
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            stationsInEvent++;
            ProcessStation(*it);
        }
        
        // Debug: Report number of stations
        if (fEventCount <= 10 || (stationsInEvent > 0 && fEventCount <= 20)) {
            ostringstream msg;
            msg << "  Event " << fEventCount << " has " << stationsInEvent 
                << " stations in SEvent";
            if (hasShower) {
                msg << " (E=" << fEnergy/1e18 << " EeV)";
            }
            INFO(msg.str());
        }
    } else {
        if (fEventCount <= 10) {
            INFO("  Event has no SEvent data - no stations to process");
        }
    }
    
    return eSuccess;
}

// Process a single station
void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();
    
    // More aggressive debugging for early stations
    static int totalStationsProcessed = 0;
    totalStationsProcessed++;
    bool doDebug = (totalStationsProcessed <= 50);
    
    // Check if station has SimData
    bool hasSimData = station.HasSimData();
    
    if (doDebug && totalStationsProcessed <= 10) {
        ostringstream msg;
        msg << "    Processing Station " << fStationId 
            << " (total #" << totalStationsProcessed << ")"
            << ": HasSimData=" << (hasSimData ? "yes" : "no");
        INFO(msg.str());
    }
    
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
        if (doDebug) {
            INFO("      Could not get station position");
        }
        return;
    }
    
    // Process PMTs
    const int firstPMT = sdet::Station::GetFirstPMTId();
    int pmtsProcessed = 0;
    
    for (int p = 0; p < 3; p++) {
        const int pmtId = p + firstPMT;
        
        if (!station.HasPMT(pmtId)) {
            if (doDebug && totalStationsProcessed <= 10 && p == 0) {
                ostringstream msg;
                msg << "      Station " << fStationId << " has no PMT " << pmtId;
                INFO(msg.str());
            }
            continue;
        }
        
        const sevt::PMT& pmt = station.GetPMT(pmtId);
        
        if (doDebug && totalStationsProcessed <= 10 && p == 0) {
            ostringstream msg;
            msg << "      Checking PMT " << pmtId;
            INFO(msg.str());
        }
        
        // Method 1: Try direct FADC trace access
        bool traceFound = false;
        vector<double> trace_data;
        
        if (pmt.HasFADCTrace()) {
            try {
                const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
                traceFound = true;
                
                // Convert to vector<double>
                for (int i = 0; i < 2048; i++) {
                    double val = 50.0;  // default baseline
                    try {
                        val = trace[i];
                    } catch (...) {
                        // Use default
                    }
                    trace_data.push_back(val);
                }
                
                if (doDebug && totalStationsProcessed <= 5 && p == 0) {
                    INFO("      -> Found FADC trace from DIRECT PMT access");
                }
            } catch (const exception& e) {
                traceFound = false;
                if (doDebug && p == 0) {
                    ostringstream msg;
                    msg << "      -> Direct FADC access failed: " << e.what();
                    INFO(msg.str());
                }
            }
        } else {
            if (doDebug && totalStationsProcessed <= 10 && p == 0) {
                INFO("      -> PMT has no direct FADC trace");
            }
        }
        
        // Method 2: If direct access failed, try simulation data
        if (!traceFound && pmt.HasSimData()) {
            if (doDebug && totalStationsProcessed <= 10 && p == 0) {
                INFO("      -> Checking simulation data for FADC trace...");
            }
            
            try {
                const sevt::PMTSimData& simData = pmt.GetSimData();
                
                // Try FADC trace from simulation with eTotal component
                if (simData.HasFADCTrace(sevt::StationConstants::eTotal)) {
                    const auto& fadcTrace = simData.GetFADCTrace(
                        sdet::PMTConstants::eHighGain,
                        sevt::StationConstants::eTotal
                    );
                    
                    traceFound = true;
                    
                    // Convert TimeDistribution to vector<double>
                    for (int i = 0; i < 2048; i++) {
                        double val = 50.0;  // default baseline
                        try {
                            val = fadcTrace[i];
                        } catch (...) {
                            // Use default
                        }
                        trace_data.push_back(val);
                    }
                    
                    if (doDebug && totalStationsProcessed <= 5 && p == 0) {
                        INFO("      -> SUCCESS: Found FADC trace from SIMULATION DATA");
                    }
                } else {
                    if (doDebug && totalStationsProcessed <= 10 && p == 0) {
                        INFO("      -> SimData has no FADC trace for eTotal");
                    }
                }
            } catch (const exception& e) {
                if (doDebug && p == 0) {
                    ostringstream msg;
                    msg << "      -> Simulation FADC access failed: " << e.what();
                    INFO(msg.str());
                }
            }
        } else if (!traceFound && !pmt.HasSimData()) {
            if (doDebug && totalStationsProcessed <= 10 && p == 0) {
                INFO("      -> PMT has no SimData");
            }
        }
        
        // If no trace found by either method, skip this PMT
        if (!traceFound) {
            if (doDebug && totalStationsProcessed <= 10 && p == 0) {
                ostringstream msg;
                msg << "      -> NO TRACE FOUND for PMT " << pmtId;
                INFO(msg.str());
            }
            continue;
        }
        
        // Now process the trace_data we found
        {
            // Check trace range
            double maxVal = 0, minVal = 1e10;
            for (const auto& val : trace_data) {
                if (val > maxVal) maxVal = val;
                if (val < minVal) minVal = val;
            }
            
            // Debug: Check if trace has any signal
            if (doDebug && totalStationsProcessed <= 10 && pmtsProcessed == 0) {
                ostringstream msg;
                msg << "      -> Trace range: [" << minVal << ", " << maxVal << "] ADC";
                INFO(msg.str());
            }
            
            // Only process if there's significant signal
            if (maxVal - minVal < 5.0) {
                if (doDebug && totalStationsProcessed <= 10 && pmtsProcessed == 0) {
                    ostringstream msg;
                    msg << "      -> Trace has no significant signal (range=" 
                        << (maxVal-minVal) << " ADC), skipping";
                    INFO(msg.str());
                }
                continue;
            }
            
            // Extract features
            fFeatures = ExtractFeatures(trace_data);
            
            // Calculate photon score
            fPhotonScore = CalculatePhotonScore(fFeatures);
            
            // Update counters and performance metrics
            fStationCount++;
            bool identifiedAsPhoton = (fPhotonScore > 0.5);
            
            if (identifiedAsPhoton) {
                fPhotonLikeCount++;
                if (fIsActualPhoton) {
                    fTruePositives++;  // Correctly identified photon
                } else {
                    fFalsePositives++; // Incorrectly identified hadron as photon
                }
            } else {
                fHadronLikeCount++;
                if (!fIsActualPhoton) {
                    fTrueNegatives++;  // Correctly identified hadron
                } else {
                    fFalseNegatives++; // Incorrectly identified photon as hadron
                }
            }
            
            // Debug: Report features for first few
            if (fStationCount <= 20 || totalStationsProcessed <= 5) {
                ostringstream msg;
                msg << "  *** ML PROCESSED: Station " << fStationId << " PMT " << pmtId
                    << " | True: " << fPrimaryType 
                    << " | Score: " << fPhotonScore
                    << " | Prediction: " << (identifiedAsPhoton ? "PHOTON" : "HADRON")
                    << " | " << (identifiedAsPhoton == fIsActualPhoton ? "CORRECT" : "WRONG")
                    << " | Charge=" << fFeatures.total_charge << " VEM";
                INFO(msg.str());
            }
            
            // Fill histograms
            hPhotonScore->Fill(fPhotonScore);
            
            // Fill separate histograms for photons and hadrons
            if (fIsActualPhoton) {
                hPhotonScorePhotons->Fill(fPhotonScore);
            } else {
                hPhotonScoreHadrons->Fill(fPhotonScore);
            }
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
            pmtsProcessed++;
        }
    }
    
    if (doDebug && totalStationsProcessed <= 10 && pmtsProcessed == 0) {
        ostringstream msg;
        msg << "    -> WARNING: Station " << fStationId << " - no PMTs processed!";
        INFO(msg.str());
    } else if (pmtsProcessed > 0 && totalStationsProcessed <= 5) {
        ostringstream msg;
        msg << "    -> Station " << fStationId << " complete: " 
            << pmtsProcessed << " PMTs processed";
        INFO(msg.str());
    }
}

// Extract features from trace
PhotonTriggerML::Features PhotonTriggerML::ExtractFeatures(const vector<double>& trace, double baseline)
{
    Features features;
    const int trace_size = trace.size();
    const double ADC_PER_VEM = 180.0;
    
    // Initialize features
    features.risetime = 0;
    features.falltime = 0;
    features.peak_charge_ratio = 0;
    features.smoothness = 0;
    features.early_late_ratio = 0;
    features.num_peaks = 0;
    features.total_charge = 0;
    features.peak_amplitude = 0;
    
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
    
    // If no significant peak, return default features
    if (peak_value < 5.0) {  // Less than 5 ADC counts above baseline
        return features;
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
        if (trace[i] <= peak_10) {
            bin_10_rise = i;
            break;
        }
    }
    features.risetime = (bin_90_rise - bin_10_rise) * 25.0;  // Convert to ns
    
    // Fall time (90% to 10%)
    int bin_90_fall = peak_bin;
    int bin_10_fall = peak_bin;
    for (int i = peak_bin; i < trace_size; i++) {
        if (trace[i] <= peak_90 && bin_90_fall == peak_bin) bin_90_fall = i;
        if (trace[i] <= peak_10) {
            bin_10_fall = i;
            break;
        }
    }
    features.falltime = (bin_10_fall - bin_90_fall) * 25.0;  // Convert to ns
    
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
    features.early_late_ratio = (late_charge > 0.01) ? early_charge / late_charge : 10.0;
    
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

// SaveAndDisplaySummary method - called on interrupt
void PhotonTriggerML::SaveAndDisplaySummary()
{
    ostringstream msg;
    msg << "\n=== PhotonTriggerML Interrupt Handler ===\n"
        << "Processed " << fEventCount << " events\n"
        << "Found " << fStationCount << " stations with valid traces\n"
        << "Photon-like: " << fPhotonLikeCount << "\n"
        << "Hadron-like: " << fHadronLikeCount;
    INFO(msg.str());
    
    // Print confusion matrix and performance metrics
    if (fStationCount > 0) {
        ostringstream perfMsg;
        perfMsg << "\n=== ML Performance Analysis ===\n"
                << "Confusion Matrix:\n"
                << "                 Predicted\n"
                << "           | Photon | Hadron |\n"
                << "Actual ----+--------+--------+\n"
                << "Photon     |  " << setw(5) << fTruePositives 
                << " |  " << setw(5) << fFalseNegatives << " |\n"
                << "Hadron     |  " << setw(5) << fFalsePositives 
                << " |  " << setw(5) << fTrueNegatives << " |\n\n";
        
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
        
        perfMsg << "Performance Metrics:\n"
                << "  Accuracy:  " << fixed << setprecision(1) << accuracy << "%\n"
                << "  Precision: " << fixed << setprecision(1) << precision << "%\n"
                << "  Recall:    " << fixed << setprecision(1) << recall << "%\n"
                << "  F1-Score:  " << fixed << setprecision(1) << f1Score << "%\n"
                << "===============================";
        
        INFO(perfMsg.str());
    }
    
    // Save all data to file
    if (fOutputFile) {
        INFO("Writing data to photon_trigger_ml.root...");
        fOutputFile->cd();
        
        // Write tree
        if (fMLTree && fMLTree->GetEntries() > 0) {
            ostringstream treeMsg;
            treeMsg << "Writing ML tree with " << fMLTree->GetEntries() << " entries";
            INFO(treeMsg.str());
            fMLTree->Write();
        } else {
            INFO("WARNING: ML tree is empty or null!");
        }
        
        // Write histograms
        if (hPhotonScore) {
            INFO("Writing histograms...");
            hPhotonScore->Write();
            hRisetime->Write();
            hSmoothness->Write();
            hEarlyLateRatio->Write();
            hTotalCharge->Write();
            hScoreVsEnergy->Write();
            hScoreVsDistance->Write();
            hPhotonScorePhotons->Write();
            hPhotonScoreHadrons->Write();
            hROCCurve->Write();
        }
        
        // Close file
        fOutputFile->Write();
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = nullptr;
        
        INFO("Data saved successfully to photon_trigger_ml.root");
    } else {
        ERROR("Output file is null - cannot save data!");
    }
}

// Finish method
VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    
    // Call the save and display method (same as interrupt handler)
    SaveAndDisplaySummary();
    
    return eSuccess;
}
