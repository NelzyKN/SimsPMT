// PMTTraceModule.cc
// Complete version with proper histogram saving and ML integration

#include "PMTTraceModule.h"
#include "PhotonTriggerML.h"  // Added to access ML results

#include <fwk/CentralConfig.h>
#include <utl/ErrorLogger.h>
#include <utl/TimeDistribution.h>
#include <utl/CoordinateSystemPtr.h>

#include <evt/Event.h>
#include <evt/ShowerSimData.h>
#include <sevt/SEvent.h>
#include <sevt/Station.h>
#include <sevt/PMT.h>
#include <sevt/PMTSimData.h>
#include <sevt/StationConstants.h>
#include <sevt/StationTriggerData.h>
#include <sevt/StationSimData.h>

#include <sdet/SDetector.h>
#include <sdet/Station.h>
#include <sdet/PMTConstants.h>
#include <det/Detector.h>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TObjArray.h>
#include <TString.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TApplication.h>
#include <TROOT.h>
#include <TDirectory.h>
#include <TMath.h>
#include <TPad.h>
#include <TPaveStats.h>
#include <TStyle.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
#include <csignal>
#include <map>
#include <algorithm>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace sevt;
using namespace det;
using namespace sdet;
using namespace utl;

// Static instance pointer for signal handler
PMTTraceModule* PMTTraceModule::fInstance = 0;

// Trigger bit definitions (from sde_trigger_defs.h)
const int COMPATIBILITY_SHWR_BUF_TRIG_SB = (1 << 0);    // Simple Binary (Threshold)
const int COMPATIBILITY_SHWR_BUF_TRIG_TOT = (1 << 1);   // Time over Threshold
const int COMPATIBILITY_SHWR_BUF_TRIG_TOTD = (1 << 2);  // Time over Threshold Deconvolved
const int COMPATIBILITY_SHWR_BUF_TRIG_MOPS = (1 << 3);  // Multiplicity of Positive Steps
const int SHWR_BUF_TRIG_EM = (1 << 19);                 // EM trigger
const int SHWR_BUF_TRIG_DL = (1 << 18);                 // DL trigger

// Signal handler function
void SignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        cout << "\n\nInterrupt signal received. Saving data and displaying sample traces...\n" << endl;
        
        if (PMTTraceModule::fInstance) {
            PMTTraceModule::fInstance->SaveAndDisplayTraces();
        }
        
        exit(0);
    }
}

// Constructor
PMTTraceModule::PMTTraceModule() :
    fOutputFileName("pmt_traces_1EeV.root"),
    fEventCount(0),
    fProcessedEvents(0),
    fTracesFound(0),
    fEventId(0),
    fStationId(0),
    fPmtId(0),
    fEnergy(0),
    fZenith(0),
    fCoreX(0),
    fCoreY(0),
    fStationX(0),
    fStationY(0),
    fDistance(0),
    fPrimaryId(0),
    fPrimaryType("Unknown"),
    fIsPhoton(false),
    fMLPhotonScore(-1),
    fMLIdentifiedAsPhoton(false),
    fMLPrediction("No ML"),
    fTraceSize(0),
    fPeakBin(0),
    fPeakValue(0),
    fTotalCharge(0),
    fVEMCharge(0),
    fOutputFile(0),
    fTraceTree(0),
    hEventEnergy(0),
    hZenithAngle(0),
    hNStations(0),
    hNTracesPerEvent(0),
    hTraceLength(0),
    hPeakValue(0),
    hTotalCharge(0),
    hVEMCharge(0),
    hChargeVsDistance(0),
    hTriggerTypes(0),
    hVEMChargeMOPS(0),
    hVEMChargeToTD(0),
    hVEMChargeThreshold(0),
    hVEMChargeOther(0),
    hChargeVsDistanceMOPS(0),
    hChargeVsDistanceToTD(0),
    hParticleTypes(0),
    hVEMChargePhoton(0),
    hVEMChargeProton(0),
    hVEMChargeIron(0),
    fTraceHistograms(0),
    fMaxHistograms(500)
{
    fTraceData.reserve(2048);
    fInstance = this;
}

// Destructor
PMTTraceModule::~PMTTraceModule()
{
}

// Init method
VModule::ResultFlag PMTTraceModule::Init()
{
    INFO("PMTTraceModule::Init() - Starting initialization");
    
    // Set output filename
    fOutputFileName = "pmt_traces_1EeV.root";
    
    INFO("=== PMT Trace Extractor Configuration ===");
    INFO("FADC trace length: 2048 bins");
    INFO("Sampling rate: 40 MHz (25 ns per bin)");
    INFO("Total trace duration: 51.2 microseconds");
    INFO("Expected baseline: ~50 ADC (simulation) or ~350 ADC (real data)");
    INFO("Trigger types monitored: SB, ToT, ToTD, MOPS, EM, DL");
    INFO("Primary particle types tracked in histograms");
    INFO("Maximum trace histograms to save: " + to_string(fMaxHistograms));
    INFO("==========================================");
    
    // Create output file
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
    // Create TraceHistograms directory immediately
    TDirectory* traceDir = fOutputFile->mkdir("TraceHistograms");
    if (!traceDir) {
        ERROR("Failed to create TraceHistograms directory");
        return eFailure;
    }
    INFO("Created TraceHistograms directory in ROOT file");
    
    // Return to main directory
    fOutputFile->cd();
    
    // Create histograms
    hEventEnergy = new TH1D("hEventEnergy", "Primary Energy;E [eV];Events", 100, 1e16, 1e20);
    hZenithAngle = new TH1D("hZenithAngle", "Zenith Angle;#theta [deg];Events", 90, 0, 90);
    hNStations = new TH1D("hNStations", "Number of Triggered Stations;N;Events", 50, 0, 50);
    hNTracesPerEvent = new TH1D("hNTracesPerEvent", "Traces per Event;N;Events", 150, 0, 150);
    hTraceLength = new TH1D("hTraceLength", "Trace Length;Bins;Entries", 200, 0, 2100);
    hPeakValue = new TH1D("hPeakValue", "Peak Value;ADC;Entries", 200, 0, 1000);
    hTotalCharge = new TH1D("hTotalCharge", "Total Charge;ADC;Entries", 200, 0, 50000);
    hVEMCharge = new TH1D("hVEMCharge", "VEM Charge;VEM;Entries", 200, 0, 300);
    hChargeVsDistance = new TH2D("hChargeVsDistance", "Charge vs Distance;r [m];VEM", 
                                 50, 0, 3000, 100, 0.1, 1000);
    hChargeVsDistance->SetOption("COLZ");
    
    // Add trigger type specific histograms
    hTriggerTypes = new TH1D("hTriggerTypes", "Trigger Types;Algorithm;Count", 12, 0, 12);
    hTriggerTypes->GetXaxis()->SetBinLabel(1, "SB");
    hTriggerTypes->GetXaxis()->SetBinLabel(2, "ToTD");
    hTriggerTypes->GetXaxis()->SetBinLabel(3, "MOPS");
    hTriggerTypes->GetXaxis()->SetBinLabel(4, "ToT");
    hTriggerTypes->GetXaxis()->SetBinLabel(5, "EM");
    hTriggerTypes->GetXaxis()->SetBinLabel(6, "DL");
    hTriggerTypes->GetXaxis()->SetBinLabel(7, "T1");
    hTriggerTypes->GetXaxis()->SetBinLabel(8, "T2");
    hTriggerTypes->GetXaxis()->SetBinLabel(9, "Unknown");
    hTriggerTypes->GetXaxis()->SetBinLabel(10, "Inferred-SB");
    hTriggerTypes->GetXaxis()->SetBinLabel(11, "Inferred-ToTD");
    hTriggerTypes->GetXaxis()->SetBinLabel(12, "Inferred-MOPS");
    
    // Add particle type histogram
    hParticleTypes = new TH1D("hParticleTypes", "Primary Particle Types;Type;Count", 10, 0, 10);
    hParticleTypes->GetXaxis()->SetBinLabel(1, "Photon");
    hParticleTypes->GetXaxis()->SetBinLabel(2, "Electron");
    hParticleTypes->GetXaxis()->SetBinLabel(3, "Proton");
    hParticleTypes->GetXaxis()->SetBinLabel(4, "Iron");
    hParticleTypes->GetXaxis()->SetBinLabel(5, "Muon");
    hParticleTypes->GetXaxis()->SetBinLabel(6, "Nucleus");
    hParticleTypes->GetXaxis()->SetBinLabel(7, "Unknown");
    
    // VEM charge per trigger type
    hVEMChargeMOPS = new TH1D("hVEMChargeMOPS", "VEM Charge (MOPS trigger);VEM;Entries", 200, 0, 300);
    hVEMChargeToTD = new TH1D("hVEMChargeToTD", "VEM Charge (ToTD trigger);VEM;Entries", 200, 0, 300);
    hVEMChargeThreshold = new TH1D("hVEMChargeThreshold", "VEM Charge (SB trigger);VEM;Entries", 200, 0, 300);
    hVEMChargeOther = new TH1D("hVEMChargeOther", "VEM Charge (Other trigger);VEM;Entries", 200, 0, 300);
    
    // VEM charge per particle type
    hVEMChargePhoton = new TH1D("hVEMChargePhoton", "VEM Charge (Photon primary);VEM;Entries", 200, 0, 300);
    hVEMChargeProton = new TH1D("hVEMChargeProton", "VEM Charge (Proton primary);VEM;Entries", 200, 0, 300);
    hVEMChargeIron = new TH1D("hVEMChargeIron", "VEM Charge (Iron primary);VEM;Entries", 200, 0, 300);
    
    // Charge vs distance per trigger type
    hChargeVsDistanceMOPS = new TH2D("hChargeVsDistanceMOPS", 
                                     "Charge vs Distance (MOPS);r [m];VEM", 
                                     50, 0, 3000, 100, 0.1, 1000);
    hChargeVsDistanceMOPS->SetOption("COLZ");
    
    hChargeVsDistanceToTD = new TH2D("hChargeVsDistanceToTD", 
                                     "Charge vs Distance (ToTD);r [m];VEM", 
                                     50, 0, 3000, 100, 0.1, 1000);
    hChargeVsDistanceToTD->SetOption("COLZ");
    
    // Initialize trace histogram array
    fTraceHistograms = new TObjArray();
    fTraceHistograms->SetOwner(kTRUE);
    
    // Initialize trigger counters
    fTriggerCounts.clear();
    fParticleTypeCounts.clear();
    fTriggerCountsByParticle.clear();
    
    // Create trace tree with particle type information and ML results
    fTraceTree = new TTree("TraceTree", "PMT Traces from CORSIKA");
    fTraceTree->Branch("eventId", &fEventId, "eventId/I");
    fTraceTree->Branch("stationId", &fStationId, "stationId/I");
    fTraceTree->Branch("pmtId", &fPmtId, "pmtId/I");
    fTraceTree->Branch("energy", &fEnergy, "energy/D");
    fTraceTree->Branch("zenith", &fZenith, "zenith/D");
    fTraceTree->Branch("coreX", &fCoreX, "coreX/D");
    fTraceTree->Branch("coreY", &fCoreY, "coreY/D");
    fTraceTree->Branch("stationX", &fStationX, "stationX/D");
    fTraceTree->Branch("stationY", &fStationY, "stationY/D");
    fTraceTree->Branch("distance", &fDistance, "distance/D");
    fTraceTree->Branch("primaryId", &fPrimaryId, "primaryId/I");
    fTraceTree->Branch("primaryType", &fPrimaryType);
    fTraceTree->Branch("isPhoton", &fIsPhoton, "isPhoton/O");
    fTraceTree->Branch("mlPhotonScore", &fMLPhotonScore, "mlPhotonScore/D");
    fTraceTree->Branch("mlIdentifiedAsPhoton", &fMLIdentifiedAsPhoton, "mlIdentifiedAsPhoton/O");
    fTraceTree->Branch("mlPrediction", &fMLPrediction);
    fTraceTree->Branch("traceSize", &fTraceSize, "traceSize/I");
    fTraceTree->Branch("peakBin", &fPeakBin, "peakBin/I");
    fTraceTree->Branch("peakValue", &fPeakValue, "peakValue/D");
    fTraceTree->Branch("totalCharge", &fTotalCharge, "totalCharge/D");
    fTraceTree->Branch("vemCharge", &fVEMCharge, "vemCharge/D");
    fTraceTree->Branch("triggerType", &fTriggerType);
    fTraceTree->Branch("triggerAlgorithm", &fTriggerAlgorithm);
    fTraceTree->Branch("isT1", &fIsT1, "isT1/O");
    fTraceTree->Branch("isT2", &fIsT2, "isT2/O");
    fTraceTree->Branch("traceData", &fTraceData);
    
    ostringstream msg;
    msg << "PMTTraceModule initialized. Output: " << fOutputFileName;
    INFO(msg.str());
    
    // Set up signal handlers
    signal(SIGINT, SignalHandler);
    signal(SIGTSTP, SignalHandler);
    INFO("Signal handlers installed - use Ctrl+C to interrupt and see sample traces");
    
    return eSuccess;
}

// Extract primary particle information
void PMTTraceModule::ExtractPrimaryParticleInfo(const Event& event)
{
    // Reset to defaults
    fPrimaryId = 0;
    fPrimaryType = "Unknown";
    fIsPhoton = false;
    
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fPrimaryId = shower.GetPrimaryParticle();
        fPrimaryType = GetParticleTypeFromId(fPrimaryId);
        fIsPhoton = (fPrimaryId == 22);  // PDG code for photon
        
        // Update particle type statistics
        fParticleTypeCounts[fPrimaryType]++;
        
        // Fill particle type histogram
        if (fPrimaryType == "photon") hParticleTypes->Fill(0);
        else if (fPrimaryType == "electron") hParticleTypes->Fill(1);
        else if (fPrimaryType == "proton") hParticleTypes->Fill(2);
        else if (fPrimaryType == "iron") hParticleTypes->Fill(3);
        else if (fPrimaryType == "muon") hParticleTypes->Fill(4);
        else if (fPrimaryType == "nucleus") hParticleTypes->Fill(5);
        else hParticleTypes->Fill(6);  // Unknown
        
        if (fEventCount <= 10) {
            ostringstream msg;
            msg << "Event " << fEventCount 
                << " Primary: " << fPrimaryType 
                << " (ID=" << fPrimaryId << ")"
                << " IsPhoton: " << (fIsPhoton ? "YES" : "NO");
            INFO(msg.str());
        }
    }
}

// Get particle type string from PDG ID
string PMTTraceModule::GetParticleTypeFromId(int pdgId)
{
    switch(pdgId) {
        case 22:
            return "photon";
        case 11:
        case -11:
            return "electron";
        case 2212:
            return "proton";
        case 1000026056:
            return "iron";
        case 13:
        case -13:
            return "muon";
        default:
            if (pdgId > 1000000000) {
                return "nucleus";
            } else {
                return "unknown";
            }
    }
}

// Run method
VModule::ResultFlag PMTTraceModule::Run(Event& event)
{
    fEventCount++;
    fEventId = fEventCount;
    
    if (fEventCount % 10 == 0) {
        ostringstream msg;
        msg << "Processing event " << fEventCount << " - Found " << fTracesFound << " traces so far";
        INFO(msg.str());
    }
    
    // Get shower data if available
    fEnergy = 0;
    fZenith = 0;
    fCoreX = 0;
    fCoreY = 0;
    
    // Extract primary particle information FIRST
    ExtractPrimaryParticleInfo(event);
    
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        fZenith = shower.GetGroundParticleCoordinateSystemZenith();
        
        // Get core position if available
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }
        
        hEventEnergy->Fill(fEnergy);
        hZenithAngle->Fill(fZenith * 180.0 / M_PI);
        
        if (fEventCount <= 5) {
            ostringstream debugMsg;
            debugMsg << "Event " << fEventCount 
                    << " energy: " << fEnergy/1e18 << " EeV"
                    << " primary: " << fPrimaryType;
            INFO(debugMsg.str());
        }
        
        fProcessedEvents++;
    }
    
    // Process stations
    if (event.HasSEvent()) {
        ProcessStations(event);
    }
    
    return eSuccess;
}

// ProcessStations method
void PMTTraceModule::ProcessStations(const Event& event)
{
    const SEvent& sevent = event.GetSEvent();
    const Detector& detector = Detector::GetInstance();
    const SDetector& sdetector = detector.GetSDetector();
    
    int nStations = 0;
    int nTracesThisEvent = 0;
    
    // Loop over stations
    for (SEvent::ConstStationIterator it = sevent.StationsBegin(); 
         it != sevent.StationsEnd(); ++it) {
        
        const sevt::Station& station = *it;
        fStationId = station.GetId();
        
        // For simulations, we need stations with simulation data
        if (!station.HasSimData()) {
            continue;
        }
        
        // Initialize trigger information for this station
        fTriggerType = "Unknown";
        fTriggerAlgorithm = "Unknown";
        fIsT1 = false;
        fIsT2 = false;
        int trig_id = 0;  // Trigger bitmask
        
        // Check trigger information from simulation data
        bool hasValidTrigger = false;
        const sevt::StationSimData& sSim = station.GetSimData();
        
        // Look for trigger times in simulation data
        for (sevt::StationSimData::TriggerTimeIterator trigIt = sSim.TriggerTimesBegin();
             trigIt != sSim.TriggerTimesEnd(); ++trigIt) {
            
            const sevt::StationTriggerData& triggerData = sSim.GetTriggerData(*trigIt);
            
            // Get the trigger bitmask
            trig_id = triggerData.GetPLDTrigger();
            
            // Check trigger levels
            fIsT1 = triggerData.IsT1();
            fIsT2 = triggerData.IsT2();
            
            // Build trigger type string based on which bits are set
            std::vector<std::string> triggerTypes;
            
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_SB) {
                triggerTypes.push_back("SB");
                fTriggerCounts["SB"]++;
                fTriggerCountsByParticle[fPrimaryType]["SB"]++;
                hTriggerTypes->Fill(0);
            }
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_TOTD) {
                triggerTypes.push_back("ToTD");
                fTriggerCounts["ToTD"]++;
                fTriggerCountsByParticle[fPrimaryType]["ToTD"]++;
                hTriggerTypes->Fill(1);
            }
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_MOPS) {
                triggerTypes.push_back("MOPS");
                fTriggerCounts["MOPS"]++;
                fTriggerCountsByParticle[fPrimaryType]["MOPS"]++;
                hTriggerTypes->Fill(2);
            }
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_TOT) {
                triggerTypes.push_back("ToT");
                fTriggerCounts["ToT"]++;
                fTriggerCountsByParticle[fPrimaryType]["ToT"]++;
                hTriggerTypes->Fill(3);
            }
            if (trig_id & SHWR_BUF_TRIG_EM) {
                triggerTypes.push_back("EM");
                fTriggerCounts["EM"]++;
                fTriggerCountsByParticle[fPrimaryType]["EM"]++;
                hTriggerTypes->Fill(4);
            }
            if (trig_id & SHWR_BUF_TRIG_DL) {
                triggerTypes.push_back("DL");
                fTriggerCounts["DL"]++;
                fTriggerCountsByParticle[fPrimaryType]["DL"]++;
                hTriggerTypes->Fill(5);
            }
            
            // Create combined trigger type string
            if (!triggerTypes.empty()) {
                fTriggerType = "";
                for (size_t i = 0; i < triggerTypes.size(); ++i) {
                    if (i > 0) fTriggerType += "+";
                    fTriggerType += triggerTypes[i];
                }
                hasValidTrigger = true;
            } else if (fIsT1 || fIsT2) {
                // Has T1/T2 but no specific algorithm identified
                fTriggerType = (fIsT2 ? "T2" : "T1");
                hasValidTrigger = true;
                hTriggerTypes->Fill(fIsT2 ? 7 : 6);
            }
            
            break;  // Process only first trigger time
        }
        
        // For simulations without trigger data, process all stations with sim data
        if (!hasValidTrigger) {
            // We'll infer trigger type from signal characteristics later
            fTriggerType = "Inferred";
            hTriggerTypes->Fill(8);  // Unknown
        }
        
        nStations++;
        
        // Get ML results from PhotonTriggerML if available
        PhotonTriggerML::MLResult mlResult;
        bool hasMLResults = PhotonTriggerML::GetMLResultForStation(fStationId, mlResult);
        if (hasMLResults) {
            fMLPhotonScore = mlResult.photonScore;
            fMLIdentifiedAsPhoton = mlResult.identifiedAsPhoton;
            fMLPrediction = mlResult.identifiedAsPhoton ? "ML-Photon" : "ML-Hadron";
            
            if (fTracesFound <= 10) {
                ostringstream msg;
                msg << "Station " << fStationId 
                    << " has ML results: Score=" << fixed << setprecision(3) << fMLPhotonScore
                    << ", Prediction=" << fMLPrediction
                    << ", True=" << fPrimaryType;
                INFO(msg.str());
            }
        } else {
            // No ML results available for this station
            fMLPhotonScore = -1;
            fMLIdentifiedAsPhoton = false;
            fMLPrediction = "No ML";
        }
        
        // Get station position
        try {
            const sdet::Station& detStation = sdetector.GetStation(fStationId);
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            fStationX = detStation.GetPosition().GetX(siteCS);
            fStationY = detStation.GetPosition().GetY(siteCS);
            fDistance = sqrt(pow(fStationX - fCoreX, 2) + pow(fStationY - fCoreY, 2));
        } catch (const exception& e) {
            // Station not found in detector
            continue;
        }
        
        // Process PMTs
        int tracesThisStation = ProcessPMTs(station);
        nTracesThisEvent += tracesThisStation;
        
        if (fTracesFound <= 10 && tracesThisStation > 0) {
            ostringstream msg;
            msg << "Station " << fStationId << " processed"
                << ", Distance: " << fDistance << " m"
                << ", Traces found: " << tracesThisStation
                << ", Trigger: " << fTriggerType
                << ", Primary: " << fPrimaryType;
            INFO(msg.str());
        }
    }
    
    if (nStations > 0) {
        hNStations->Fill(nStations);
        hNTracesPerEvent->Fill(nTracesThisEvent);
    }
    
    if (fEventCount <= 10) {
        ostringstream msg;
        msg << "Event " << fEventCount << ": Processed " << nStations 
            << " stations with sim data, found " << nTracesThisEvent << " traces"
            << " (Primary: " << fPrimaryType << ")";
        INFO(msg.str());
    }
}

// ProcessPMTs method - with improved histogram saving
int PMTTraceModule::ProcessPMTs(const sevt::Station& station)
{
    int tracesFound = 0;
    const int firstPMT = sdet::Station::GetFirstPMTId();
    
    // Loop through PMTs (typically 1-3)
    for (int p = 0; p < 3; ++p) {
        const int pmtId = p + firstPMT;
        
        if (!station.HasPMT(pmtId)) {
            continue;
        }
        
        const sevt::PMT& pmt = station.GetPMT(pmtId);
        fPmtId = pmtId;
        
        // Method 1: Try to get FADC trace directly from PMT
        if (pmt.HasFADCTrace()) {
            try {
                const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
                
                // Process the trace
                fTraceData.clear();
                
                // Get the actual trace size - FADC traces should have 2048 bins
                int traceSize = 2048;
                
                // Read all trace data directly
                fTraceSize = traceSize;
                fPeakValue = 0;
                fPeakBin = 0;
                fTotalCharge = 0;
                
                // Read the full trace
                for (int i = 0; i < traceSize; i++) {
                    double value = 0;
                    try {
                        value = trace[i];
                    } catch (...) {
                        value = 50.0;  // baseline
                    }
                    fTraceData.push_back(value);
                    
                    double signal = value - 50.0;
                    if (signal > 0) {
                        fTotalCharge += signal;
                    }
                    
                    if (value > fPeakValue) {
                        fPeakValue = value;
                        fPeakBin = i;
                    }
                }
                
                // Calculate VEM
                fVEMCharge = fTotalCharge / 180.0;
                
                // Only save traces with significant signals
                if (fTotalCharge < 10.0) {
                    continue;
                }
                
                // If trigger type is still unknown/inferred, try to determine from signal
                if (fTriggerType == "Inferred" || fTriggerType == "Unknown") {
                    InferTriggerType();
                }
                
                // Create histogram with trigger type, particle type, AND ML results in title
                if (fTracesFound < fMaxHistograms) {
                    TString histName = Form("traceHist_%d", fTracesFound);
                    TString histTitle;
                    
                    if (fMLPhotonScore >= 0) {
                        // Include ML results if available
                        histTitle = Form("Event %d, Station %d, PMT %d [%s trigger] [%s primary] [ML: %.3f %s]", 
                                        fEventId, fStationId, fPmtId, 
                                        fTriggerType.c_str(), fPrimaryType.c_str(),
                                        fMLPhotonScore, fMLPrediction.c_str());
                    } else {
                        // No ML results available
                        histTitle = Form("Event %d, Station %d, PMT %d [%s trigger] [%s primary]", 
                                        fEventId, fStationId, fPmtId, 
                                        fTriggerType.c_str(), fPrimaryType.c_str());
                    }
                    
                    TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
                    traceHist->GetXaxis()->SetTitle("Time bin (25 ns)");
                    traceHist->GetYaxis()->SetTitle("ADC");
                    traceHist->SetStats(kTRUE);
                    
                    // Fill histogram
                    for (int i = 0; i < 2048; i++) {
                        traceHist->SetBinContent(i+1, fTraceData[i]);
                    }
                    
                    // Add to collection
                    fTraceHistograms->Add(traceHist);
                    
                    // IMPORTANT: Write histogram immediately to file
                    if (fOutputFile) {
                        // Save to TraceHistograms directory
                        TDirectory* traceDir = fOutputFile->GetDirectory("TraceHistograms");
                        if (traceDir) {
                            traceDir->cd();
                            traceHist->Write();
                            fOutputFile->cd();
                        }
                        
                        // Also save to main directory for quick access (first 100 only)
                        if (fTracesFound < 100) {
                            fOutputFile->cd();
                            traceHist->Write();
                        }
                    }
                }
                
                // Fill summary histograms
                hTraceLength->Fill(fTraceSize);
                hPeakValue->Fill(fPeakValue);
                hTotalCharge->Fill(fTotalCharge);
                hVEMCharge->Fill(fVEMCharge);
                
                // Fill trigger-specific histograms
                FillTriggerSpecificHistograms();
                
                // Fill particle-specific histograms
                FillParticleSpecificHistograms();
                
                if (fDistance > 0 && fVEMCharge > 0.1) {
                    hChargeVsDistance->Fill(fDistance, fVEMCharge);
                }
                
                // Save to tree
                if (fTraceTree) {
                    fTraceTree->Fill();
                }
                
                fTracesFound++;
                tracesFound++;
                
                if (fTracesFound <= 10) {
                    ostringstream msg;
                    msg << "Found FADC trace: Station " << fStationId 
                        << ", PMT " << fPmtId 
                        << ", Trigger: " << fTriggerType
                        << ", Primary: " << fPrimaryType
                        << ", Peak: " << fPeakValue << " ADC at bin " << fPeakBin
                        << ", Total charge: " << fTotalCharge 
                        << ", VEM: " << fVEMCharge;
                    if (fMLPhotonScore >= 0) {
                        msg << ", ML Score: " << fixed << setprecision(3) << fMLPhotonScore
                            << " (" << fMLPrediction << ")";
                    }
                    INFO(msg.str());
                }
                
                continue;
            } catch (const exception& e) {
                // Direct PMT access failed
                if (fEventCount <= 5) {
                    ostringstream msg;
                    msg << "Direct PMT FADC access failed: " << e.what();
                    INFO(msg.str());
                }
            }
        }
        
        // Method 2: For simulations, check simulation data
        if (pmt.HasSimData()) {
            const PMTSimData& simData = pmt.GetSimData();
            
            // Try FADC trace from simulation
            try {
                if (simData.HasFADCTrace(sevt::StationConstants::eTotal)) {
                    const auto& fadcTrace = simData.GetFADCTrace(
                        sdet::PMTConstants::eHighGain,
                        sevt::StationConstants::eTotal
                    );
                    
                    if (ProcessTimeDistribution(fadcTrace)) {
                        tracesFound++;
                        if (fTracesFound <= 5) {
                            INFO("Found FADC trace from simulation data");
                        }
                        continue;
                    }
                }
            } catch (const exception& e) {
                // Simulation FADC access failed
            }
        }
    }
    
    return tracesFound;
}

// ProcessTimeDistribution method
bool PMTTraceModule::ProcessTimeDistribution(const utl::TimeDistribution<int>& timeDist)
{
    // Extract trace data from TimeDistribution
    fTraceData.clear();
    
    // FADC traces should have 2048 bins
    const int expectedSize = 2048;
    
    // Read all bins from the TimeDistribution
    fTraceSize = expectedSize;
    fPeakValue = 0;
    fPeakBin = 0;
    fTotalCharge = 0;
    
    // Read the full trace
    for (int i = 0; i < expectedSize; i++) {
        double value = 0;
        try {
            value = timeDist[i];
        } catch (...) {
            value = 50.0;  // baseline
        }
        fTraceData.push_back(value);
        
        double signal = value - 50.0;
        if (signal > 0) {
            fTotalCharge += signal;
        }
        
        if (value > fPeakValue) {
            fPeakValue = value;
            fPeakBin = i;
        }
    }
    
    // Only process traces with significant signal
    if (fTotalCharge < 10.0) {
        return false;
    }
    
    // Calculate VEM charge
    fVEMCharge = fTotalCharge / 180.0;
    
    // If trigger type is still unknown/inferred, try to determine from signal
    if (fTriggerType == "Inferred" || fTriggerType == "Unknown") {
        InferTriggerType();
    }
    
    // Create histogram with trigger type, particle type, AND ML results
    if (fTracesFound < fMaxHistograms) {
        TString histName = Form("traceHist_%d", fTracesFound);
        TString histTitle;
        
        if (fMLPhotonScore >= 0) {
            // Include ML results if available
            histTitle = Form("Event %d, Station %d, PMT %d [%s trigger] [%s primary] [ML: %.3f %s]", 
                            fEventId, fStationId, fPmtId, 
                            fTriggerType.c_str(), fPrimaryType.c_str(),
                            fMLPhotonScore, fMLPrediction.c_str());
        } else {
            // No ML results available
            histTitle = Form("Event %d, Station %d, PMT %d [%s trigger] [%s primary]", 
                            fEventId, fStationId, fPmtId, 
                            fTriggerType.c_str(), fPrimaryType.c_str());
        }
        
        TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
        traceHist->GetXaxis()->SetTitle("Time bin (25 ns)");
        traceHist->GetYaxis()->SetTitle("ADC");
        traceHist->SetStats(kTRUE);
        
        // Fill histogram
        for (int i = 0; i < 2048; i++) {
            traceHist->SetBinContent(i+1, fTraceData[i]);
        }
        
        fTraceHistograms->Add(traceHist);
        
        // Write immediately to file
        if (fOutputFile) {
            // Save to TraceHistograms directory
            TDirectory* traceDir = fOutputFile->GetDirectory("TraceHistograms");
            if (traceDir) {
                traceDir->cd();
                traceHist->Write();
                fOutputFile->cd();
            }
            
            // Also save to main directory for quick access (first 100 only)
            if (fTracesFound < 100) {
                fOutputFile->cd();
                traceHist->Write();
            }
        }
    }
    
    // Fill summary histograms  
    hTraceLength->Fill(fTraceSize);
    hPeakValue->Fill(fPeakValue);
    hTotalCharge->Fill(fTotalCharge);
    hVEMCharge->Fill(fVEMCharge);
    
    // Fill trigger-specific histograms
    FillTriggerSpecificHistograms();
    
    // Fill particle-specific histograms
    FillParticleSpecificHistograms();
    
    if (fDistance > 0 && fVEMCharge > 0.1) {
        hChargeVsDistance->Fill(fDistance, fVEMCharge);
    }
    
    // Save to tree
    if (fTraceTree) {
        fTraceTree->Fill();
    }
    
    fTracesFound++;
    
    if (fTracesFound <= 10) {
        ostringstream msg;
        msg << "Found trace from TimeDistribution: Station " << fStationId 
            << ", PMT " << fPmtId 
            << ", Trigger: " << fTriggerType
            << ", Primary: " << fPrimaryType
            << ", Peak: " << fPeakValue << " ADC at bin " << fPeakBin
            << ", Total charge: " << fTotalCharge 
            << ", VEM: " << fVEMCharge;
        INFO(msg.str());
    }
    
    return true;
}

// Helper method to infer trigger type from signal characteristics
void PMTTraceModule::InferTriggerType()
{
    // Infer trigger type based on signal characteristics
    // These are approximations based on typical trigger behavior
    
    // Count bins above threshold
    int binsAbove3VEM = 0;
    int binsAbove1_75VEM = 0;
    double threshold3VEM = 50.0 + 3.0 * 180.0;  // baseline + 3 VEM
    double threshold1_75VEM = 50.0 + 1.75 * 180.0;  // baseline + 1.75 VEM
    
    for (int i = 0; i < fTraceSize; i++) {
        if (fTraceData[i] > threshold3VEM) binsAbove3VEM++;
        if (fTraceData[i] > threshold1_75VEM) binsAbove1_75VEM++;
    }
    
    // Determine most likely trigger type
    if (fPeakValue > 50.0 + 5.0 * 180.0) {
        // Very high peak - likely SB (Threshold) trigger
        fTriggerType = "Inferred-SB";
        hTriggerTypes->Fill(9);
    } else if (binsAbove1_75VEM > 13) {  // 13 bins ~ 325 ns
        // Sustained signal above 1.75 VEM - ToTD-like
        fTriggerType = "Inferred-ToTD";
        hTriggerTypes->Fill(10);
    } else if (fVEMCharge > 3.0 && fPeakValue > 50.0 + 2.0 * 180.0) {
        // Moderate signal with good integrated charge - MOPS-like
        fTriggerType = "Inferred-MOPS";
        hTriggerTypes->Fill(11);
    } else {
        fTriggerType = "Inferred-Low";
    }
    
    if (fTracesFound <= 10) {
        ostringstream msg;
        msg << "Inferred trigger type for Station " << fStationId 
            << ": " << fTriggerType
            << " (Peak=" << fPeakValue 
            << ", VEM=" << fVEMCharge
            << ", BinsAbove1.75VEM=" << binsAbove1_75VEM
            << ", Primary=" << fPrimaryType << ")";
        INFO(msg.str());
    }
}

// Helper method to fill trigger-specific histograms
void PMTTraceModule::FillTriggerSpecificHistograms()
{
    // Fill trigger-specific histograms based on trigger type
    // Note: A station can have multiple trigger types
    if (fTriggerType.find("MOPS") != string::npos) {
        hVEMChargeMOPS->Fill(fVEMCharge);
        if (fDistance > 0 && fVEMCharge > 0.1) {
            hChargeVsDistanceMOPS->Fill(fDistance, fVEMCharge);
        }
    }
    if (fTriggerType.find("ToTD") != string::npos) {
        hVEMChargeToTD->Fill(fVEMCharge);
        if (fDistance > 0 && fVEMCharge > 0.1) {
            hChargeVsDistanceToTD->Fill(fDistance, fVEMCharge);
        }
    }
    if (fTriggerType.find("SB") != string::npos) {
        hVEMChargeThreshold->Fill(fVEMCharge);
    }
    if (fTriggerType.find("ToT") != string::npos && 
        fTriggerType.find("ToTD") == string::npos) {  // ToT but not ToTD
        hVEMChargeOther->Fill(fVEMCharge);
    }
    if (fTriggerType == "Unknown" || fTriggerType == "T1" || fTriggerType == "T2") {
        hVEMChargeOther->Fill(fVEMCharge);
    }
}

// Fill particle-specific histograms
void PMTTraceModule::FillParticleSpecificHistograms()
{
    if (fPrimaryType == "photon") {
        hVEMChargePhoton->Fill(fVEMCharge);
    } else if (fPrimaryType == "proton") {
        hVEMChargeProton->Fill(fVEMCharge);
    } else if (fPrimaryType == "iron") {
        hVEMChargeIron->Fill(fVEMCharge);
    }
}

// SaveAndDisplayTraces method - called on interrupt
void PMTTraceModule::SaveAndDisplayTraces()
{
    ostringstream msg;
    msg << "Interrupt handler called. Found " << fTracesFound << " traces so far.";
    INFO(msg.str());
    
    // Print trigger type summary
    INFO("=== Trigger Type Summary ===");
    for (map<string, int>::const_iterator it = fTriggerCounts.begin(); 
         it != fTriggerCounts.end(); ++it) {
        ostringstream triggerMsg;
        triggerMsg << "  " << it->first << ": " << it->second << " stations";
        INFO(triggerMsg.str());
    }
    INFO("=============================");
    
    // Print particle type summary
    INFO("=== Particle Type Summary ===");
    for (map<string, int>::const_iterator it = fParticleTypeCounts.begin();
         it != fParticleTypeCounts.end(); ++it) {
        ostringstream particleMsg;
        particleMsg << "  " << it->first << ": " << it->second << " events";
        INFO(particleMsg.str());
    }
    INFO("=============================");
    
    // Print trigger counts by particle type
    INFO("=== Triggers by Particle Type ===");
    for (const auto& particleEntry : fTriggerCountsByParticle) {
        INFO("  " + particleEntry.first + ":");
        for (const auto& triggerEntry : particleEntry.second) {
            ostringstream msg;
            msg << "    " << triggerEntry.first << ": " << triggerEntry.second;
            INFO(msg.str());
        }
    }
    INFO("==================================");
    
    // Save all data to file
    if (fOutputFile) {
        fOutputFile->cd();
        
        // Write all summary histograms
        if (hEventEnergy) hEventEnergy->Write("", TObject::kOverwrite);
        if (hZenithAngle) hZenithAngle->Write("", TObject::kOverwrite);
        if (hNStations) hNStations->Write("", TObject::kOverwrite);
        if (hNTracesPerEvent) hNTracesPerEvent->Write("", TObject::kOverwrite);
        if (hTraceLength) hTraceLength->Write("", TObject::kOverwrite);
        if (hPeakValue) hPeakValue->Write("", TObject::kOverwrite);
        if (hTotalCharge) hTotalCharge->Write("", TObject::kOverwrite);
        if (hVEMCharge) hVEMCharge->Write("", TObject::kOverwrite);
        if (hChargeVsDistance) hChargeVsDistance->Write("", TObject::kOverwrite);
        if (hTriggerTypes) hTriggerTypes->Write("", TObject::kOverwrite);
        if (hParticleTypes) hParticleTypes->Write("", TObject::kOverwrite);
        
        // Write the tree
        if (fTraceTree) {
            fTraceTree->Write("", TObject::kOverwrite);
        }
        
        // Make sure all trace histograms are written
        if (fTraceHistograms && fTraceHistograms->GetEntries() > 0) {
            INFO("Writing/updating trace histograms to file...");
            
            TDirectory* traceDir = fOutputFile->GetDirectory("TraceHistograms");
            if (traceDir) {
                traceDir->cd();
                
                for (int i = 0; i < fTraceHistograms->GetEntries(); i++) {
                    TH1D* hist = (TH1D*)fTraceHistograms->At(i);
                    if (hist) {
                        hist->Write("", TObject::kOverwrite);
                    }
                }
                
                fOutputFile->cd();
            }
            
            ostringstream msg2;
            msg2 << "Updated " << fTraceHistograms->GetEntries() << " trace histograms";
            INFO(msg2.str());
        }
        
        fOutputFile->Write();
        INFO("Data saved to file.");
    }
    
    // Display sample traces if not in batch mode
    if (!gROOT->IsBatch() && fTraceHistograms && fTraceHistograms->GetEntries() > 0) {
        INFO("Displaying sample FADC traces...");
        
        // Create canvas for display
        TCanvas* c1 = new TCanvas("c1", "Sample FADC Traces", 1200, 800);
        c1->Divide(2, 2);
        
        // Select interesting traces to display
        int nHists = fTraceHistograms->GetEntries();
        int displayed = 0;
        
        // Try to find one of each particle type
        vector<string> particleTypes = {"photon", "proton", "iron", "electron"};
        for (const string& particle : particleTypes) {
            for (int i = 0; i < nHists && displayed < 4; i++) {
                TH1D* hist = (TH1D*)fTraceHistograms->At(i);
                if (hist) {
                    TString title = hist->GetTitle();
                    if (title.Contains(particle.c_str())) {
                        c1->cd(displayed + 1);
                        gPad->SetGrid();
                        
                        hist->SetLineColor(kBlue);
                        hist->SetLineWidth(2);
                        hist->Draw();
                        hist->GetXaxis()->SetRangeUser(0, 2048);
                        
                        displayed++;
                        break;
                    }
                }
            }
        }
        
        // If we didn't find all types, just display first few
        if (displayed < 4) {
            for (int i = 0; i < TMath::Min(4, nHists) && displayed < 4; i++) {
                c1->cd(displayed + 1);
                gPad->SetGrid();
                
                TH1D* hist = (TH1D*)fTraceHistograms->At(i);
                if (hist) {
                    hist->SetLineColor(kBlue);
                    hist->SetLineWidth(2);
                    hist->Draw();
                    hist->GetXaxis()->SetRangeUser(0, 2048);
                    displayed++;
                }
            }
        }
        
        c1->Update();
        
        // Save canvas
        c1->SaveAs("sample_fadc_traces.png");
        c1->SaveAs("sample_fadc_traces.pdf");
        INFO("Sample traces saved to sample_fadc_traces.png and .pdf");
        
        // Keep displayed for a moment
        gSystem->ProcessEvents();
        gSystem->Sleep(3000);
        
        delete c1;
    }
    
    // Close the output file
    if (fOutputFile) {
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = 0;
        
        ostringstream msg3;
        msg3 << "Output written to " << fOutputFileName;
        INFO(msg3.str());
        INFO("To view trace histograms: root -l " + fOutputFileName + " and browse TraceHistograms directory");
    }
}

// Finish method
VModule::ResultFlag PMTTraceModule::Finish()
{
    INFO("PMTTraceModule::Finish() - Normal completion");
    
    ostringstream msg;
    msg << "Summary:\n"
        << "  Total events: " << fEventCount << "\n"
        << "  Processed events: " << fProcessedEvents << "\n"
        << "  Traces found: " << fTracesFound << "\n"
        << "  Histograms created: " << TMath::Min(fTracesFound, fMaxHistograms);
    INFO(msg.str());
    
    // Print final trigger type summary
    INFO("=== Final Trigger Type Summary ===");
    for (map<string, int>::const_iterator it = fTriggerCounts.begin(); 
         it != fTriggerCounts.end(); ++it) {
        ostringstream triggerMsg;
        triggerMsg << "  " << it->first << ": " << it->second << " stations";
        INFO(triggerMsg.str());
    }
    INFO("===================================");
    
    // Print final particle type summary
    INFO("=== Final Particle Type Summary ===");
    for (map<string, int>::const_iterator it = fParticleTypeCounts.begin();
         it != fParticleTypeCounts.end(); ++it) {
        ostringstream particleMsg;
        particleMsg << "  " << it->first << ": " << it->second << " events";
        INFO(particleMsg.str());
    }
    INFO("====================================");
    
    if (fOutputFile) {
        fOutputFile->cd();
        
        // Write all summary histograms
        INFO("Writing summary histograms...");
        if (hEventEnergy) hEventEnergy->Write();
        if (hZenithAngle) hZenithAngle->Write();
        if (hNStations) hNStations->Write();
        if (hNTracesPerEvent) hNTracesPerEvent->Write();
        if (hTraceLength) hTraceLength->Write();
        if (hPeakValue) hPeakValue->Write();
        if (hTotalCharge) hTotalCharge->Write();
        if (hVEMCharge) hVEMCharge->Write();
        if (hChargeVsDistance) hChargeVsDistance->Write();
        if (hTriggerTypes) hTriggerTypes->Write();
        if (hParticleTypes) hParticleTypes->Write();
        if (hVEMChargeMOPS) hVEMChargeMOPS->Write();
        if (hVEMChargeToTD) hVEMChargeToTD->Write();
        if (hVEMChargeThreshold) hVEMChargeThreshold->Write();
        if (hVEMChargeOther) hVEMChargeOther->Write();
        if (hVEMChargePhoton) hVEMChargePhoton->Write();
        if (hVEMChargeProton) hVEMChargeProton->Write();
        if (hVEMChargeIron) hVEMChargeIron->Write();
        if (hChargeVsDistanceMOPS) hChargeVsDistanceMOPS->Write();
        if (hChargeVsDistanceToTD) hChargeVsDistanceToTD->Write();
        
        // Write the tree
        if (fTraceTree) {
            fTraceTree->Write();
            ostringstream treeMsg;
            treeMsg << "Wrote tree with " << fTraceTree->GetEntries() << " entries";
            INFO(treeMsg.str());
        }
        
        // Final write of trace histograms
        if (fTraceHistograms && fTraceHistograms->GetEntries() > 0) {
            INFO("Finalizing trace histograms...");
            
            // Make sure TraceHistograms directory exists and is complete
            TDirectory* traceDir = fOutputFile->GetDirectory("TraceHistograms");
            if (!traceDir) {
                traceDir = fOutputFile->mkdir("TraceHistograms");
            }
            traceDir->cd();
            
            // Write all histograms (overwrite if they exist)
            for (int i = 0; i < fTraceHistograms->GetEntries(); i++) {
                TH1D* hist = (TH1D*)fTraceHistograms->At(i);
                if (hist) {
                    hist->Write("", TObject::kOverwrite);
                }
            }
            
            ostringstream msg2;
            msg2 << "Finalized " << fTraceHistograms->GetEntries() 
                 << " trace histograms in TraceHistograms directory";
            INFO(msg2.str());
            
            // Also ensure first 100 are in main directory for quick access
            fOutputFile->cd();
            for (int i = 0; i < TMath::Min(100, fTraceHistograms->GetEntries()); i++) {
                TH1D* hist = (TH1D*)fTraceHistograms->At(i);
                if (hist) {
                    hist->Write("", TObject::kOverwrite);
                }
            }
            
            INFO("Also wrote first 100 trace histograms to main directory for quick access");
        }
        
        // Final file write and close
        fOutputFile->Write();
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = 0;
        
        ostringstream msg3;
        msg3 << "Output written to " << fOutputFileName;
        INFO(msg3.str());
        INFO("========================================");
        INFO("To view trace histograms:");
        INFO("  root -l " + fOutputFileName);
        INFO("  TBrowser b;  // Then browse to TraceHistograms directory");
        INFO("  OR use: root -l check_traces.C");
        INFO("========================================");
    }
    
    return eSuccess;
}
