// PMTTraceModule.cc
// Enhanced version with correct trigger type checking based on T1RatesDFN.cc

#include "PMTTraceModule.h"

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

#include <iostream>
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
    fOutputFileName("pmt_traces_01EeV.root"),
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
    
    // Hardcode the output filename
    fOutputFileName = "pmt_traces_1EeV.root";
    
    INFO("=== PMT Trace Extractor Configuration ===");
    INFO("FADC trace length: 2048 bins");
    INFO("Sampling rate: 40 MHz (25 ns per bin)");
    INFO("Total trace duration: 51.2 microseconds");
    INFO("Expected baseline: ~50 ADC (simulation) or ~350 ADC (real data)");
    INFO("Trigger types monitored: SB, ToT, ToTD, MOPS, EM, DL");
    INFO("==========================================");
    
    // Create output file
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create output file");
        return eFailure;
    }
    
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
    
    // VEM charge per trigger type
    hVEMChargeMOPS = new TH1D("hVEMChargeMOPS", "VEM Charge (MOPS trigger);VEM;Entries", 200, 0, 300);
    hVEMChargeToTD = new TH1D("hVEMChargeToTD", "VEM Charge (ToTD trigger);VEM;Entries", 200, 0, 300);
    hVEMChargeThreshold = new TH1D("hVEMChargeThreshold", "VEM Charge (SB trigger);VEM;Entries", 200, 0, 300);
    hVEMChargeOther = new TH1D("hVEMChargeOther", "VEM Charge (Other trigger);VEM;Entries", 200, 0, 300);
    
    // Charge vs distance per trigger type
    hChargeVsDistanceMOPS = new TH2D("hChargeVsDistanceMOPS", 
                                     "Charge vs Distance (MOPS);r [m];VEM", 
                                     50, 0, 3000, 100, 0.1, 1000);
    hChargeVsDistanceToTD = new TH2D("hChargeVsDistanceToTD", 
                                     "Charge vs Distance (ToTD);r [m];VEM", 
                                     50, 0, 3000, 100, 0.1, 1000);
    
    // Initialize trace histogram array
    fTraceHistograms = new TObjArray();
    fTraceHistograms->SetOwner(kTRUE);
    
    // Initialize trigger counters
    fTriggerCounts.clear();
    
    // Create trace tree
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
            debugMsg << "Event " << fEventCount << " energy: " << fEnergy/1e18 << " EeV";
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

// ProcessStations method - Corrected based on T1RatesDFN.cc
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
            
            // Debug: Log raw trigger data for first few stations
            if (fEventCount <= 5 && nStations <= 3) {
                ostringstream debugMsg;
                debugMsg << "Station " << fStationId << " trigger debug:"
                        << " HasTriggerData=yes"
                        << ", PLDTrigger=0x" << std::hex << trig_id << std::dec
                        << ", IsT1=" << (triggerData.IsT1() ? "yes" : "no")
                        << ", IsT2=" << (triggerData.IsT2() ? "yes" : "no");
                INFO(debugMsg.str());
            }
            
            // Check trigger levels
            fIsT1 = triggerData.IsT1();
            fIsT2 = triggerData.IsT2();
            
            // Build trigger type string based on which bits are set
            std::vector<std::string> triggerTypes;
            
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_SB) {
                triggerTypes.push_back("SB");
                fTriggerCounts["SB"]++;
                hTriggerTypes->Fill(0);
            }
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_TOTD) {
                triggerTypes.push_back("ToTD");
                fTriggerCounts["ToTD"]++;
                hTriggerTypes->Fill(1);
            }
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_MOPS) {
                triggerTypes.push_back("MOPS");
                fTriggerCounts["MOPS"]++;
                hTriggerTypes->Fill(2);
            }
            if (trig_id & COMPATIBILITY_SHWR_BUF_TRIG_TOT) {
                triggerTypes.push_back("ToT");
                fTriggerCounts["ToT"]++;
                hTriggerTypes->Fill(3);
            }
            if (trig_id & SHWR_BUF_TRIG_EM) {
                triggerTypes.push_back("EM");
                fTriggerCounts["EM"]++;
                hTriggerTypes->Fill(4);
            }
            if (trig_id & SHWR_BUF_TRIG_DL) {
                triggerTypes.push_back("DL");
                fTriggerCounts["DL"]++;
                hTriggerTypes->Fill(5);
            }
            
            // Also check the shifted bits for delayed triggers
            const int shifted_trig = trig_id >> 8;
            if ((shifted_trig & COMPATIBILITY_SHWR_BUF_TRIG_SB) && 
                (std::find(triggerTypes.begin(), triggerTypes.end(), "SB") == triggerTypes.end())) {
                triggerTypes.push_back("SB_delayed");
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
            
            // Check if any T1 trigger bits are set
            int t1_bits = (COMPATIBILITY_SHWR_BUF_TRIG_SB |
                          COMPATIBILITY_SHWR_BUF_TRIG_TOT |
                          COMPATIBILITY_SHWR_BUF_TRIG_TOTD |
                          COMPATIBILITY_SHWR_BUF_TRIG_MOPS |
                          SHWR_BUF_TRIG_EM |
                          SHWR_BUF_TRIG_DL);
            
            if (trig_id & t1_bits) {
                hasValidTrigger = true;
            }
            
            // Log trigger information for first few stations
            if (hasValidTrigger && fTracesFound <= 10) {
                ostringstream msg;
                msg << "Station " << fStationId 
                    << " - Trigger: 0x" << std::hex << trig_id << std::dec
                    << " = " << fTriggerType
                    << ", T1: " << (fIsT1 ? "yes" : "no")
                    << ", T2: " << (fIsT2 ? "yes" : "no");
                INFO(msg.str());
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
                << ", Trigger: " << fTriggerType;
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
            << " stations with sim data, found " << nTracesThisEvent << " traces";
        INFO(msg.str());
    }
}

// ProcessPMTs method
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
                
                // Create histogram with trigger type in title
                if (fTracesFound < fMaxHistograms) {
                    TString histName = Form("traceHist_%d", fTracesFound);
                    TString histTitle = Form("Event %d, Station %d, PMT %d [%s trigger]", 
                                            fEventId, fStationId, fPmtId, fTriggerType.c_str());
                    
                    TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
                    traceHist->GetXaxis()->SetTitle("Time bin (25 ns)");
                    traceHist->GetYaxis()->SetTitle("ADC");
                    traceHist->SetStats(kTRUE);
                    
                    // Fill histogram
                    for (int i = 0; i < 2048; i++) {
                        traceHist->SetBinContent(i+1, fTraceData[i]);
                    }
                    
                    fTraceHistograms->Add(traceHist);
                }
                
                // Fill summary histograms
                hTraceLength->Fill(fTraceSize);
                hPeakValue->Fill(fPeakValue);
                hTotalCharge->Fill(fTotalCharge);
                hVEMCharge->Fill(fVEMCharge);
                
                // Fill trigger-specific histograms based on trigger type
                FillTriggerSpecificHistograms();
                
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
                        << ", Peak: " << fPeakValue << " ADC at bin " << fPeakBin
                        << ", Total charge: " << fTotalCharge 
                        << ", VEM: " << fVEMCharge;
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
    
    // Create histogram with trigger type
    if (fTracesFound < fMaxHistograms) {
        TString histName = Form("traceHist_%d", fTracesFound);
        TString histTitle = Form("Event %d, Station %d, PMT %d [%s trigger]", 
                                fEventId, fStationId, fPmtId, fTriggerType.c_str());
        
        TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
        traceHist->GetXaxis()->SetTitle("Time bin (25 ns)");
        traceHist->GetYaxis()->SetTitle("ADC");
        traceHist->SetStats(kTRUE);
        
        // Fill histogram
        for (int i = 0; i < 2048; i++) {
            traceHist->SetBinContent(i+1, fTraceData[i]);
        }
        
        fTraceHistograms->Add(traceHist);
    }
    
    // Fill summary histograms  
    hTraceLength->Fill(fTraceSize);
    hPeakValue->Fill(fPeakValue);
    hTotalCharge->Fill(fTotalCharge);
    hVEMCharge->Fill(fVEMCharge);
    
    // Fill trigger-specific histograms
    FillTriggerSpecificHistograms();
    
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
            << ", BinsAbove1.75VEM=" << binsAbove1_75VEM << ")";
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

// SaveAndDisplayTraces method
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
    
    // First save all data to file
    if (fOutputFile) {
        fOutputFile->cd();
        
        // Write trace histograms
        if (fTraceHistograms && fTraceHistograms->GetEntries() > 0) {
            INFO("Writing trace histograms to file...");
            
            // Create a directory for trace histograms
            TDirectory* traceDir = fOutputFile->mkdir("TraceHistograms");
            traceDir->cd();
            
            // Write all histograms
            fTraceHistograms->Write();
            
            ostringstream msg2;
            msg2 << "Wrote " << fTraceHistograms->GetEntries() << " trace histograms";
            INFO(msg2.str());
            
            // Go back to main directory
            fOutputFile->cd();
        }
        
        // Write everything else
        fOutputFile->Write();
        INFO("Data saved to file.");
    }
    
    // Display sample traces if not in batch mode
    if (!gROOT->IsBatch() && fTraceHistograms && fTraceHistograms->GetEntries() > 0) {
        INFO("Displaying sample FADC traces...");
        
        // Create canvas for display
        TCanvas* c1 = new TCanvas("c1", "Sample FADC Traces", 1200, 800);
        c1->Divide(2, 2);
        
        // Select interesting traces to display (try to get different trigger types)
        int nHists = fTraceHistograms->GetEntries();
        int indices[4];
        
        if (nHists <= 4) {
            for (int i = 0; i < nHists; i++) indices[i] = i;
        } else {
            indices[0] = 0;
            indices[1] = nHists / 3;
            indices[2] = 2 * nHists / 3;
            indices[3] = nHists - 1;
        }
        
        // Display selected traces
        for (int i = 0; i < TMath::Min(4, nHists); i++) {
            c1->cd(i + 1);
            gPad->SetGrid();
            
            TH1D* hist = (TH1D*)fTraceHistograms->At(indices[i]);
            if (hist) {
                hist->SetLineColor(kBlue);
                hist->SetLineWidth(2);
                hist->Draw();
                
                // Set x-axis range to show full trace
                hist->GetXaxis()->SetRangeUser(0, 2048);
                
                // Add statistics
                double maxBin = hist->GetMaximumBin();
                double maxVal = hist->GetMaximum();
                double integral = hist->Integral();
                
                ostringstream traceInfo;
                traceInfo << "Trace index " << indices[i] << ": " << hist->GetTitle()
                          << " (Peak: " << maxVal << " at bin " << maxBin 
                          << ", Total: " << integral << ")";
                INFO(traceInfo.str());
            }
        }
        
        c1->Update();
        
        // Save canvas
        c1->SaveAs("sample_fadc_traces.png");
        c1->SaveAs("sample_fadc_traces.pdf");
        INFO("Sample traces saved to sample_fadc_traces.png and .pdf");
        
        // Create trigger summary canvas
        TCanvas* c2 = new TCanvas("c2", "Trigger Analysis", 1200, 800);
        c2->Divide(2, 2);
        
        c2->cd(1);
        gPad->SetLogy();
        hTriggerTypes->Draw();
        
        c2->cd(2);
        gPad->SetLogy();
        hVEMChargeMOPS->SetLineColor(kRed);
        hVEMChargeMOPS->Draw();
        hVEMChargeToTD->SetLineColor(kBlue);
        hVEMChargeToTD->Draw("same");
        hVEMChargeThreshold->SetLineColor(kGreen);
        hVEMChargeThreshold->Draw("same");
        
        c2->cd(3);
        gPad->SetLogz();
        hChargeVsDistanceMOPS->Draw("colz");
        
        c2->cd(4);
        gPad->SetLogz();
        hChargeVsDistanceToTD->Draw("colz");
        
        c2->Update();
        c2->SaveAs("trigger_analysis.png");
        c2->SaveAs("trigger_analysis.pdf");
        INFO("Trigger analysis saved to trigger_analysis.png and .pdf");
        
        // Keep displayed for a moment
        gSystem->ProcessEvents();
        gSystem->Sleep(3000);
        
        delete c1;
        delete c2;
    }
    
    // Close the output file
    if (fOutputFile) {
        fOutputFile->Close();
        delete fOutputFile;
        
        ostringstream msg3;
        msg3 << "Output written to " << fOutputFileName;
        INFO(msg3.str());
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
    
    if (fOutputFile) {
        fOutputFile->cd();
        
        // Write trace histograms
        if (fTraceHistograms && fTraceHistograms->GetEntries() > 0) {
            INFO("Writing trace histograms...");
            
            TDirectory* traceDir = fOutputFile->mkdir("TraceHistograms");
            traceDir->cd();
            
            fTraceHistograms->Write();
            
            ostringstream msg2;
            msg2 << "Wrote " << fTraceHistograms->GetEntries() << " trace histograms";
            INFO(msg2.str());
            
            fOutputFile->cd();
        }
        
        fOutputFile->Write();
        fOutputFile->Close();
        delete fOutputFile;
        fOutputFile = 0;
        
        ostringstream msg3;
        msg3 << "Output written to " << fOutputFileName;
        INFO(msg3.str());
    }
    
    return eSuccess;
}
