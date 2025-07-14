// PMTTraceModule.cc
// Fixed version for extracting FADC traces from CORSIKA simulations

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

using namespace std;
using namespace fwk;
using namespace evt;
using namespace sevt;
using namespace det;
using namespace sdet;
using namespace utl;

// Static instance pointer for signal handler
PMTTraceModule* PMTTraceModule::fInstance = 0;

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
    INFO("========================================");
    
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
    hPeakValue = new TH1D("hPeakValue", "Peak Value;ADC;Entries", 200, 0, 1000);  // Increased range
    hTotalCharge = new TH1D("hTotalCharge", "Total Charge;ADC;Entries", 200, 0, 50000);  // Increased range
    hVEMCharge = new TH1D("hVEMCharge", "VEM Charge;VEM;Entries", 200, 0, 300);  // Increased range
    hChargeVsDistance = new TH2D("hChargeVsDistance", "Charge vs Distance;r [m];VEM", 
                                 50, 0, 3000, 100, 0.1, 1000);  // Increased VEM range
    
    // Initialize trace histogram array
    fTraceHistograms = new TObjArray();
    fTraceHistograms->SetOwner(kTRUE);
    
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
        
        // Check if station has trigger data and was actually triggered
        if (!station.HasTriggerData()) {
            continue;
        }
        
        const sevt::StationTriggerData& triggerData = station.GetTriggerData();
        
        // Only process stations that actually triggered (T1 or T2)
        // Skip silent triggers and error states
        if (triggerData.IsSilent() || triggerData.GetErrorCode() != 0) {
            continue;
        }
        
        // Check if it's a valid trigger (T1 or T2)
        bool isT1 = triggerData.IsT1();
        bool isT2 = triggerData.IsT2();
        
        if (!isT1 && !isT2) {
            continue;
        }
        
        // For simulations, also check if station has simulation data
        if (!station.HasSimData()) {
            continue;
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
        
        if (fTracesFound <= 5 && tracesThisStation > 0) {
            ostringstream msg;
            msg << "Station " << fStationId << " triggered with " 
                << (isT1 ? "T1" : "T2") << " trigger, "
                << "Algorithm: " << triggerData.GetAlgorithmName()
                << ", Distance: " << fDistance << " m";
            INFO(msg.str());
        }
    }
    
    if (nStations > 0) {
        hNStations->Fill(nStations);
        hNTracesPerEvent->Fill(nTracesThisEvent);
    }
}

// ProcessPMTs method - Fixed version
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
        
        // Method 1: Try to get FADC trace directly from PMT (like in EAStripperDFN.cc)
        if (pmt.HasFADCTrace()) {
            try {
                // Use auto to avoid type issues
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
                        // If we can't read this bin, use baseline
                        value = 50.0;
                    }
                    fTraceData.push_back(value);
                    
                    // For simulation, baseline is typically around 50 ADC
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
                    continue;  // Skip traces with almost no signal
                }
                
                // Create histogram
                if (fTracesFound < fMaxHistograms) {
                    TString histName = Form("eventHist_%d", 1000000000 + fTracesFound);
                    TString histTitle = Form("Event %d, Station %d, PMT %d", 
                                            fEventId, fStationId, fPmtId);
                    
                    TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
                    traceHist->GetXaxis()->SetTitle("Time bin");
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
                
                if (fDistance > 0 && fVEMCharge > 0.1) {
                    hChargeVsDistance->Fill(fDistance, fVEMCharge);
                }
                
                // Save to tree
                if (fTraceTree) {
                    fTraceTree->Fill();
                }
                
                fTracesFound++;
                tracesFound++;
                
                if (fTracesFound <= 5) {
                    ostringstream msg;
                    msg << "Found FADC trace: Station " << fStationId 
                        << ", PMT " << fPmtId 
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
    
    // Read the full trace - TimeDistribution should contain full FADC data
    for (int i = 0; i < expectedSize; i++) {
        double value = 0;
        try {
            value = timeDist[i];
        } catch (...) {
            // If we can't read this bin, it might be beyond the actual data
            // For simulation baseline, use 50 ADC
            value = 50.0;
        }
        fTraceData.push_back(value);
        
        // For simulation data, baseline is around 50 ADC
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
        return false;  // Skip traces with almost no signal
    }
    
    // Calculate VEM charge
    fVEMCharge = fTotalCharge / 180.0;
    
    // Create histogram
    if (fTracesFound < fMaxHistograms) {
        TString histName = Form("eventHist_%d", 1000000000 + fTracesFound);
        TString histTitle = Form("Event %d, Station %d, PMT %d", 
                                fEventId, fStationId, fPmtId);
        
        TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
        traceHist->GetXaxis()->SetTitle("Time bin");
        traceHist->GetYaxis()->SetTitle("ADC");
        traceHist->SetStats(kTRUE);
        
        // Fill histogram with all 2048 bins
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
    
    if (fDistance > 0 && fVEMCharge > 0.1) {
        hChargeVsDistance->Fill(fDistance, fVEMCharge);
    }
    
    // Save to tree
    if (fTraceTree) {
        fTraceTree->Fill();
    }
    
    fTracesFound++;
    
    if (fTracesFound <= 5) {
        ostringstream msg;
        msg << "Found trace from TimeDistribution: Station " << fStationId 
            << ", PMT " << fPmtId 
            << ", Peak: " << fPeakValue << " ADC at bin " << fPeakBin
            << ", Total charge: " << fTotalCharge 
            << ", VEM: " << fVEMCharge;
        INFO(msg.str());
    }
    
    return true;
}

// SaveAndDisplayTraces method
void PMTTraceModule::SaveAndDisplayTraces()
{
    ostringstream msg;
    msg << "Interrupt handler called. Found " << fTracesFound << " traces so far.";
    INFO(msg.str());
    
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
        
        // Select interesting traces to display
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
        
        // Keep displayed for a moment
        gSystem->ProcessEvents();
        gSystem->Sleep(3000);
        
        delete c1;
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
