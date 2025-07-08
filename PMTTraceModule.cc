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
    hNStations = new TH1D("hNStations", "Number of Stations;N;Events", 50, 0, 50);
    hNTracesPerEvent = new TH1D("hNTracesPerEvent", "Traces per Event;N;Events", 150, 0, 150);
    hTraceLength = new TH1D("hTraceLength", "Trace Length;Bins;Entries", 200, 0, 2100);
    hPeakValue = new TH1D("hPeakValue", "Peak Value;ADC;Entries", 200, 0, 500);  // Adjusted for lower baseline
    hTotalCharge = new TH1D("hTotalCharge", "Total Charge;ADC;Entries", 200, 0, 10000);  // Adjusted
    hVEMCharge = new TH1D("hVEMCharge", "VEM Charge;VEM;Entries", 200, 0, 100);  // Adjusted
    hChargeVsDistance = new TH2D("hChargeVsDistance", "Charge vs Distance;r [m];VEM", 
                                 50, 0, 3000, 100, 0.1, 100);  // Adjusted VEM range
    
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
        
        // For simulations, we're interested in stations that have simulation data
        // This means they were hit by particles from the shower
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
                
                // First, determine the actual trace size
                int actualTraceSize = 0;
                try {
                    actualTraceSize = trace.GetSize();
                } catch (...) {
                    // If GetSize() doesn't work, determine by iteration
                    actualTraceSize = 0;
                    for (int i = 0; i < 2048; i++) {
                        try {
                            (void)trace[i];
                            actualTraceSize = i + 1;
                        } catch (...) {
                            break;
                        }
                    }
                }
                
                if (actualTraceSize <= 0) {
                    continue;  // Skip empty traces
                }
                
                // Debug output
                if (fTracesFound < 5) {
                    ostringstream msg;
                    msg << "FADC actual trace size: " << actualTraceSize << ", First 20 values: ";
                    for (int i = 0; i < TMath::Min(20, actualTraceSize); i++) {
                        msg << trace[i] << " ";
                    }
                    msg << "... Last 5 values: ";
                    for (int i = TMath::Max(0, actualTraceSize-5); i < actualTraceSize; i++) {
                        msg << trace[i] << " ";
                    }
                    INFO(msg.str());
                    
                    // Check baseline level
                    double baseline_sum = 0;
                    int baseline_count = 0;
                    for (int i = 0; i < 100 && i < actualTraceSize; i++) {
                        baseline_sum += trace[i];
                        baseline_count++;
                    }
                    if (baseline_count > 0) {
                        ostringstream baseMsg;
                        baseMsg << "Estimated baseline (first 100 bins): " << baseline_sum/baseline_count << " ADC";
                        INFO(baseMsg.str());
                    }
                }
                
                // Process trace data - fill all 2048 bins
                fTraceSize = 2048;  // Always use full size for histograms
                fPeakValue = 0;
                fPeakBin = 0;
                fTotalCharge = 0;
                
                // First, read actual trace data
                for (int i = 0; i < actualTraceSize; i++) {
                    double value = trace[i];
                    fTraceData.push_back(value);
                    
                    double signal = value - 50.0;  // Baseline subtraction
                    if (signal > 0) {
                        fTotalCharge += signal;
                    }
                    
                    if (value > fPeakValue) {
                        fPeakValue = value;
                        fPeakBin = i;
                    }
                }
                
                // Then fill the rest with baseline to reach 2048 bins
                for (int i = actualTraceSize; i < 2048; i++) {
                    fTraceData.push_back(50.0);  // Fill with baseline
                }
                
                // Calculate VEM
                fVEMCharge = fTotalCharge / 180.0;
                
                // Create histogram - always 2048 bins
                if (fTracesFound < fMaxHistograms) {
                    TString histName = Form("eventHist_%d", 1000000000 + fTracesFound);
                    TString histTitle = Form("Histogram of Event %d, Station %d, PMT %d", 
                                            fEventId, fStationId, fPmtId);
                    
                    TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
                    traceHist->GetXaxis()->SetTitle("Time bin");
                    traceHist->GetYaxis()->SetTitle("ADC");
                    traceHist->SetStats(kTRUE);
                    
                    // Fill all 2048 bins
                    for (int i = 0; i < 2048; i++) {
                        traceHist->SetBinContent(i+1, fTraceData[i]);
                    }
                    
                    fTraceHistograms->Add(traceHist);
                    
                    // Log actual vs displayed size
                    if (fTracesFound < 5) {
                        ostringstream msg;
                        msg << "Created histogram with 2048 bins (actual data: " << actualTraceSize << " bins)";
                        INFO(msg.str());
                    }
                }
                
                // Fill summary histograms
                hTraceLength->Fill(actualTraceSize);  // Store actual data length
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
                
                if (fTracesFound % 10 == 0) {
                    ostringstream msg;
                    msg << "Found " << fTracesFound << " traces so far";
                    INFO(msg.str());
                }
                
                if (fTracesFound <= 5) {
                    INFO("Found FADC trace directly from PMT");
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
            
            // Try different simulation data sources in order of preference
            
            // First try: FADC trace from simulation (post-filtering)
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
            
            // Second try: PE time distribution (if no FADC available)
            try {
                if (simData.HasPETimeDistribution()) {
                    const auto& peTrace = simData.GetPETimeDistribution(
                        sevt::StationConstants::eTotal
                    );
                    
                    if (ProcessTimeDistribution(peTrace)) {
                        tracesFound++;
                        if (fTracesFound <= 5) {
                            INFO("WARNING: Using PE time distribution - not a true FADC trace");
                        }
                        continue;
                    }
                }
            } catch (const exception& e) {
                // PE distribution access failed
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
    
    // First determine actual size
    int actualSize = 0;
    for (int i = 0; i < 2048; i++) {
        try {
            (void)timeDist[i];
            actualSize = i + 1;
        } catch (...) {
            break;
        }
    }
    
    if (actualSize == 0) {
        return false;  // No data
    }
    
    // Debug output
    if (fTracesFound < 5) {
        ostringstream msg;
        msg << "TimeDistribution actual size: " << actualSize << ", First 20 values: ";
        for (int i = 0; i < TMath::Min(20, actualSize); i++) {
            msg << timeDist[i] << " ";
        }
        if (actualSize > 20) {
            msg << "... Last 5 values: ";
            for (int i = TMath::Max(0, actualSize-5); i < actualSize; i++) {
                msg << timeDist[i] << " ";
            }
        }
        INFO(msg.str());
    }
    
    fTraceSize = 2048;  // Always use full size for histograms
    fPeakValue = 0;
    fPeakBin = 0;
    fTotalCharge = 0;
    
    // Get actual trace data
    for (int i = 0; i < actualSize; i++) {
        double value = timeDist[i];
        fTraceData.push_back(value);
        
        // For simulation data, baseline appears to be around 50 ADC
        double signal = value - 50.0;
        if (signal > 0) {
            fTotalCharge += signal;
        }
        
        if (value > fPeakValue) {
            fPeakValue = value;
            fPeakBin = i;
        }
    }
    
    // Fill the rest with baseline to reach 2048 bins
    for (int i = actualSize; i < 2048; i++) {
        fTraceData.push_back(50.0);
    }
    
    // Calculate VEM charge
    fVEMCharge = fTotalCharge / 180.0;
    
    // Create histogram - always 2048 bins
    if (fTracesFound < fMaxHistograms) {
        TString histName = Form("eventHist_%d", 1000000000 + fTracesFound);
        TString histTitle = Form("Histogram of Event %d, Station %d, PMT %d", 
                                fEventId, fStationId, fPmtId);
        
        TH1D* traceHist = new TH1D(histName, histTitle, 2048, 0, 2048);
        traceHist->GetXaxis()->SetTitle("Time bin");
        traceHist->GetYaxis()->SetTitle("ADC");
        traceHist->SetStats(kTRUE);
        
        // Fill all 2048 bins
        for (int i = 0; i < 2048; i++) {
            traceHist->SetBinContent(i+1, fTraceData[i]);
        }
        
        fTraceHistograms->Add(traceHist);
        
        // Log actual vs displayed size
        if (fTracesFound < 5) {
            ostringstream msg;
            msg << "Created histogram with 2048 bins (actual data: " << actualSize << " bins)";
            INFO(msg.str());
        }
    }
    
    // Fill summary histograms  
    hTraceLength->Fill(actualSize);  // Store actual data length
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
    
    if (fTracesFound % 10 == 0) {
        ostringstream msg;
        msg << "Found " << fTracesFound << " traces so far";
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
