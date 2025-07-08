#ifndef _PMTTraceModule_h_
#define _PMTTraceModule_h_

/**
 * \file PMTTraceModule.h
 * \brief Module to extract PMT traces from CORSIKA shower simulations
 * 
 * This module reads CORSIKA shower data and extracts FADC traces
 * from PMT simulation data. The traces are saved to a ROOT file
 * for further analysis.
 */

#include <fwk/VModule.h>
#include <utl/TimeDistribution.h>  // Include full definition
#include <string>
#include <vector>

class TFile;
class TTree;
class TH1D;
class TH2D;
class TObjArray;

namespace evt { class Event; }
namespace sevt { 
    class Station; 
    class PMT;
    // TraceI is defined in PMT headers
}

class PMTTraceModule : public fwk::VModule {
public:
    PMTTraceModule();
    virtual ~PMTTraceModule();
    
    fwk::VModule::ResultFlag Init();
    fwk::VModule::ResultFlag Run(evt::Event& event);
    fwk::VModule::ResultFlag Finish();
    
    // Signal handling - public for signal handler access
    static PMTTraceModule* fInstance;  // For signal handler access
    void SaveAndDisplayTraces();       // Called on interrupt
    
private:
    // Configuration
    std::string fOutputFileName;
    
    // Event counters
    int fEventCount;
    int fProcessedEvents;
    int fTracesFound;
    
    // Current event data
    int fEventId;
    int fStationId;
    int fPmtId;
    double fEnergy;
    double fZenith;
    double fCoreX;
    double fCoreY;
    double fStationX;
    double fStationY;
    double fDistance;
    
    // Trace data
    std::vector<double> fTraceData;
    int fTraceSize;
    int fPeakBin;
    double fPeakValue;
    double fTotalCharge;
    double fVEMCharge;
    
    // Output file and tree
    TFile* fOutputFile;
    TTree* fTraceTree;
    
    // Histograms
    TH1D* hEventEnergy;
    TH1D* hZenithAngle;
    TH1D* hNStations;
    TH1D* hNTracesPerEvent;
    TH1D* hTraceLength;
    TH1D* hPeakValue;
    TH1D* hTotalCharge;
    TH1D* hVEMCharge;
    TH2D* hChargeVsDistance;
    
    // Trace histogram storage
    TObjArray* fTraceHistograms;
    int fMaxHistograms;
    
    // Helper methods
    void ProcessStations(const evt::Event& event);
    int ProcessPMTs(const sevt::Station& station);
    bool ProcessTimeDistribution(const utl::TimeDistribution<int>& timeDist);
    
    // Register module with framework
    REGISTER_MODULE("PMTTraceModule", PMTTraceModule);
};

#endif // _PMTTraceModule_h_
