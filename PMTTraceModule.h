#ifndef _PMTTraceModule_h_
#define _PMTTraceModule_h_

/**
 * \file PMTTraceModule.h
 * \brief Module to extract PMT traces from CORSIKA shower simulations
 * 
 * Enhanced version that identifies and stores trigger types (MOPS, ToTD, etc.)
 * for each signal, providing detailed trigger analysis capabilities.
 */

#include <fwk/VModule.h>
#include <utl/TimeDistribution.h>
#include <string>
#include <vector>
#include <map>

class TFile;
class TTree;
class TH1D;
class TH2D;
class TObjArray;

namespace evt { class Event; }
namespace sevt { 
    class Station; 
    class PMT;
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
    
    // Trigger information
    std::string fTriggerType;        // MOPS, ToTD, Threshold, etc.
    std::string fTriggerAlgorithm;   // Full algorithm name
    bool fIsT1;                       // T1 trigger flag
    bool fIsT2;                       // T2 trigger flag
    std::map<std::string, int> fTriggerCounts;  // Counter for each trigger type
    
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
    
    // General histograms
    TH1D* hEventEnergy;
    TH1D* hZenithAngle;
    TH1D* hNStations;
    TH1D* hNTracesPerEvent;
    TH1D* hTraceLength;
    TH1D* hPeakValue;
    TH1D* hTotalCharge;
    TH1D* hVEMCharge;
    TH2D* hChargeVsDistance;
    
    // Trigger-specific histograms
    TH1D* hTriggerTypes;              // Summary of trigger types
    TH1D* hVEMChargeMOPS;            // VEM charge for MOPS triggers
    TH1D* hVEMChargeToTD;            // VEM charge for ToTD triggers
    TH1D* hVEMChargeThreshold;       // VEM charge for Threshold triggers
    TH1D* hVEMChargeOther;           // VEM charge for other triggers
    TH2D* hChargeVsDistanceMOPS;     // Charge vs distance for MOPS
    TH2D* hChargeVsDistanceToTD;     // Charge vs distance for ToTD
    
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
