#ifndef _PMTTraceModule_h_
#define _PMTTraceModule_h_

/**
 * \file PMTTraceModule.h
 * \brief Module to extract PMT traces from CORSIKA shower simulations
 * 
 * This module reads CORSIKA shower data and extracts FADC traces
 * from PMT simulation data. The traces are saved to a ROOT file
 * for further analysis. Now includes primary particle type in histogram titles.
 */

#include <fwk/VModule.h>
#include <utl/TimeDistribution.h>  // Include full definition
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
    
    // Primary particle information
    int fPrimaryId;
    std::string fPrimaryType;
    bool fIsPhoton;
    
    // ML Analysis results from PhotonTriggerML
    double fMLPhotonScore;
    bool fMLIdentifiedAsPhoton;
    std::string fMLPrediction;
    
    // Trace data
    std::vector<double> fTraceData;
    int fTraceSize;
    int fPeakBin;
    double fPeakValue;
    double fTotalCharge;
    double fVEMCharge;
    
    // Trigger information
    std::string fTriggerType;
    std::string fTriggerAlgorithm;
    bool fIsT1;
    bool fIsT2;
    std::map<std::string, int> fTriggerCounts;
    
    // Particle type statistics
    std::map<std::string, int> fParticleTypeCounts;
    std::map<std::string, std::map<std::string, int>> fTriggerCountsByParticle;
    
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
    
    // Trigger type specific histograms
    TH1D* hTriggerTypes;
    TH1D* hVEMChargeMOPS;
    TH1D* hVEMChargeToTD;
    TH1D* hVEMChargeThreshold;
    TH1D* hVEMChargeOther;
    TH2D* hChargeVsDistanceMOPS;
    TH2D* hChargeVsDistanceToTD;
    
    // Particle type histograms
    TH1D* hParticleTypes;
    TH1D* hVEMChargePhoton;
    TH1D* hVEMChargeProton;
    TH1D* hVEMChargeIron;
    
    // Trace histogram storage
    TObjArray* fTraceHistograms;
    int fMaxHistograms;
    
    // Helper methods
    void ProcessStations(const evt::Event& event);
    int ProcessPMTs(const sevt::Station& station);
    bool ProcessTimeDistribution(const utl::TimeDistribution<int>& timeDist);
    
    /**
     * \brief Extract primary particle information from event
     * 
     * Determines the primary particle type from the shower simulation data
     * and sets fPrimaryId, fPrimaryType, and fIsPhoton accordingly.
     * 
     * \param event The current event being processed
     */
    void ExtractPrimaryParticleInfo(const evt::Event& event);
    
    /**
     * \brief Get particle type string from PDG ID
     * 
     * Converts PDG particle ID to human-readable string
     * 
     * \param pdgId The PDG particle ID
     * \return String representation of particle type
     */
    std::string GetParticleTypeFromId(int pdgId);
    
    /**
     * \brief Infer trigger type from signal characteristics
     * 
     * When trigger data is not explicitly available, this method
     * analyzes the FADC trace to infer the most likely trigger type
     * based on signal characteristics like peak height, duration, etc.
     */
    void InferTriggerType();
    
    /**
     * \brief Fill trigger-specific histograms
     * 
     * Fills the appropriate trigger-type-specific histograms based on
     * the current trigger type. Handles multiple trigger types if present.
     */
    void FillTriggerSpecificHistograms();
    
    /**
     * \brief Fill particle-type-specific histograms
     * 
     * Fills histograms based on the primary particle type
     */
    void FillParticleSpecificHistograms();
    
    // Register module with framework
    REGISTER_MODULE("PMTTraceModule", PMTTraceModule);
};

#endif // _PMTTraceModule_h_
