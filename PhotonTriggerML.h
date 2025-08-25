#ifndef _PhotonTriggerML_h_
#define _PhotonTriggerML_h_

/**
 * \file PhotonTriggerML.h
 * \brief ML-based photon trigger module for the Pierre Auger Observatory
 * 
 * This module implements a machine learning-based trigger system designed to
 * discriminate between photon-induced and hadronic air showers in real-time.
 * It extracts key features from PMT FADC traces and applies a lightweight
 * neural network-inspired scoring algorithm to identify photon candidates.
 * 
 * The module targets:
 * - 50% improvement in photon detection efficiency
 * - 30% reduction in background hadronic triggers
 * - Sensitivity to photon fluxes ~10^-4 km^-2 yr^-1 at 10^19 eV
 * 
 * Author: Khoa Nguyen
 * Institution: Michigan Technological University
 * Supervisor: David F. Nitz
 * Date: 2025
 * 
 * Part of PhD research on "Development of Advanced Photon Triggers 
 * for the Pierre Auger Collaboration"
 */

#include <fwk/VModule.h>
#include <vector>
#include <string>
#include <map>
#include <fstream>  // Added for log file

// Forward declarations
class TFile;
class TTree;
class TH1D;
class TH2D;

namespace evt { class Event; }
namespace sevt { 
    class Station; 
    class PMT;
}

/**
 * \class PhotonTriggerML
 * \brief Machine learning-based photon trigger for Surface Detector stations
 * 
 * This module processes FADC traces from water-Cherenkov detectors to identify
 * photon-induced air showers based on their characteristic electromagnetic
 * signatures. The algorithm uses features such as signal rise time, smoothness,
 * and temporal distribution to discriminate photons from hadronic backgrounds.
 */
class PhotonTriggerML : public fwk::VModule {
public:
    /**
     * \brief Constructor - initializes counters and pointers
     */
    PhotonTriggerML();
    
    /**
     * \brief Destructor - cleanup (file closing handled in Finish)
     */
    virtual ~PhotonTriggerML();
    
    /**
     * \brief Initialize the module
     * 
     * Creates output ROOT file, initializes histograms and TTrees,
     * and sets up the ML discriminator parameters.
     * 
     * \return eSuccess if initialization successful, eFailure otherwise
     */
    fwk::VModule::ResultFlag Init();
    
    /**
     * \brief Process one event
     * 
     * Main processing method called once per event. Loops through all
     * stations and PMTs, extracts FADC traces, calculates features,
     * and applies ML discrimination.
     * 
     * \param event Reference to the current event data
     * \return eSuccess if processing successful
     */
    fwk::VModule::ResultFlag Run(evt::Event& event);
    
    /**
     * \brief Finalize the module
     * 
     * Calculates final statistics, writes all data to file,
     * and prints summary information.
     * 
     * \return eSuccess if finalization successful
     */
    fwk::VModule::ResultFlag Finish();
    
    // Signal handling - public for signal handler access
    static PhotonTriggerML* fInstance;  ///< Static instance for signal handler
    
    /**
     * \brief Save data and display summary on interrupt
     * 
     * Called when Ctrl+C is pressed to save partial results
     */
    void SaveAndDisplaySummary();
    
    /**
     * \brief Get primary particle type for external access
     * 
     * Used by PMTTraceModule to include particle type in histograms
     */
    std::string GetPrimaryType() const { return fPrimaryType; }
    int GetPrimaryId() const { return fPrimaryId; }
    
    /**
     * \struct Features
     * \brief Container for extracted FADC trace features
     * 
     * These features are used by the ML algorithm to discriminate
     * between photon and hadronic showers. Made public for MLResult struct.
     */
    struct Features {
        double risetime;           ///< Rise time from 10% to 90% of peak [ns]
        double falltime;           ///< Fall time from 90% to 10% of peak [ns]
        double peak_charge_ratio;  ///< Ratio of peak amplitude to total charge
        double smoothness;         ///< RMS of second derivative (signal smoothness)
        double early_late_ratio;   ///< Ratio of early to late charge in trace
        int num_peaks;            ///< Number of significant peaks in trace
        double total_charge;      ///< Total integrated charge [VEM]
        double peak_amplitude;    ///< Peak amplitude [VEM]
    };
    
    /**
     * \struct MLResult
     * \brief Container for ML analysis results per station
     */
    struct MLResult {
        double photonScore;      ///< ML photon probability score (0-1)
        bool identifiedAsPhoton; ///< True if score > threshold
        bool isActualPhoton;     ///< True if primary is photon
        double vemCharge;        ///< Total VEM charge
        Features features;       ///< Extracted features
        std::string primaryType; ///< Primary particle type
        
        MLResult() : photonScore(0), identifiedAsPhoton(false), 
                    isActualPhoton(false), vemCharge(0), primaryType("Unknown") {}
    };
    
    /**
     * \brief Get ML results for a specific station
     * 
     * Used by PMTTraceModule to access ML analysis results
     * 
     * \param stationId The station ID to query
     * \param result Output parameter filled with ML results
     * \return True if results exist for this station
     */
    static bool GetMLResultForStation(int stationId, MLResult& result);
    
    /**
     * \brief Clear all stored ML results
     * 
     * Called at the beginning of each event
     */
    static void ClearMLResults();
    
private:
    
    /**
     * \brief Extract features from FADC trace
     * 
     * Analyzes a PMT FADC trace to extract discriminating features
     * that characterize photon vs hadronic showers.
     * 
     * \param trace Vector containing FADC trace data (2048 bins)
     * \param baseline Baseline ADC value (default 50 for simulation)
     * \return Features struct containing extracted characteristics
     */
    Features ExtractFeatures(const std::vector<double>& trace, double baseline = 50.0);
    
    /**
     * \brief Calculate photon probability score
     * 
     * Applies ML-inspired scoring algorithm to determine likelihood
     * that the signal originated from a photon-induced shower.
     * Uses weighted combination of features based on photon characteristics:
     * - Shorter rise times (electromagnetic cascade)
     * - Smoother signals (less muon content)
     * - Higher early/late ratio (faster development)
     * 
     * \param features Extracted trace features
     * \return Photon score between 0 (hadron-like) and 1 (photon-like)
     */
    double CalculatePhotonScore(const Features& features);
    
    /**
     * \brief Process a single station
     * 
     * Loops through all PMTs in a station, extracts FADC traces,
     * and applies photon discrimination. Handles both direct PMT
     * trace access and simulation data fallback.
     * 
     * \param station Reference to the station to process
     */
    void ProcessStation(const sevt::Station& station);
    
    /**
     * \brief Write ML analysis to log file
     * 
     * Writes detailed ML analysis results, confusion matrix,
     * and performance metrics to the log file.
     */
    void WriteMLAnalysisToLog();
    
    // ===== Event and Station Counters =====
    int fEventCount;          ///< Total number of events processed
    int fStationCount;        ///< Total number of stations with valid traces
    int fPhotonLikeCount;     ///< Number of photon-like signals (score > 0.5)
    int fHadronLikeCount;     ///< Number of hadron-like signals (score <= 0.5)
    
    // ===== Current Event Data =====
    double fEnergy;           ///< Primary particle energy [eV]
    double fCoreX;            ///< Shower core X position [m]
    double fCoreY;            ///< Shower core Y position [m]
    int fPrimaryId;           ///< Primary particle ID from simulation
    std::string fPrimaryType; ///< Primary particle type (photon, proton, iron, etc.)
    
    // ===== Current Station/PMT Data =====
    double fPhotonScore;      ///< ML photon probability score (0-1)
    double fDistance;         ///< Distance from shower core [m]
    int fStationId;          ///< Current station ID
    bool fIsActualPhoton;     ///< True if primary particle is a photon
    Features fFeatures;       ///< Current trace features
    
    // ===== Output File and Trees =====
    TFile* fOutputFile;       ///< Output ROOT file
    TTree* fMLTree;          ///< Tree containing ML analysis results
    
    // ===== Log File =====
    std::string fLogFileName; ///< Name of the log file
    std::ofstream fLogFile;   ///< Log file for ML analysis results
    
    // ===== Analysis Histograms =====
    
    // 1D Histograms
    TH1D* hPhotonScore;       ///< Distribution of photon scores
    TH1D* hRisetime;         ///< Distribution of signal rise times
    TH1D* hSmoothness;       ///< Distribution of signal smoothness values
    TH1D* hEarlyLateRatio;   ///< Distribution of early/late charge ratios
    TH1D* hTotalCharge;      ///< Distribution of total charge (VEM)
    
    // 2D Correlation Histograms
    TH2D* hScoreVsEnergy;    ///< Photon score vs primary energy
    TH2D* hScoreVsDistance;  ///< Photon score vs core distance
    
    // Performance Analysis Histograms
    TH1D* hPhotonScorePhotons; ///< Score distribution for true photons
    TH1D* hPhotonScoreHadrons; ///< Score distribution for true hadrons
    TH2D* hROCCurve;           ///< ROC curve for ML performance
    
    // ===== Performance Metrics =====
    int fTruePositives;      ///< Photon correctly identified as photon
    int fFalsePositives;     ///< Hadron incorrectly identified as photon
    int fTrueNegatives;      ///< Hadron correctly identified as hadron
    int fFalseNegatives;     ///< Photon incorrectly identified as hadron
    
    // ===== Trigger Configuration =====
    double fPhotonThreshold;  ///< Threshold for photon classification (default 0.5)
    double fEnergyMin;        ///< Minimum energy for trigger [eV]
    double fEnergyMax;        ///< Maximum energy for trigger [eV]
    std::string fOutputFileName; ///< Name of output ROOT file
    
    // ===== Trigger Type Analysis =====
    std::map<std::string, int> fTriggerCounts; ///< Count of stations by trigger type
    
    // ===== Particle Type Statistics =====
    std::map<std::string, int> fParticleTypeCounts; ///< Count of events by particle type
    std::map<std::string, int> fParticleTypePhotonLike; ///< Photon-like counts by particle type
    
    // ===== Static ML Results Storage =====
    static std::map<int, MLResult> fMLResultsMap; ///< ML results by station ID for inter-module communication
    
    /**
     * \brief Register module with framework
     * 
     * This macro registers the PhotonTriggerML module with the Auger Offline
     * framework, allowing it to be invoked via XML configuration files.
     */
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
