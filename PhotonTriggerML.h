#ifndef _PhotonTriggerML_h_
#define _PhotonTriggerML_h_

/**
 * \file PhotonTriggerML.h
 * \brief Machine Learning based photon trigger for Pierre Auger Observatory
 * 
 * This module uses supervised learning to discriminate between photon-induced
 * and hadron-induced air showers based on PMT trace characteristics.
 * Focus is on time/shape features that are energy-independent.
 */

#include <fwk/VModule.h>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <map>
#include <deque>
#include <cstring>
#include <algorithm>

// Forward declarations
class TFile;
class TTree;
class TH1D;
class TH2D;
class TProfile;

namespace evt { class Event; }
namespace sevt { 
    class Station; 
    class PMT;
}

class PhotonTriggerML : public fwk::VModule {
public:
    PhotonTriggerML();
    virtual ~PhotonTriggerML();
    
    fwk::VModule::ResultFlag Init();
    fwk::VModule::ResultFlag Run(evt::Event& event);
    fwk::VModule::ResultFlag Finish();
    
    static PhotonTriggerML* fInstance;
    void SaveAndDisplaySummary();
    
    /**
     * \struct MLResult
     * \brief ML analysis results for PMTTraceModule compatibility
     */
    struct MLResult {
        double photonScore;
        bool identifiedAsPhoton;
        bool isActualPhoton;
        double vemCharge;
        std::string primaryType;
        double confidence;
        
        struct Features {
            double risetime_10_90;
            double pulse_width;
            double asymmetry;
            double peak_charge_ratio;
            double kurtosis;
            double total_charge;
            double muon_fraction;  // Key discriminator
        } features;
        
        MLResult() : photonScore(0), identifiedAsPhoton(false), 
                    isActualPhoton(false), vemCharge(0), 
                    primaryType("Unknown"), confidence(0) {}
    };
    
    // Static methods for inter-module communication
    static bool GetMLResultForStation(int stationId, MLResult& result);
    static void ClearMLResults();
    
private:
    /**
     * \struct PhysicsFeatures
     * \brief Physics-motivated features for photon/hadron discrimination
     */
    struct PhysicsFeatures {
        // Time profile features (normalized, energy-independent)
        double rise_time_norm;       // Rise time 10-90% normalized
        double fall_time_norm;       // Fall time 90-10% normalized
        double pulse_asymmetry;      // (fall - rise) / (fall + rise)
        double fwhm_norm;            // Full width half max normalized
        
        // Charge distribution features (fractions, not absolute)
        double very_early_fraction;  // Charge before peak-500 bins
        double early_fraction;        // Charge peak-500 to peak-200
        double peak_fraction;         // Charge peak-200 to peak+200
        double late_fraction;         // Charge peak+200 to peak+500
        double very_late_fraction;    // Charge after peak+500 (MUON INDICATOR)
        
        // Shape features
        double time_rms_norm;         // RMS of time distribution
        double time_skewness;         // 3rd moment
        double time_kurtosis;         // 4th moment
        double smoothness;            // Signal smoothness (2nd derivative)
        
        // Muon-specific indicators
        double muon_bump_score;       // Presence of late muon bump
        double n_sub_peaks;           // Number of secondary peaks
        double late_to_early_ratio;   // very_late / early charge
        
        // Signal quality
        double snr_estimate;          // Signal to noise ratio
        
        // Constructor
        PhysicsFeatures() { std::memset(this, 0, sizeof(PhysicsFeatures)); }
    };
    
    /**
     * \class SimpleNeuralNet
     * \brief Lightweight neural network for photon classification
     */
    class SimpleNeuralNet {
    public:
        SimpleNeuralNet();
        void Initialize(int input_size);
        double Predict(const std::vector<double>& features);
        void Train(const std::vector<double>& features, bool is_photon, double lr = 0.001);
        void UpdateWeights(const std::vector<std::pair<std::vector<double>, bool>>& batch);
        bool SaveWeights(const std::string& filename);
        bool LoadWeights(const std::string& filename);
        
    private:
        // Network architecture: input -> hidden1 -> hidden2 -> output
        std::vector<std::vector<double>> fWeights1;  // input to hidden1
        std::vector<double> fBias1;
        std::vector<std::vector<double>> fWeights2;  // hidden1 to hidden2  
        std::vector<double> fBias2;
        std::vector<std::vector<double>> fWeights3;  // hidden2 to output
        std::vector<double> fBias3;
        
        int fInputSize;
        int fHidden1Size;
        int fHidden2Size;
        bool fInitialized;
        
        // Activation functions
        double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
        double relu(double x) { return x > 0 ? x : 0.01 * x; }  // Leaky ReLU
        double sigmoidDerivative(double y) { return y * (1 - y); }
        double reluDerivative(double x) { return x > 0 ? 1.0 : 0.01; }
    };
    
    /**
     * \class EnsembleClassifier
     * \brief Ensemble of multiple classifiers for robust detection
     */
    class EnsembleClassifier {
    public:
        EnsembleClassifier();
        void Initialize();
        double Predict(const PhysicsFeatures& features);
        void AddTrainingSample(const PhysicsFeatures& features, bool is_photon);
        void Train();
        void UpdateThreshold(double target_efficiency);
        
    private:
        // Individual classifier scores
        double MuonContentScore(const PhysicsFeatures& f);
        double TimeProfileScore(const PhysicsFeatures& f);
        double ShapeScore(const PhysicsFeatures& f);
        
        // Ensemble weights
        double fMuonWeight;
        double fTimeWeight;
        double fShapeWeight;
        
        // Dynamic threshold
        double fThreshold;
        
        // Training samples
        std::vector<std::pair<PhysicsFeatures, bool>> fTrainingSamples;
        
        // Simple statistics for threshold tuning
        std::vector<double> fPhotonScores;
        std::vector<double> fHadronScores;
    };
    
    // Processing methods
    void ProcessStation(const sevt::Station& station);
    PhysicsFeatures ExtractPhysicsFeatures(const std::vector<double>& trace);
    bool ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace_data);
    void UpdatePerformanceMetrics(bool is_photon, bool predicted_photon);
    
    // Classifiers
    std::unique_ptr<SimpleNeuralNet> fNeuralNet;
    std::unique_ptr<EnsembleClassifier> fEnsemble;
    
    // Training data management
    struct TrainingBuffer {
        std::deque<std::pair<std::vector<double>, bool>> samples;
        size_t max_size;
        
        TrainingBuffer(size_t max = 10000) : max_size(max) {}
        
        void Add(const std::vector<double>& features, bool is_photon) {
            samples.push_back({features, is_photon});
            if (samples.size() > max_size) {
                samples.pop_front();
            }
        }
        
        size_t PhotonCount() const {
            return std::count_if(samples.begin(), samples.end(), 
                               [](const auto& s) { return s.second; });
        }
        
        size_t HadronCount() const {
            return samples.size() - PhotonCount();
        }
    };
    
    TrainingBuffer fTrainingBuffer;
    
    // Event counters
    int fEventCount;
    int fStationCount;
    int fPhotonEventCount;
    int fHadronEventCount;
    
    // Current event data
    double fEnergy;
    double fCoreX;
    double fCoreY;
    int fPrimaryId;
    std::string fPrimaryType;
    bool fIsActualPhoton;
    int fStationId;
    double fDistance;
    
    // Performance metrics
    int fTruePositives;
    int fFalsePositives;
    int fTrueNegatives;
    int fFalseNegatives;
    
    // Moving average performance
    std::deque<double> fRecentPrecisions;
    std::deque<double> fRecentRecalls;
    const size_t fPerformanceWindow = 100;
    
    // Configuration
    double fEnergyMin;
    double fEnergyMax;
    double fPhotonThreshold;
    double fTargetEfficiency;
    bool fUseEnsemble;
    bool fUseDynamicThreshold;
    bool fSaveFeatures;
    std::string fOutputFileName;
    std::string fWeightsFileName;
    std::string fLogFileName;
    std::ofstream fLogFile;
    
    // Output
    TFile* fOutputFile;
    TTree* fMLTree;
    TTree* fFeatureTree;
    
    // Feature data for tree
    PhysicsFeatures fCurrentFeatures;
    double fPhotonScore;
    bool fPredictedAsPhoton;
    
    // Histograms
    TH1D* hPhotonScore;
    TH1D* hPhotonScoreTrue;
    TH1D* hPhotonScoreFalse;
    TH2D* hScoreVsEnergy;
    TH2D* hScoreVsDistance;
    TH1D* hMuonFraction;
    TH1D* hMuonFractionPhoton;
    TH1D* hMuonFractionHadron;
    TProfile* hEfficiencyVsEnergy;
    TProfile* hPurityVsEnergy;
    TH2D* hConfusionMatrix;
    TH1D* hFeatureImportance;
    
    // Static ML results storage for inter-module communication
    static std::map<int, MLResult> fMLResultsMap;
    
    // Helper methods
    void InitializeHistograms();
    void FillHistograms();
    void TrainClassifiers();
    void UpdateDynamicThreshold();
    double CalculateOptimalThreshold(double target_efficiency);
    
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
