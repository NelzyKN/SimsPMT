#ifndef _PhotonTriggerML_h_
#define _PhotonTriggerML_h_

/**
 * \file PhotonTriggerML.h
 * \brief Balanced ML-based photon trigger module with improved training
 * 
 * This version includes better class balancing, improved features,
 * and more sophisticated training strategies.
 * 
 * Author: Khoa Nguyen
 * Institution: Michigan Technological University
 * Date: 2025
 */

#include <fwk/VModule.h>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <memory>
#include <array>
#include <cmath>
#include <deque>

// Forward declarations
class TFile;
class TTree;
class TH1D;
class TH2D;
class TGraph;
class TCanvas;

namespace evt { class Event; }
namespace sevt { 
    class Station; 
    class PMT;
}

/**
 * \class PhotonTriggerML
 * \brief Machine learning photon trigger with improved balance
 */
class PhotonTriggerML : public fwk::VModule {
public:
    PhotonTriggerML();
    virtual ~PhotonTriggerML();
    
    fwk::VModule::ResultFlag Init();
    fwk::VModule::ResultFlag Run(evt::Event& event);
    fwk::VModule::ResultFlag Finish();
    
    // Signal handling
    static PhotonTriggerML* fInstance;
    void SaveAndDisplaySummary();
    
    // Primary type access
    std::string GetPrimaryType() const { return fPrimaryType; }
    int GetPrimaryId() const { return fPrimaryId; }
    
    /**
     * \struct EnhancedFeatures
     * \brief Comprehensive feature set for photon discrimination
     */
    struct EnhancedFeatures {
        // Temporal features
        double risetime_10_50;     ///< Rise time from 10% to 50% of peak [ns]
        double risetime_10_90;     ///< Rise time from 10% to 90% of peak [ns]
        double falltime_90_10;     ///< Fall time from 90% to 10% [ns]
        double pulse_width;        ///< FWHM of main pulse [ns]
        double asymmetry;          ///< (falltime - risetime)/(falltime + risetime)
        
        // Amplitude features
        double peak_amplitude;     ///< Peak amplitude [VEM]
        double total_charge;       ///< Total integrated charge [VEM]
        double peak_charge_ratio;  ///< Peak/total charge ratio
        
        // Shape features
        double smoothness;         ///< RMS of second derivative
        double kurtosis;          ///< Fourth moment (peakedness)
        double skewness;          ///< Third moment (asymmetry)
        
        // Temporal distribution
        double early_fraction;     ///< Fraction of charge in first 25%
        double late_fraction;      ///< Fraction of charge in last 25%
        double time_spread;        ///< RMS time spread [ns]
        
        // Frequency domain
        double high_freq_content;  ///< High frequency power ratio
        
        // Multi-peak structure
        int num_peaks;            ///< Number of significant peaks
        double secondary_peak_ratio; ///< Ratio of 2nd largest to largest peak
        
        // Additional discriminative features
        double rise_fall_ratio;   ///< Risetime/falltime ratio
        double peak_time;         ///< Time of peak relative to start [ns]
        double charge_asymmetry;  ///< Early-late charge asymmetry
        
        EnhancedFeatures() : 
            risetime_10_50(0), risetime_10_90(0), falltime_90_10(0),
            pulse_width(0), asymmetry(0), peak_amplitude(0),
            total_charge(0), peak_charge_ratio(0), smoothness(0),
            kurtosis(0), skewness(0), early_fraction(0),
            late_fraction(0), time_spread(0), high_freq_content(0),
            num_peaks(0), secondary_peak_ratio(0), rise_fall_ratio(0),
            peak_time(0), charge_asymmetry(0) {}
    };
    
    /**
     * \class NeuralNetwork
     * \brief Improved neural network with better architecture
     */
    class NeuralNetwork {
    public:
        NeuralNetwork();
        ~NeuralNetwork() = default;
        
        /**
         * \brief Initialize network with improved architecture
         * \param input_size Number of input features
         * \param hidden_sizes Vector of hidden layer sizes
         */
        void Initialize(int input_size, const std::vector<int>& hidden_sizes);
        
        /**
         * \brief Forward pass through the network
         * \param features Input feature vector
         * \param training Whether to apply dropout
         * \return Photon probability (0-1)
         */
        double Predict(const std::vector<double>& features, bool training = false);
        
        /**
         * \brief Train with balanced batch and proper loss
         * \param features Batch of feature vectors
         * \param labels Batch of labels (1 for photon, 0 for hadron)
         * \param weights Sample weights for balancing
         * \param learning_rate Learning rate
         * \return Average loss for the batch
         */
        double Train(const std::vector<std::vector<double>>& features,
                    const std::vector<int>& labels,
                    const std::vector<double>& weights,
                    double learning_rate = 0.001);
        
        /**
         * \brief Save network weights to file
         */
        void SaveWeights(const std::string& filename);
        
        /**
         * \brief Load network weights from file
         */
        bool LoadWeights(const std::string& filename);
        
        /**
         * \brief Calculate optimal threshold using validation set
         */
        double CalculateOptimalThreshold(
            const std::vector<std::vector<double>>& val_features,
            const std::vector<int>& val_labels);
        
    private:
        // Network architecture
        int fInputSize;
        std::vector<int> fHiddenSizes;
        int fNumLayers;
        
        // Weight matrices and biases for each layer
        std::vector<std::vector<std::vector<double>>> fWeights;
        std::vector<std::vector<double>> fBiases;
        
        // Adam optimizer state
        std::vector<std::vector<std::vector<double>>> fMomentum1_w;
        std::vector<std::vector<std::vector<double>>> fMomentum2_w;
        std::vector<std::vector<double>> fMomentum1_b;
        std::vector<std::vector<double>> fMomentum2_b;
        int fTimeStep;
        
        // Regularization
        double fDropoutRate;
        double fL2Lambda;
        
        // Activation functions
        double LeakyReLU(double x, double alpha = 0.01) { 
            return x > 0 ? x : alpha * x; 
        }
        double LeakyReLUDerivative(double x, double alpha = 0.01) { 
            return x > 0 ? 1 : alpha; 
        }
        double Sigmoid(double x) { 
            // Numerically stable sigmoid
            if (x >= 0) {
                double exp_neg_x = exp(-x);
                return 1.0 / (1.0 + exp_neg_x);
            } else {
                double exp_x = exp(x);
                return exp_x / (1.0 + exp_x);
            }
        }
        
        // Batch normalization parameters (simplified)
        std::vector<std::vector<double>> fBatchNormMean;
        std::vector<std::vector<double>> fBatchNormVar;
        std::vector<std::vector<double>> fBatchNormGamma;
        std::vector<std::vector<double>> fBatchNormBeta;
    };
    
    /**
     * \struct MLResult
     * \brief ML analysis results
     */
    struct MLResult {
        double photonScore;
        bool identifiedAsPhoton;
        bool isActualPhoton;
        double vemCharge;
        EnhancedFeatures features;
        std::string primaryType;
        double confidence;
        
        MLResult() : photonScore(0), identifiedAsPhoton(false), 
                    isActualPhoton(false), vemCharge(0), 
                    primaryType("Unknown"), confidence(0) {}
    };
    
    // Static methods for inter-module communication
    static bool GetMLResultForStation(int stationId, MLResult& result);
    static void ClearMLResults();
    
private:
    /**
     * \brief Extract enhanced features from FADC trace
     */
    EnhancedFeatures ExtractEnhancedFeatures(const std::vector<double>& trace, 
                                             double baseline = 50.0);
    
    /**
     * \brief Normalize features with robust scaling
     */
    std::vector<double> NormalizeFeatures(const EnhancedFeatures& features);
    
    /**
     * \brief Update feature statistics for normalization
     */
    void UpdateFeatureStatistics(const EnhancedFeatures& features);
    
    /**
     * \brief Process a single station
     */
    void ProcessStation(const sevt::Station& station);
    
    /**
     * \brief Balanced training with proper sampling
     */
    double TrainNetwork();
    
    /**
     * \brief Generate synthetic photon samples (SMOTE-like)
     */
    void GenerateSyntheticPhotons();
    
    /**
     * \brief Calculate and display metrics
     */
    void CalculateAndDisplayMetrics();
    
    /**
     * \brief Extract trace data from PMT
     */
    bool ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace_data);
    
    /**
     * \brief Dynamic threshold adjustment based on performance
     */
    void UpdateThreshold();
    
    // Neural network
    std::unique_ptr<NeuralNetwork> fNeuralNetwork;
    
    // Training data with balanced sampling
    std::deque<std::vector<double>> fPhotonFeatures;
    std::deque<std::vector<double>> fHadronFeatures;
    
    // Validation data
    std::vector<std::vector<double>> fValidationFeatures;
    std::vector<int> fValidationLabels;
    
    // Training parameters
    bool fIsTraining;
    int fTrainingEpochs;
    int fBatchSize;
    double fLearningRate;
    double fLearningRateDecay;
    
    // Early stopping
    double fBestValidationLoss;
    int fEpochsSinceImprovement;
    int fPatienceEpochs;
    
    // Event and station counters
    int fEventCount;
    int fStationCount;
    int fPhotonLikeCount;
    int fHadronLikeCount;
    
    // Current event data
    double fEnergy;
    double fCoreX;
    double fCoreY;
    int fPrimaryId;
    std::string fPrimaryType;
    
    // Current station data
    double fPhotonScore;
    double fConfidence;
    double fDistance;
    int fStationId;
    bool fIsActualPhoton;
    EnhancedFeatures fFeatures;
    
    // Output files
    TFile* fOutputFile;
    TTree* fMLTree;
    std::string fLogFileName;
    std::ofstream fLogFile;
    
    // Histograms
    TH1D* hPhotonScore;
    TH1D* hPhotonScorePhotons;
    TH1D* hPhotonScoreHadrons;
    TH1D* hConfidence;
    TH1D* hRisetime;
    TH1D* hAsymmetry;
    TH1D* hKurtosis;
    TH2D* hScoreVsEnergy;
    TH2D* hScoreVsDistance;
    TGraph* gROCCurve;
    TH2D* hConfusionMatrix;
    TH1D* hTrainingLoss;
    TH1D* hValidationLoss;
    TH1D* hAccuracyHistory;
    
    // Performance metrics
    int fTruePositives;
    int fFalsePositives;
    int fTrueNegatives;
    int fFalseNegatives;
    
    // Configuration
    double fPhotonThreshold;
    double fAdaptiveThreshold;  // Dynamically adjusted
    double fEnergyMin;
    double fEnergyMax;
    std::string fOutputFileName;
    std::string fWeightsFileName;
    bool fLoadPretrainedWeights;
    bool fSaveFeatures;
    bool fApplyTrigger;
    
    // Feature normalization (robust scaling)
    std::vector<double> fFeatureMedians;
    std::vector<double> fFeatureIQRs;
    bool fNormalizationInitialized;
    
    // Class weight balancing
    double fPhotonWeight;
    double fHadronWeight;
    
    // Maximum buffer sizes
    static const int kMaxPhotonBuffer = 5000;
    static const int kMaxHadronBuffer = 5000;
    
    // Static ML results storage
    static std::map<int, MLResult> fMLResultsMap;
    
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
