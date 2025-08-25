#ifndef _PhotonTriggerML_h_
#define _PhotonTriggerML_h_

/**
 * \file PhotonTriggerML.h
 * \brief Enhanced ML-based photon trigger module with real neural network
 * 
 * This improved version implements a lightweight feedforward neural network
 * with proper training capabilities and better feature engineering.
 * Designed for FPGA deployment with 8-bit quantized weights.
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
#include <cmath>  // For exp() function in Sigmoid

// Forward declarations
class TFile;
class TTree;
class TH1D;
class TH2D;
class TGraph;

namespace evt { class Event; }
namespace sevt { 
    class Station; 
    class PMT;
}

/**
 * \class PhotonTriggerML
 * \brief Enhanced machine learning photon trigger with real neural network
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
     * \brief Improved feature set based on photon shower characteristics
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
        
        // Frequency domain (simple)
        double high_freq_content;  ///< High frequency power ratio
        
        // Multi-peak structure
        int num_peaks;            ///< Number of significant peaks
        double secondary_peak_ratio; ///< Ratio of 2nd largest to largest peak
        
        EnhancedFeatures() : 
            risetime_10_50(0), risetime_10_90(0), falltime_90_10(0),
            pulse_width(0), asymmetry(0), peak_amplitude(0),
            total_charge(0), peak_charge_ratio(0), smoothness(0),
            kurtosis(0), skewness(0), early_fraction(0),
            late_fraction(0), time_spread(0), high_freq_content(0),
            num_peaks(0), secondary_peak_ratio(0) {}
    };
    
    /**
     * \class NeuralNetwork
     * \brief Lightweight feedforward neural network for photon discrimination
     * 
     * 3-layer network with ReLU activation, designed for FPGA implementation
     * Uses 8-bit quantized weights for efficient hardware deployment
     */
    class NeuralNetwork {
    public:
        NeuralNetwork();
        ~NeuralNetwork() = default;
        
        /**
         * \brief Initialize network with random weights
         * \param input_size Number of input features (17 for EnhancedFeatures)
         * \param hidden1_size First hidden layer size (default 32)
         * \param hidden2_size Second hidden layer size (default 16)
         */
        void Initialize(int input_size = 17, int hidden1_size = 32, int hidden2_size = 16);
        
        /**
         * \brief Forward pass through the network
         * \param features Input feature vector
         * \return Photon probability (0-1)
         */
        double Predict(const std::vector<double>& features);
        
        /**
         * \brief Train the network on a batch of examples
         * \param features Batch of feature vectors
         * \param labels Batch of labels (1 for photon, 0 for hadron)
         * \param learning_rate Learning rate (default 0.001)
         * \return Average loss for the batch
         */
        double Train(const std::vector<std::vector<double>>& features,
                    const std::vector<int>& labels,
                    double learning_rate = 0.001);
        
        /**
         * \brief Save network weights to file
         * \param filename Output file name
         */
        void SaveWeights(const std::string& filename);
        
        /**
         * \brief Load network weights from file
         * \param filename Input file name
         * \return true if successful
         */
        bool LoadWeights(const std::string& filename);
        
        /**
         * \brief Quantize weights to 8-bit for FPGA deployment
         */
        void QuantizeWeights();
        
    private:
        // Network architecture
        int fInputSize;
        int fHidden1Size;
        int fHidden2Size;
        
        // Weight matrices (row-major storage)
        std::vector<std::vector<double>> fWeights1;  ///< Input to hidden1
        std::vector<std::vector<double>> fWeights2;  ///< Hidden1 to hidden2
        std::vector<std::vector<double>> fWeights3;  ///< Hidden2 to output
        
        // Bias vectors
        std::vector<double> fBias1;
        std::vector<double> fBias2;
        double fBias3;
        
        // Activation function (ReLU)
        double ReLU(double x) { return x > 0 ? x : 0; }
        double ReLUDerivative(double x) { return x > 0 ? 1 : 0; }
        
        // Sigmoid for output layer
        double Sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
        
        // Quantization parameters
        bool fIsQuantized;
        double fQuantizationScale;
    };
    
    /**
     * \struct MLResult
     * \brief Enhanced ML analysis results
     */
    struct MLResult {
        double photonScore;
        bool identifiedAsPhoton;
        bool isActualPhoton;
        double vemCharge;
        EnhancedFeatures features;
        std::string primaryType;
        double confidence;  ///< Confidence level (distance from 0.5)
        
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
     * \param trace FADC trace data (2048 bins)
     * \param baseline Baseline ADC value
     * \return Enhanced feature set
     */
    EnhancedFeatures ExtractEnhancedFeatures(const std::vector<double>& trace, 
                                             double baseline = 50.0);
    
    /**
     * \brief Normalize features for neural network input
     * \param features Raw features
     * \return Normalized feature vector
     */
    std::vector<double> NormalizeFeatures(const EnhancedFeatures& features);
    
    /**
     * \brief Process a single station
     * \param station Station to process
     */
    void ProcessStation(const sevt::Station& station);
    
    /**
     * \brief Train the neural network on accumulated data
     */
    void TrainNetwork();
    
    /**
     * \brief Calculate performance metrics
     */
    void CalculatePerformanceMetrics();
    
    /**
     * \brief Write analysis to log file
     */
    void WriteMLAnalysisToLog();
    
    // Neural network
    std::unique_ptr<NeuralNetwork> fNeuralNetwork;
    
    // Training data accumulation
    std::vector<std::vector<double>> fTrainingFeatures;
    std::vector<int> fTrainingLabels;
    bool fIsTraining;
    int fTrainingEpochs;
    
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
    
    // Performance metrics
    int fTruePositives;
    int fFalsePositives;
    int fTrueNegatives;
    int fFalseNegatives;
    
    // Configuration
    double fPhotonThreshold;
    double fEnergyMin;
    double fEnergyMax;
    std::string fOutputFileName;
    std::string fWeightsFileName;
    bool fLoadPretrainedWeights;
    
    // Feature normalization parameters (computed from training data)
    std::vector<double> fFeatureMeans;
    std::vector<double> fFeatureStdDevs;
    
    // Trigger type analysis
    std::map<std::string, int> fTriggerCounts;
    std::map<std::string, int> fParticleTypeCounts;
    std::map<std::string, int> fParticleTypePhotonLike;
    
    // Static ML results storage
    static std::map<int, MLResult> fMLResultsMap;
    
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
