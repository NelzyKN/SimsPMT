#ifndef _PhotonTriggerML_h_
#define _PhotonTriggerML_h_

/**
 * \file PhotonTriggerML.h
 * \brief Machine Learning-based T1 photon trigger for the Pierre Auger Observatory
 * 
 * Implements a lightweight neural network for real-time discrimination between
 * photon-induced and hadronic air showers at the T1 trigger level.
 * Target: 50% improvement in photon efficiency, 30% reduction in hadronic background
 * Energy range: 10^18 - 10^19 eV
 */

#include <fwk/VModule.h>
#include <vector>
#include <string>
#include <array>
#include <map>
#include <memory>

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

// Feature extraction parameters
const int NUM_FEATURES = 24;  // Number of features for ML model
const int TRACE_BINS = 2048;  // FADC trace length
const int TIME_WINDOWS = 8;   // Number of time windows for feature extraction

/**
 * Lightweight neural network for photon discrimination
 * Uses 8-bit quantized weights for FPGA deployment
 */
class PhotonDiscriminatorNN {
public:
    PhotonDiscriminatorNN();
    ~PhotonDiscriminatorNN();
    
    // Initialize network with weights (would be loaded from file in production)
    void Initialize();
    
    // Forward pass - returns photon probability
    float Predict(const std::vector<float>& features);
    
    // Get decision with threshold
    bool IsPhotonLike(const std::vector<float>& features, float threshold = 0.5);
    
private:
    // Network architecture: 24 inputs -> 16 hidden -> 8 hidden -> 1 output
    static const int HIDDEN1_SIZE = 16;
    static const int HIDDEN2_SIZE = 8;
    
    // Quantized weights (8-bit) - in production would be loaded from trained model
    std::vector<std::vector<int8_t>> weights1_;  // Input to hidden1
    std::vector<std::vector<int8_t>> weights2_;  // Hidden1 to hidden2
    std::vector<int8_t> weights3_;              // Hidden2 to output
    
    std::vector<int8_t> bias1_;
    std::vector<int8_t> bias2_;
    int8_t bias3_;
    
    // Quantization scale factors
    float input_scale_;
    float weight_scale_;
    float output_scale_;
    
    // Activation function (ReLU for hidden, sigmoid for output)
    float ReLU(float x) { return x > 0 ? x : 0; }
    float Sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
};

/**
 * Feature extractor for FADC traces
 * Extracts discriminating features between photon and hadronic showers
 */
class TraceFeatureExtractor {
public:
    struct Features {
        // Temporal features
        float risetime_10_50;      // Rise time from 10% to 50% of peak
        float risetime_10_90;      // Rise time from 10% to 90% of peak
        float falltime_90_10;      // Fall time from 90% to 10% of peak
        float fwhm;                // Full width at half maximum
        
        // Amplitude features
        float peak_amplitude;       // Peak value (in VEM)
        float total_charge;        // Total integrated charge (in VEM)
        float peak_to_charge;      // Ratio of peak to total charge
        
        // Shape features
        float asymmetry;           // Signal asymmetry
        float kurtosis;            // Signal kurtosis (peakedness)
        float num_peaks;           // Number of significant peaks
        
        // Time-segmented features (8 windows)
        std::array<float, TIME_WINDOWS> window_charges;  // Charge in each time window
        
        // Muon-related features (key discriminators)
        float early_to_late_ratio; // Ratio of early to late signal
        float smoothness;          // Signal smoothness (low for muons)
        float peak_density;        // Density of peaks in signal
        
        // All features as vector for NN input
        std::vector<float> GetFeatureVector() const;
    };
    
    // Extract features from FADC trace
    Features ExtractFeatures(const std::vector<double>& trace, double baseline = 50.0);
    
private:
    // Helper methods
    double CalculateBaseline(const std::vector<double>& trace);
    int FindPeakBin(const std::vector<double>& trace, double baseline);
    double CalculateFWHM(const std::vector<double>& trace, int peak_bin, double peak_val, double baseline);
    int CountPeaks(const std::vector<double>& trace, double baseline, double threshold);
    double CalculateSmoothness(const std::vector<double>& trace);
};

/**
 * Main photon trigger ML module
 */
class PhotonTriggerML : public fwk::VModule {
public:
    PhotonTriggerML();
    virtual ~PhotonTriggerML();
    
    fwk::VModule::ResultFlag Init();
    fwk::VModule::ResultFlag Run(evt::Event& event);
    fwk::VModule::ResultFlag Finish();
    
    // Configuration parameters
    struct Config {
        double photon_threshold = 0.5;     // NN threshold for photon classification
        double energy_min = 1e18;          // Minimum energy (eV)
        double energy_max = 1e19;          // Maximum energy (eV)
        bool save_features = true;         // Save features to ROOT file
        bool apply_trigger = true;         // Apply T1 trigger decision
        std::string model_file = "";       // Path to trained model weights
    };
    
private:
    Config config_;
    
    // ML components
    std::unique_ptr<PhotonDiscriminatorNN> discriminator_;
    std::unique_ptr<TraceFeatureExtractor> extractor_;
    
    // Statistics
    int total_stations_;
    int photon_triggers_;
    int hadron_triggers_;
    int true_photon_triggers_;    // If MC truth available
    int false_photon_triggers_;   // If MC truth available
    
    // Performance metrics
    double efficiency_;            // Photon detection efficiency
    double purity_;               // Photon trigger purity
    double background_reduction_; // Hadronic background reduction
    
    // Output
    TFile* output_file_;
    TTree* feature_tree_;
    TTree* performance_tree_;
    
    // Histograms
    TH1D* h_nn_output_all_;
    TH1D* h_nn_output_photon_;
    TH1D* h_nn_output_hadron_;
    TH2D* h_nn_vs_energy_;
    TH2D* h_nn_vs_distance_;
    TH1D* h_efficiency_vs_energy_;
    TH1D* h_purity_vs_energy_;
    TGraph* g_roc_curve_;
    
    // Feature histograms
    TH1D* h_risetime_photon_;
    TH1D* h_risetime_hadron_;
    TH1D* h_smoothness_photon_;
    TH1D* h_smoothness_hadron_;
    TH1D* h_early_late_photon_;
    TH1D* h_early_late_hadron_;
    
    // Tree variables
    float tree_nn_output_;
    float tree_energy_;
    float tree_distance_;
    int tree_is_photon_truth_;
    int tree_trigger_decision_;
    std::vector<float> tree_features_;
    
    // Helper methods
    void ProcessStation(const evt::Event& event, const sevt::Station& station);
    bool ApplyT1Trigger(float nn_output, const TraceFeatureExtractor::Features& features);
    void UpdatePerformanceMetrics();
    void CreateROCCurve();
    
    // Register module
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
