#ifndef _PhotonTriggerML_h_
#define _PhotonTriggerML_h_

#include <fwk/VModule.h>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <map>

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
        } features;
        
        MLResult() : photonScore(0), identifiedAsPhoton(false), 
                    isActualPhoton(false), vemCharge(0), 
                    primaryType("Unknown"), confidence(0) {}
    };
    
    // Static methods for inter-module communication
    static bool GetMLResultForStation(int stationId, MLResult& result);
    static void ClearMLResults();
    
    /**
     * \class Autoencoder
     * \brief Deep autoencoder for anomaly detection
     */
    class Autoencoder {
    public:
        Autoencoder();
        ~Autoencoder() = default;
        
        void Initialize(int input_size);
        double GetReconstructionError(const std::vector<double>& features);
        void Train(const std::vector<std::vector<double>>& features, double learning_rate);
        void UpdateThreshold(const std::vector<std::vector<double>>& hadron_features, 
                           double percentile = 0.85);
        double GetThreshold() const { return fAnomalyThreshold; }
        
    private:
        int fInputSize;
        int fLatentSize;
        bool fInitialized;
        int fTrainingStep;
        
        // Deep encoder weights and biases (4 layers)
        std::vector<std::vector<double>> fEncoderWeights1;
        std::vector<double> fEncoderBias1;
        std::vector<std::vector<double>> fEncoderWeights2;
        std::vector<double> fEncoderBias2;
        std::vector<std::vector<double>> fEncoderWeights3;
        std::vector<double> fEncoderBias3;
        std::vector<std::vector<double>> fEncoderWeights4;
        std::vector<double> fEncoderBias4;
        
        // Deep decoder weights and biases (4 layers)
        std::vector<std::vector<double>> fDecoderWeights1;
        std::vector<double> fDecoderBias1;
        std::vector<std::vector<double>> fDecoderWeights2;
        std::vector<double> fDecoderBias2;
        std::vector<std::vector<double>> fDecoderWeights3;
        std::vector<double> fDecoderBias3;
        std::vector<std::vector<double>> fDecoderWeights4;
        std::vector<double> fDecoderBias4;
        
        // Adam optimizer momentum and velocity for encoder
        std::vector<std::vector<double>> fEncoderMomentum1;
        std::vector<std::vector<double>> fEncoderVelocity1;
        std::vector<std::vector<double>> fEncoderMomentum2;
        std::vector<std::vector<double>> fEncoderVelocity2;
        std::vector<std::vector<double>> fEncoderMomentum3;
        std::vector<std::vector<double>> fEncoderVelocity3;
        std::vector<std::vector<double>> fEncoderMomentum4;
        std::vector<std::vector<double>> fEncoderVelocity4;
        
        // Adam optimizer for encoder biases
        std::vector<double> fEncoderBiasMomentum1;
        std::vector<double> fEncoderBiasVelocity1;
        std::vector<double> fEncoderBiasMomentum2;
        std::vector<double> fEncoderBiasVelocity2;
        std::vector<double> fEncoderBiasMomentum3;
        std::vector<double> fEncoderBiasVelocity3;
        std::vector<double> fEncoderBiasMomentum4;
        std::vector<double> fEncoderBiasVelocity4;
        
        // Adam optimizer momentum and velocity for decoder
        std::vector<std::vector<double>> fDecoderMomentum1;
        std::vector<std::vector<double>> fDecoderVelocity1;
        std::vector<std::vector<double>> fDecoderMomentum2;
        std::vector<std::vector<double>> fDecoderVelocity2;
        std::vector<std::vector<double>> fDecoderMomentum3;
        std::vector<std::vector<double>> fDecoderVelocity3;
        std::vector<std::vector<double>> fDecoderMomentum4;
        std::vector<std::vector<double>> fDecoderVelocity4;
        
        // Adam optimizer for decoder biases
        std::vector<double> fDecoderBiasMomentum1;
        std::vector<double> fDecoderBiasVelocity1;
        std::vector<double> fDecoderBiasMomentum2;
        std::vector<double> fDecoderBiasVelocity2;
        std::vector<double> fDecoderBiasMomentum3;
        std::vector<double> fDecoderBiasVelocity3;
        std::vector<double> fDecoderBiasMomentum4;
        std::vector<double> fDecoderBiasVelocity4;
        
        // Compatibility with original simple version
        std::vector<std::vector<double>> fEncoderWeights;
        std::vector<double> fEncoderBias;
        std::vector<std::vector<double>> fDecoderWeights;
        std::vector<double> fDecoderBias;
        std::vector<std::vector<double>> fEncoderMomentum;
        std::vector<std::vector<double>> fDecoderMomentum;
        
        double fAnomalyThreshold;
        
        // Helper activation functions
        double relu(double x);
        double reluDerivative(double x);
        double sigmoid(double x);
        double sigmoidDerivative(double y);
        double tanhDerivative(double y);
        
        // Adam optimizer update functions
        void updateWeightsAdam(std::vector<std::vector<double>>& weights,
                              const std::vector<double>& gradients,
                              const std::vector<double>& inputs,
                              std::vector<std::vector<double>>& momentum,
                              std::vector<std::vector<double>>& velocity,
                              double lr, double beta1, double beta2, double epsilon);
        
        void updateBiasAdam(std::vector<double>& bias,
                           const std::vector<double>& gradients,
                           std::vector<double>& momentum,
                           std::vector<double>& velocity,
                           double lr, double beta1, double beta2, double epsilon);
    };
    
private:
    void ProcessStation(const sevt::Station& station);
    std::vector<double> ExtractSimplifiedFeatures(const std::vector<double>& trace);
    std::vector<double> ExtractComprehensiveFeatures(const std::vector<double>& trace);
    void UpdateFeatureStatistics(const std::vector<double>& features);
    std::vector<double> NormalizeFeatures(const std::vector<double>& features);
    void CalculateAndDisplayMetrics();
    bool ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace_data);
    MLResult::Features ExtractCompatibilityFeatures(const std::vector<double>& trace);
    
    // Autoencoder
    std::unique_ptr<Autoencoder> fAutoencoder;
    
    // Training data
    std::vector<std::vector<double>> fHadronFeatures;
    std::vector<std::vector<double>> fPhotonFeatures;  // New: separate photon storage
    
    // Feature normalization
    std::vector<double> fFeatureMeans;
    std::vector<double> fFeatureStds;
    int fFeatureCount;
    
    // Event counters
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
    int fStationId;
    double fDistance;
    double fReconstructionError;
    bool fIsAnomaly;
    bool fIsActualPhoton;
    
    // Performance metrics
    int fTruePositives;
    int fFalsePositives;
    int fTrueNegatives;
    int fFalseNegatives;
    
    // Configuration
    double fEnergyMin;
    double fEnergyMax;
    std::string fOutputFileName;
    std::string fLogFileName;
    std::ofstream fLogFile;
    
    // New configuration parameters
    double fTargetPercentile;
    double fCurrentThreshold;
    bool fDynamicThresholdEnabled;
    double fValidationFraction;
    
    // Output
    TFile* fOutputFile;
    TTree* fMLTree;
    
    // Histograms
    TH1D* hReconstructionError;
    TH1D* hErrorPhotons;
    TH1D* hErrorHadrons;
    TH2D* hConfusionMatrix;
    TH1D* hThresholdEvolution;
    TH1D* hFeatureImportance;
    
    // Static ML results storage for inter-module communication
    static std::map<int, MLResult> fMLResultsMap;
    
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
