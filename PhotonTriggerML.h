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
        double photonScore;          // Anomaly score (reconstruction error)
        bool identifiedAsPhoton;     // Is it an anomaly?
        bool isActualPhoton;         // Ground truth
        double vemCharge;            // Total charge
        std::string primaryType;     // Primary particle type
        double confidence;           // Confidence in the prediction
        
        // Simple feature storage for compatibility
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
     * \brief Simple autoencoder for anomaly detection
     */
    class Autoencoder {
    public:
        Autoencoder();
        ~Autoencoder() = default;
        
        void Initialize(int input_size);
        double GetReconstructionError(const std::vector<double>& features);
        void Train(const std::vector<std::vector<double>>& features, double learning_rate);
        void UpdateThreshold(const std::vector<std::vector<double>>& hadron_features);
        double GetThreshold() const { return fAnomalyThreshold; }
        
    private:
        int fInputSize;
        int fLatentSize;
        bool fInitialized;
        
        // Encoder weights and biases
        std::vector<std::vector<double>> fEncoderWeights;
        std::vector<double> fEncoderBias;
        
        // Decoder weights and biases
        std::vector<std::vector<double>> fDecoderWeights;
        std::vector<double> fDecoderBias;
        
        // Momentum for optimization
        std::vector<std::vector<double>> fEncoderMomentum;
        std::vector<std::vector<double>> fDecoderMomentum;
        
        double fAnomalyThreshold;
    };
    
private:
    void ProcessStation(const sevt::Station& station);
    std::vector<double> ExtractSimplifiedFeatures(const std::vector<double>& trace);
    void CalculateAndDisplayMetrics();
    bool ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace_data);
    
    // Store simple features for MLResult
    MLResult::Features ExtractCompatibilityFeatures(const std::vector<double>& trace);
    
    // Autoencoder
    std::unique_ptr<Autoencoder> fAutoencoder;
    
    // Training data
    std::vector<std::vector<double>> fHadronFeatures;
    
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
    
    // Output
    TFile* fOutputFile;
    TTree* fMLTree;
    
    // Histograms
    TH1D* hReconstructionError;
    TH1D* hErrorPhotons;
    TH1D* hErrorHadrons;
    TH2D* hConfusionMatrix;
    
    // Static ML results storage for inter-module communication
    static std::map<int, MLResult> fMLResultsMap;
    
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
