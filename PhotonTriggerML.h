#ifndef _PhotonTriggerML_h_
#define _PhotonTriggerML_h_

//pls dont fuck up
#include <fwk/VModule.h>
#include <vector>
#include <string>

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
    
private:
    // Simple feature extraction
    struct Features {
        double risetime;           // Rise time 10-90%
        double falltime;           // Fall time 90-10%
        double peak_charge_ratio;  // Peak to total charge
        double smoothness;         // Signal smoothness
        double early_late_ratio;   // Early to late charge
        int num_peaks;            // Number of peaks
        double total_charge;      // Total charge (VEM)
        double peak_amplitude;    // Peak amplitude (VEM)
    };
    
    // Extract features from trace
    Features ExtractFeatures(const std::vector<double>& trace, double baseline = 50.0);
    
    // Calculate photon probability score (0-1)
    double CalculatePhotonScore(const Features& features);
    
    // Process a single station
    void ProcessStation(const sevt::Station& station);
    
    // Counters
    int fEventCount;
    int fStationCount;
    int fPhotonLikeCount;
    int fHadronLikeCount;
    
    // Current event data
    double fEnergy;
    double fCoreX;
    double fCoreY;
    
    // Output
    TFile* fOutputFile;
    TTree* fMLTree;
    
    // Histograms
    TH1D* hPhotonScore;
    TH1D* hRisetime;
    TH1D* hSmoothness;
    TH1D* hEarlyLateRatio;
    TH1D* hTotalCharge;
    TH2D* hScoreVsEnergy;
    TH2D* hScoreVsDistance;
    
    // Tree variables
    double fPhotonScore;
    double fDistance;
    int fStationId;
    Features fFeatures;
    
    // Register module
    REGISTER_MODULE("PhotonTriggerML", PhotonTriggerML);
};

#endif // _PhotonTriggerML_h_
