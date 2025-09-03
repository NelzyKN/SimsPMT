#include "PhotonTriggerML.h"

#include <fwk/CentralConfig.h>
#include <utl/ErrorLogger.h>
#include <evt/Event.h>
#include <evt/ShowerSimData.h>
#include <sevt/SEvent.h>
#include <sevt/Station.h>
#include <sevt/PMT.h>
#include <sevt/PMTSimData.h>
#include <sevt/StationConstants.h>
#include <sdet/Station.h>
#include <sdet/PMTConstants.h>
#include <det/Detector.h>
#include <sdet/SDetector.h>
#include <utl/CoordinateSystemPtr.h>
#include <utl/TimeDistribution.h>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TGraph.h>
#include <TMath.h>
#include <TRandom3.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TPaveText.h>
#include <TLegend.h>
#include <TDirectory.h>
#include <TKey.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <csignal>
#include <iomanip>
#include <ctime>
#include <random>
#include <deque>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace det;
using namespace utl;

// ============================================================================
// Static globals
// ============================================================================
PhotonTriggerML* PhotonTriggerML::fInstance = 0;
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;
static volatile sig_atomic_t gCalledFromSignal = 0;

// Small helpers
static inline double clamp01(double x){ return (x<0.0?0.0:(x>1.0?1.0:x)); }
static inline double sigmoid(double x){ return 1.0/(1.0+std::exp(-x)); }

// ----------------------------------------------------------------------------
void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        std::cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << std::endl;
        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); }
            catch (...) { /* swallow */ }
        }
        std::signal(signal, SIG_DFL);
        std::raise(signal);
        _Exit(0);
    }
}

// ============================================================================
// Neural network (unchanged topology; cleaned warnings)
// ============================================================================
PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.10),
    fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int h1, int h2)
{
    fInputSize = input_size; fHidden1Size = h1; fHidden2Size = h2;

    std::cout << "Initializing Physics-Based Neural Network: "
              << input_size << " -> " << h1 << " -> " << h2 << " -> 1" << std::endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.assign(h1, std::vector<double>(input_size, 0.0));
    for (int i=0;i<h1;++i)
        for (int j=0;j<input_size;++j)
            fWeights1[i][j] = dist(gen)/std::sqrt((double)input_size);

    fWeights2.assign(h2, std::vector<double>(h1, 0.0));
    for (int i=0;i<h2;++i)
        for (int j=0;j<h1;++j)
            fWeights2[i][j] = dist(gen)/std::sqrt((double)h1);

    fWeights3.assign(1, std::vector<double>(h2, 0.0));
    for (int j=0;j<h2;++j)
        fWeights3[0][j] = dist(gen)/std::sqrt((double)h2);

    fBias1.assign(h1, 0.0);
    fBias2.assign(h2, 0.0);
    fBias3 = 0.0;

    // momentum buffers
    fMomentum1_w1.assign(h1, std::vector<double>(input_size, 0.0));
    fMomentum1_w2.assign(h2, std::vector<double>(h1, 0.0));
    fMomentum1_w3.assign(1, std::vector<double>(h2, 0.0));
    fMomentum2_w1.assign(h1, std::vector<double>(input_size, 0.0));
    fMomentum2_w2.assign(h2, std::vector<double>(h1, 0.0));
    fMomentum2_w3.assign(1, std::vector<double>(h2, 0.0));

    fMomentum1_b1.assign(h1, 0.0);
    fMomentum1_b2.assign(h2, 0.0);
    fMomentum1_b3 = 0.0;
    fMomentum2_b1.assign(h1, 0.0);
    fMomentum2_b2.assign(h2, 0.0);
    fMomentum2_b3 = 0.0;

    fTimeStep = 0;
    std::cout << "Neural Network initialized." << std::endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if ((int)x.size()!=fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size,0.0), h2(fHidden2Size,0.0);

    for (int i=0;i<fHidden1Size;++i){
        double s = fBias1[i];
        for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
        h1[i] = sigmoid(s);
        if (training && (std::rand()/double(RAND_MAX))<fDropoutRate) h1[i]=0.0;
    }
    for (int i=0;i<fHidden2Size;++i){
        double s = fBias2[i];
        for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
        h2[i] = sigmoid(s);
        if (training && (std::rand()/double(RAND_MAX))<fDropoutRate) h2[i]=0.0;
    }
    double out = fBias3;
    for (int j=0;j<fHidden2Size;++j) out += fWeights3[0][j]*h2[j];
    return sigmoid(out);
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y,
                                             double lr)
{
    if (X.empty() || X.size()!=y.size()) return -1.0;

    const int N = (int)X.size();
    int num_photons = std::count(y.begin(), y.end(), 1);
    double w_photon = (num_photons>0)?1.5:1.0;
    double w_hadron = 1.0;

    std::vector<std::vector<double>> gW1(fHidden1Size, std::vector<double>(fInputSize,0.0));
    std::vector<std::vector<double>> gW2(fHidden2Size, std::vector<double>(fHidden1Size,0.0));
    std::vector<std::vector<double>> gW3(1, std::vector<double>(fHidden2Size,0.0));
    std::vector<double> gB1(fHidden1Size,0.0), gB2(fHidden2Size,0.0);
    double gB3 = 0.0;

    double total_loss=0.0;

    for (int n=0;n<N;++n){
        const std::vector<double>& x = X[n];
        const int label = y[n];
        const double w = (label==1)?w_photon:w_hadron;

        std::vector<double> h1(fHidden1Size,0.0), h2(fHidden2Size,0.0);
        std::vector<double> r1(fHidden1Size,0.0), r2(fHidden2Size,0.0);

        for (int i=0;i<fHidden1Size;++i){
            double s=fBias1[i]; for(int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
            r1[i]=s; h1[i]=sigmoid(s);
        }
        for (int i=0;i<fHidden2Size;++i){
            double s=fBias2[i]; for(int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
            r2[i]=s; h2[i]=sigmoid(s);
        }
        double r3=fBias3; for(int j=0;j<fHidden2Size;++j) r3 += fWeights3[0][j]*h2[j];
        double p = sigmoid(r3);

        total_loss += -w*( label*std::log(p+1e-7) + (1-label)*std::log(1-p+1e-7) );
        double d3 = w*(p - label);

        for (int j=0;j<fHidden2Size;++j) gW3[0][j] += d3*h2[j];
        gB3 += d3;

        std::vector<double> d2(fHidden2Size,0.0);
        for (int i=0;i<fHidden2Size;++i){
            d2[i] = fWeights3[0][i]*d3 * h2[i]*(1-h2[i]);
        }
        for (int i=0;i<fHidden2Size;++i){
            for (int j=0;j<fHidden1Size;++j) gW2[i][j] += d2[i]*h1[j];
            gB2[i] += d2[i];
        }

        std::vector<double> d1(fHidden1Size,0.0);
        for (int j=0;j<fHidden1Size;++j){
            double s=0.0; for (int i=0;i<fHidden2Size;++i) s += fWeights2[i][j]*d2[i];
            d1[j] = s * h1[j]*(1-h1[j]);
        }
        for (int i=0;i<fHidden1Size;++i){
            for (int j=0;j<fInputSize;++j) gW1[i][j] += d1[i]*x[j];
            gB1[i] += d1[i];
        }
    }

    const double mom=0.9; fTimeStep++;

    for (int i=0;i<fHidden1Size;++i){
        for (int j=0;j<fInputSize;++j){
            double g = gW1[i][j]/N;
            fMomentum1_w1[i][j] = mom*fMomentum1_w1[i][j] - lr*g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        double gb = gB1[i]/N;
        fMomentum1_b1[i] = mom*fMomentum1_b1[i] - lr*gb;
        fBias1[i] += fMomentum1_b1[i];
    }

    for (int i=0;i<fHidden2Size;++i){
        for (int j=0;j<fHidden1Size;++j){
            double g = gW2[i][j]/N;
            fMomentum1_w2[i][j] = mom*fMomentum1_w2[i][j] - lr*g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        double gb = gB2[i]/N;
        fMomentum1_b2[i] = mom*fMomentum1_b2[i] - lr*gb;
        fBias2[i] += fMomentum1_b2[i];
    }

    for (int j=0;j<fHidden2Size;++j){
        double g = gW3[0][j]/N;
        fMomentum1_w3[0][j] = mom*fMomentum1_w3[0][j] - lr*g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }
    {
        double gb = gB3/N;
        fMomentum1_b3 = mom*fMomentum1_b3 - lr*gb;
        fBias3 += fMomentum1_b3;
    }
    return total_loss/N;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Error: Could not save weights to " << filename << std::endl;
        return;
    }
    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";

    for (const auto& row : fWeights1){ for (double w : row) file << w << " "; file << "\n"; }
    for (double b : fBias1) { file << b << " "; } file << "\n";

    for (const auto& row : fWeights2){ for (double w : row) file << w << " "; file << "\n"; }
    for (double b : fBias2) { file << b << " "; } file << "\n";

    for (double w : fWeights3[0]) { file << w << " "; } file << "\n";
    file << fBias3 << "\n";
    file.close();
    std::cout << "Weights saved to " << filename << std::endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Warning: Could not load weights from " << filename << std::endl;
        return false;
    }
    file >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize,0.0));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size,0.0));
    fWeights3.assign(1, std::vector<double>(fHidden2Size,0.0));
    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);

    for (auto& row : fWeights1) for (double& w : row) file >> w;
    for (double& b : fBias1) file >> b;

    for (auto& row : fWeights2) for (double& w : row) file >> w;
    for (double& b : fBias2) file >> b;

    for (double& w : fWeights3[0]) file >> w;
    file >> fBias3;

    file.close();
    std::cout << "Weights loaded from " << filename << std::endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights(){ fIsQuantized = true; }

// ============================================================================
// PhotonTriggerML implementation
// ============================================================================
PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
    fIsTraining(true),
    fTrainingEpochs(500),
    fTrainingStep(0),
    fBestValidationLoss(1e9),
    fEpochsSinceImprovement(0),
    fEventCount(0),
    fStationCount(0),
    fPhotonLikeCount(0),
    fHadronLikeCount(0),
    fEnergy(0),
    fCoreX(0),
    fCoreY(0),
    fPrimaryId(0),
    fPrimaryType("Unknown"),
    fPhotonScore(0),
    fConfidence(0),
    fDistance(0),
    fStationId(0),
    fIsActualPhoton(false),
    fOutputFile(nullptr),
    fMLTree(nullptr),
    fLogFileName("photon_trigger_ml_physics.log"),
    hPhotonScore(nullptr),
    hPhotonScorePhotons(nullptr),
    hPhotonScoreHadrons(nullptr),
    hConfidence(nullptr),
    hRisetime(nullptr),
    hAsymmetry(nullptr),
    hKurtosis(nullptr),
    hScoreVsEnergy(nullptr),
    hScoreVsDistance(nullptr),
    gROCCurve(nullptr),
    hConfusionMatrix(nullptr),
    hTrainingLoss(nullptr),
    hValidationLoss(nullptr),
    hAccuracyHistory(nullptr),
    fTruePositives(0),
    fFalsePositives(0),
    fTrueNegatives(0),
    fFalseNegatives(0),
    // Start slightly conservative; training/validation will calibrate
    fPhotonThreshold(0.62),
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_physics.root"),
    fWeightsFileName("photon_trigger_weights_physics.txt"),
    fLoadPretrainedWeights(true)
{
    fInstance = this;
    fFeatureMeans.assign(17, 0.0);
    fFeatureStdDevs.assign(17, 1.0);

    std::cout << "\n==========================================\n"
              << "PhotonTriggerML Constructor\n"
              << "Output file: " << fOutputFileName << "\n"
              << "Log file: "    << fLogFileName    << "\n"
              << "==========================================\n";
}

PhotonTriggerML::~PhotonTriggerML(){ std::cout << "PhotonTriggerML Destructor called\n"; }

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init()");
    // open log
    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()){
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    time_t now = time(0);
    fLogFile << "==========================================\n";
    fLogFile << "PhotonTriggerML Physics-Based Version Log\n";
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================\n\n";
    fLogFile.flush();

    // NN
    fNeuralNetwork->Initialize(17, 8, 4);
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)){
        std::cout << "Loaded pre-trained weights from " << fWeightsFileName << std::endl;
        // If weights are loaded, default to inference-only unless user wants online training.
        fIsTraining = false;
    } else {
        std::cout << "Starting with random weights (training mode)\n";
    }

    // ROOT file + objects
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()){
        ERROR("Failed to create output file");
        return eFailure;
    }
    fMLTree = new TTree("MLTree", "PhotonTriggerML Physics Tree");
    fMLTree->Branch("eventId", &fEventCount, "eventId/I");
    fMLTree->Branch("stationId", &fStationId, "stationId/I");
    fMLTree->Branch("energy", &fEnergy, "energy/D");
    fMLTree->Branch("distance", &fDistance, "distance/D");
    fMLTree->Branch("photonScore", &fPhotonScore, "photonScore/D");
    fMLTree->Branch("confidence", &fConfidence, "confidence/D");
    fMLTree->Branch("primaryId", &fPrimaryId, "primaryId/I");
    fMLTree->Branch("primaryType", &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton, "isActualPhoton/O");

    hPhotonScore = new TH1D("hPhotonScore", "ML Photon Score (All);Score;Count", 50, 0, 1);
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons", "ML Score (True Photons);Score;Count", 50, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons", "ML Score (True Hadrons);Score;Count", 50, 0, 1);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence = new TH1D("hConfidence", "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime = new TH1D("hRisetime", "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry = new TH1D("hAsymmetry", "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis = new TH1D("hKurtosis", "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score", 50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 50, 0, 3000, 50, 0, 1);

    hConfusionMatrix = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");

    hTrainingLoss   = new TH1D("hTrainingLoss",   "Training Loss;Batch;Loss",   10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss", 10000, 0, 10000);
    hAccuracyHistory= new TH1D("hAccuracyHistory","Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    std::cout << "Initialization complete!\n\n";
    INFO("PhotonTriggerML initialized successfully.");
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    if (fEventCount % 50 == 1) {
        std::cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n";
        std::cout <<   "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│\n";
        std::cout <<   "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n";
    }

    // Shower info
    fEnergy=0; fCoreX=0; fCoreY=0; fPrimaryId=0; fPrimaryType="Unknown"; fIsActualPhoton=false;

    if (event.HasSimShower()){
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        fPrimaryId = shower.GetPrimaryParticle();
        switch (fPrimaryId){
            case 22: fPrimaryType="photon"; fIsActualPhoton=true; break;
            case 11: case -11: fPrimaryType="electron"; break;
            case 2212: fPrimaryType="proton"; break;
            case 1000026056: fPrimaryType="iron"; break;
            default: fPrimaryType = (fPrimaryId>1000000000) ? "nucleus" : "unknown";
        }
        fParticleTypeCounts[fPrimaryType]++;
        if (shower.GetNSimCores()>0){
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }
        if (fEventCount<=5){
            std::cout << "\nEvent " << fEventCount
                      << ": Energy=" << fEnergy/1e18 << " EeV"
                      << ", Primary=" << fPrimaryType
                      << " (ID=" << fPrimaryId << ")\n";
        }
    }

    // Stations
    if (event.HasSEvent()){
        const sevt::SEvent& sevent = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it=sevent.StationsBegin();
             it!=sevent.StationsEnd(); ++it){
            ProcessStation(*it);
        }
    }

    // Update table + confusion matrix every 10 events
    if (fEventCount % 10 == 0){
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1,1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2,1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1,2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2,2, fTruePositives);
    }

    // Training & validation + threshold calibration (balanced accuracy)
    if (fIsTraining && fTrainingFeatures.size()>=20 && fEventCount%5==0){
        double vloss = TrainNetwork(); // updates hValidationLoss inside

        int batch_num = fEventCount/5;
        hTrainingLoss->SetBinContent(batch_num, vloss);

        int correct = fTruePositives + fTrueNegatives;
        int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total>0){
            double acc = 100.0*correct/total;
            hAccuracyHistory->SetBinContent(batch_num, acc);
        }
        if (vloss < fBestValidationLoss){
            fBestValidationLoss = vloss;
            fEpochsSinceImprovement = 0;
            fNeuralNetwork->SaveWeights("best_" + fWeightsFileName);
        } else {
            fEpochsSinceImprovement++;
        }
    }
    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // station pos -> distance to core
    try{
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        double sx = detStation.GetPosition().GetX(siteCS);
        double sy = detStation.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((sx-fCoreX)*(sx-fCoreX) + (sy-fCoreY)*(sy-fCoreY));
    } catch (...){
        fDistance = -1; return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    for (int p=0;p<3;++p){
        const int pmtId = p + firstPMT;
        if (!station.HasPMT(pmtId)) continue;
        const sevt::PMT& pmt = station.GetPMT(pmtId);

        std::vector<double> trace;
        if (!ExtractTraceData(pmt, trace) || trace.size()!=2048) continue;

        // require some dynamics
        auto mm = std::minmax_element(trace.begin(), trace.end());
        if ((*mm.second - *mm.first) < 10.0) continue;

        // features
        fFeatures = ExtractEnhancedFeatures(trace);

        // minimal physics tag (debug)
        bool physicsPhotonLike = (fFeatures.risetime_10_90 < 150.0 &&
                                  fFeatures.pulse_width     < 300.0 &&
                                  fFeatures.secondary_peak_ratio < 0.30 &&
                                  fFeatures.peak_charge_ratio    > 0.15);

        // fill hists + stats
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        UpdateFeatureStatistics(fFeatures);

        // normalized NN input (+ emphasis)
        std::vector<double> x = NormalizeFeatures(fFeatures);
        x[1]  *= 2.0; // risetime_10_90
        x[3]  *= 2.0; // FWHM
        x[7]  *= 1.5; // peak_charge_ratio
        x[16] *= 1.5; // secondary_peak_ratio

        // NN score
        double ml = fNeuralNetwork->Predict(x, false);

        // ---- Physics prior from raw trace (TOT10 etc.) ----
        auto compute_TOT_frac = [](const std::vector<double>& t, double frac)->double{
            if (t.empty()) return 0.0;
            const int N = (int)t.size();
            int nb = std::min(100, N);
            double base=0.0; for(int i=0;i<nb;++i) base += t[i]; base/=nb;
            int pbin=0; double pval=0.0;
            std::vector<double> s(N,0.0);
            for (int i=0;i<N;++i){
                double v = t[i]-base; if (v<0) v=0;
                s[i]=v; if (v>pval){ pval=v; pbin=i; }
            }
            if (pval<=0) return 0.0;
            const double thr = frac*pval;
            int L=0, R=N-1;
            for (int i=pbin;i>=0;--i){ if (s[i]<=thr){ L=i; break; } }
            for (int i=pbin;i<N; ++i){ if (s[i]<=thr){ R=i; break; } }
            return (R-L)*25.0; // ns
        };
        const double TOT10_ns = compute_TOT_frac(trace, 0.10);

        // elements of prior (scaled into [0,1])
        double em_width   = clamp01((fFeatures.pulse_width - 80.0)/400.0);
        double em_rise    = clamp01((fFeatures.risetime_10_90 - 60.0)/250.0);
        double em_tot10   = clamp01((TOT10_ns - 120.0)/500.0);
        double em_smooth  = clamp01((0.04 - fFeatures.high_freq_content)/0.04); // less HF is more EM-like
        double em_multi   = 0.0;
        if (fFeatures.secondary_peak_ratio < 0.35) em_multi += 0.4;
        if ((int)fFeatures.num_peaks >= 3)        em_multi += 0.3;
        em_multi = clamp01(em_multi);

        double physics_prior = 0.25*em_width + 0.25*em_tot10 + 0.20*em_smooth + 0.20*em_rise + 0.10*em_multi;

        // spike penalties
        double penalty = 0.0;
        if (fFeatures.pulse_width < 80.0 && fFeatures.peak_charge_ratio > 0.40) penalty -= 0.15;
        if (fFeatures.risetime_10_90 < 40.0 && fFeatures.peak_charge_ratio > 0.35) penalty -= 0.10;

        // blended score
        fPhotonScore = clamp01(0.65*ml + 0.35*physics_prior + penalty);
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // --------- diagnostics TSV (append) ----------
        {
            static bool header_written = false;
            std::ofstream tsv("photon_trigger_scores.tsv", std::ios::app);
            if (tsv.is_open()){
                if (!header_written){
                    tsv << "event,station,pmt,primary,dist_m,score,risetime_10_90,Width_ns,Asym,Kurt,"
                           "PeakChargeRatio,SecPeakRatio,TOT10_ns,isPhotonLabel\n";
                    header_written = true;
                }
                tsv << fEventCount << "," << fStationId << "," << pmtId << ","
                    << fPrimaryType << "," << fDistance << ","
                    << std::fixed << std::setprecision(6) << fPhotonScore << ","
                    << fFeatures.risetime_10_90 << "," << fFeatures.pulse_width << ","
                    << fFeatures.asymmetry << "," << fFeatures.kurtosis << ","
                    << fFeatures.peak_charge_ratio << "," << fFeatures.secondary_peak_ratio << ","
                    << TOT10_ns << "," << (fIsActualPhoton?1:0) << "\n";
            }
        }

        // ---- training buffers ----
        bool isValidation = (fStationCount % 10 == 0);
        if (fIsTraining){
            if (!isValidation){
                if (fIsActualPhoton){
                    // slight photon oversampling (2x)
                    std::normal_distribution<> noise(0.0, 0.02);
                    static std::mt19937 gen(1234567u);
                    for (int k=0;k<2;++k){
                        std::vector<double> v = x;
                        for (double& z : v) z += noise(gen);
                        fTrainingFeatures.push_back(v);
                        fTrainingLabels.push_back(1);
                    }
                } else {
                    fTrainingFeatures.push_back(x);
                    fTrainingLabels.push_back(0);
                }
            } else {
                fValidationFeatures.push_back(x);
                fValidationLabels.push_back(fIsActualPhoton?1:0);
            }
        }

        // ---- decision with live governor ----
        static int seen=0, positives=0;
        double thr = fPhotonThreshold;
        bool identifiedAsPhoton = (fPhotonScore > thr);

        seen++; if (identifiedAsPhoton) positives++;
        if (seen>=200){ // very gentle adjustment each 200 stations
            double rate = (positives>0)? (double)positives/seen : 0.0;
            if (rate < 0.08) fPhotonThreshold = std::max(0.45, fPhotonThreshold - 0.02);
            if (rate > 0.70) fPhotonThreshold = std::min(0.85, fPhotonThreshold + 0.02);
            seen=0; positives=0;
        }

        // store result and counters
        MLResult mlr;
        mlr.photonScore = fPhotonScore;
        mlr.identifiedAsPhoton = identifiedAsPhoton;
        mlr.isActualPhoton = fIsActualPhoton;
        mlr.vemCharge = fFeatures.total_charge;
        mlr.features = fFeatures;
        mlr.primaryType = fPrimaryType;
        mlr.confidence = fConfidence;
        fMLResultsMap[fStationId] = mlr;

        if (identifiedAsPhoton){
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
        }

        // ROOT fills
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);
        fMLTree->Fill();

        if ((fIsActualPhoton || physicsPhotonLike) && fStationCount<=100){
            std::cout << "  Station " << fStationId << " PMT " << pmtId
                      << ": Score=" << std::fixed << std::setprecision(3) << fPhotonScore
                      << " (True: " << fPrimaryType << ")"
                      << " Rise="  << fFeatures.risetime_10_90 << "ns"
                      << " Width=" << fFeatures.pulse_width     << "ns"
                      << " TOT10=" << TOT10_ns                  << "ns"
                      << " PhysicsTag=" << (physicsPhotonLike?"YES":"NO") << "\n";
        }

        fStationCount++;
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace)
{
    trace.clear();
    if (pmt.HasFADCTrace()){
        try{
            const auto& t = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            for (int i=0;i<2048;++i) trace.push_back(t[i]);
            return true;
        }catch(...){}
    }
    if (pmt.HasSimData()){
        try{
            const sevt::PMTSimData& sd = pmt.GetSimData();
            if (sd.HasFADCTrace(sevt::StationConstants::eTotal)){
                const auto& t = sd.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                sevt::StationConstants::eTotal);
                for (int i=0;i<2048;++i) trace.push_back(t[i]);
                return true;
            }
        }catch(...){}
    }
    return false;
}

PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
    EnhancedFeatures F;
    const int N = (int)trace.size();
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    int peak_bin=0; double peak_val=0.0, total=0.0;
    std::vector<double> sig(N,0.0);

    // baseline (first 100 bins)
    double base=0.0; int nb = std::min(100, N);
    for (int i=0;i<nb;++i) base += trace[i];
    base /= std::max(1, nb);

    for (int i=0;i<N;++i){
        double v = trace[i]-base; if (v<0) v=0;
        sig[i]=v; total+=v;
        if (v>peak_val){ peak_val=v; peak_bin=i; }
    }
    if (peak_val<5.0 || total<10.0) return F;

    F.peak_amplitude = peak_val/ADC_PER_VEM;
    F.total_charge   = total/ADC_PER_VEM;
    F.peak_charge_ratio = F.peak_amplitude/(F.total_charge + 1e-3);

    const double v10 = 0.10*peak_val;
    const double v50 = 0.50*peak_val;
    const double v90 = 0.90*peak_val;

    int r10=0, r50=0, r90=peak_bin;
    for (int i=peak_bin;i>=0;--i){
        if (sig[i]<=v90 && r90==peak_bin) r90=i;
        if (sig[i]<=v50 && r50==0)       r50=i;
        if (sig[i]<=v10){ r10=i; break; }
    }
    int f90=peak_bin, f10=N-1;
    for (int i=peak_bin;i<N;++i){
        if (sig[i]<=v90 && f90==peak_bin) f90=i;
        if (sig[i]<=v10){ f10=i; break; }
    }

    F.risetime_10_50 = std::abs(r50 - r10)*NS_PER_BIN;
    F.risetime_10_90 = std::abs(r90 - r10)*NS_PER_BIN;
    F.falltime_90_10 = std::abs(f10 - f90)*NS_PER_BIN;

    // FWHM
    int hL=r10, hR=f10;
    const double half = 0.5*peak_val;
    for (int i=r10;i<=peak_bin;++i){ if (sig[i]>=half){ hL=i; break; } }
    for (int i=peak_bin;i<N; ++i){   if (sig[i]<=half){ hR=i; break; } }
    F.pulse_width = std::abs(hR - hL)*NS_PER_BIN;

    // asymmetry
    double rise = F.risetime_10_90, fall = F.falltime_90_10;
    F.asymmetry = (fall - rise)/(fall + rise + 1e-3);

    // moments
    double mt=0.0; for (int i=0;i<N;++i) mt += i*sig[i];
    mt /= (total + 1e-3);
    double var=0.0, skew=0.0, kurt=0.0;
    for (int i=0;i<N;++i){
        double d = i - mt;
        double w = sig[i]/(total + 1e-3);
        var  += d*d*w;
        skew += d*d*d*w;
        kurt += d*d*d*d*w;
    }
    double sd = std::sqrt(var + 1e-3);
    F.time_spread = sd*NS_PER_BIN;
    F.skewness = skew/(sd*sd*sd + 1e-3);
    F.kurtosis = kurt/(var*var + 1e-3) - 3.0;

    // early/late
    int Q = N/4; double early=0.0, late=0.0;
    for (int i=0;i<Q; ++i)        early += sig[i];
    for (int i=3*Q;i<N;++i)       late  += sig[i];
    F.early_fraction = early/(total + 1e-3);
    F.late_fraction  = late /(total + 1e-3);

    // smoothness & HF
    double ssd=0.0; int sc=0;
    for (int i=1;i<N-1;++i){
        if (sig[i]>0.1*peak_val){
            double sec = sig[i+1] - 2*sig[i] + sig[i-1];
            ssd += sec*sec; sc++;
        }
    }
    F.smoothness = std::sqrt(ssd/(sc + 1));
    double hfacc=0.0;
    for (int i=1;i<N-1;++i){ double d = sig[i+1]-sig[i-1]; hfacc += d*d; }
    F.high_freq_content = hfacc/(total*total + 1e-3);

    // peaks
    F.num_peaks = 0.0;
    double secondary=0.0;
    const double pthr = 0.25*peak_val; // slightly stricter
    for (int i=1;i<N-1;++i){
        if (sig[i]>pthr && sig[i]>sig[i-1] && sig[i]>sig[i+1]){
            F.num_peaks += 1.0;
            if (i!=peak_bin && sig[i]>secondary) secondary=sig[i];
        }
    }
    F.secondary_peak_ratio = secondary/(peak_val + 1e-3);
    return F;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& f)
{
    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
    };
    static int n=0; n++;
    for (size_t i=0;i<raw.size();++i){
        double d = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += d/n;
        double d2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += d*d2;
    }
    if (n>1){
        for (size_t i=0;i<raw.size();++i){
            fFeatureStdDevs[i] = std::sqrt(fFeatureStdDevs[i]/(n-1));
            if (fFeatureStdDevs[i] < 1e-3) fFeatureStdDevs[i] = 1.0;
        }
    }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& f)
{
    std::vector<double> raw = {
        f.risetime_10_50, f.risetime_10_90, f.falltime_90_10, f.pulse_width,
        f.asymmetry, f.peak_amplitude, f.total_charge, f.peak_charge_ratio,
        f.smoothness, f.kurtosis, f.skewness, f.early_fraction, f.late_fraction,
        f.time_spread, f.high_freq_content, (double)f.num_peaks, f.secondary_peak_ratio
    };
    std::vector<double> mins = {0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
    std::vector<double> maxs = {500,1000,1000,1000,1,10,100,1,100,20,5,1,1,1000,10,10,1};
    std::vector<double> z; z.reserve(raw.size());
    for (size_t i=0;i<raw.size();++i){
        double v = (raw[i]-mins[i])/(maxs[i]-mins[i] + 1e-3);
        z.push_back(clamp01(v));
    }
    return z;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;

    std::cout << "\n  Training with " << fTrainingFeatures.size() << " samples...";

    std::vector<std::vector<double>> Bx;
    std::vector<int> By;
    const int maxN = std::min(100, (int)fTrainingFeatures.size());
    for (int i=0;i<maxN;++i){ Bx.push_back(fTrainingFeatures[i]); By.push_back(fTrainingLabels[i]); }

    int P = std::count(By.begin(), By.end(), 1);
    int H = (int)By.size() - P;
    std::cout << " (P:" << P << " H:" << H << ")";

    double lr = 0.01;
    double total=0.0;
    for (int e=0;e<10;++e) total += fNeuralNetwork->Train(Bx, By, lr);
    double train_loss = total/10.0;
    std::cout << " Loss: " << std::fixed << std::setprecision(4) << train_loss;

    // ----- validation & balanced-accuracy threshold sweep -----
    double val_loss=0.0;
    if (!fValidationFeatures.empty()){
        const int N = (int)fValidationFeatures.size();
        std::vector<double> preds(N,0.0);
        for (int i=0;i<N;++i) preds[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

        int correct=0;
        for (int i=0;i<N;++i){
            int y = fValidationLabels[i];
            double p = preds[i];
            val_loss += - (y*std::log(p+1e-7) + (1-y)*std::log(1-p+1e-7));
            int yhat = (p > fPhotonThreshold)?1:0;
            if (yhat==y) correct++;
        }
        val_loss /= N;
        double acc = 100.0*correct/N;
        std::cout << " Val: " << val_loss << " (Acc: " << acc << "%)";

        // sweep in [0.45, 0.85]; choose best balanced accuracy with guard on positive rate
        double bestBA=-1.0, bestThr=fPhotonThreshold;
        for (double thr=0.45; thr<=0.85+1e-9; thr+=0.01){
            int TP=0,FP=0,TN=0,FN=0;
            for (int i=0;i<N;++i){
                int y=fValidationLabels[i];
                int yh=(preds[i]>thr)?1:0;
                if (yh==1 && y==1) TP++;
                else if (yh==1 && y==0) FP++;
                else if (yh==0 && y==0) TN++;
                else FN++;
            }
            double tpr = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            double tnr = (TN+FP>0)? (double)TN/(TN+FP) : 0.0;
            double ba  = 0.5*(tpr+tnr);
            double ppr = (TP+FP>0)? (double)(TP+FP)/N : 0.0; // predicted positive rate
            if (ba>bestBA && ppr>0.10 && ppr<0.90){ bestBA=ba; bestThr=thr; }
        }
        double old = fPhotonThreshold;
        // smooth update
        fPhotonThreshold = 0.7*fPhotonThreshold + 0.3*bestThr;
        std::cout << " | Threshold-> " << std::setprecision(3) << fPhotonThreshold
                  << " (BA-opt=" << bestThr << ")";
        if (fLogFile.is_open()){
            fLogFile << "ThresholdCalib old=" << old << " new=" << fPhotonThreshold
                     << " best=" << bestThr << " (balanced-accuracy)\n";
            fLogFile.flush();
        }

        int batch_num = fEventCount/5;
        hValidationLoss->SetBinContent(batch_num, val_loss);
    }
    std::cout << std::endl;

    // keep buffer bounded
    if ((int)fTrainingFeatures.size()>1000){
        fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin()+200);
        fTrainingLabels.erase(fTrainingLabels.begin(), fTrainingLabels.begin()+200);
    }
    fTrainingStep++;
    return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total==0) return;

    double accuracy  = 100.0*(fTruePositives + fTrueNegatives)/total;
    double precision = (fTruePositives + fFalsePositives > 0) ? 100.0*fTruePositives/(fTruePositives + fFalsePositives) : 0.0;
    double recall    = (fTruePositives + fFalseNegatives > 0) ? 100.0*fTruePositives/(fTruePositives + fFalseNegatives) : 0.0;
    double f1        = (precision + recall > 0) ? 2.0*precision*recall/(precision + recall) : 0.0;
    double photon_frac = (fPhotonLikeCount + fHadronLikeCount > 0) ?
                          100.0*fPhotonLikeCount/(fPhotonLikeCount + fHadronLikeCount) : 0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << photon_frac << "%"
              << " │ " << std::setw(7) << accuracy  << "%"
              << " │ " << std::setw(8) << precision << "%"
              << " │ " << std::setw(7) << recall    << "%"
              << " │ " << std::setw(7) << f1        << "%│\n";

    if (fLogFile.is_open()){
        fLogFile << "Event " << fEventCount
                 << " - Acc: " << accuracy
                 << "% Prec: " << precision
                 << "% Rec: "  << recall
                 << "% F1: "   << f1 << "%\n";
        fLogFile.flush();
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    std::cout << "\n==========================================\n"
              << "PERFORMANCE METRICS\n"
              << "==========================================\n";

    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total==0){ std::cout << "No predictions made yet!\n"; return; }

    double accuracy  = 100.0*(fTruePositives + fTrueNegatives)/total;
    double precision = (fTruePositives + fFalsePositives > 0) ? 100.0*fTruePositives/(fTruePositives + fFalsePositives) : 0.0;
    double recall    = (fTruePositives + fFalseNegatives > 0) ? 100.0*fTruePositives/(fTruePositives + fFalseNegatives) : 0.0;
    double f1        = (precision + recall > 0) ? 2.0*precision*recall/(precision + recall) : 0.0;

    std::cout << "Accuracy:  " << std::fixed << std::setprecision(1) << accuracy  << "%\n";
    std::cout << "Precision: " << precision << "%\n";
    std::cout << "Recall:    " << recall    << "%\n";
    std::cout << "F1-Score:  " << f1        << "%\n\n";

    std::cout << "CONFUSION MATRIX:\n";
    std::cout << "                Predicted\n";
    std::cout << "             Hadron   Photon\n";
    std::cout << "Actual Hadron  " << std::setw(6) << fTrueNegatives  << "   " << std::setw(6) << fFalsePositives << "\n";
    std::cout << "       Photon  " << std::setw(6) << fFalseNegatives << "   " << std::setw(6) << fTruePositives  << "\n\n";

    std::cout << "Total Stations: " << fStationCount << "\n";
    std::cout << "Photon-like: "   << fPhotonLikeCount << " ("
              << (100.0*fPhotonLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount)) << "%)\n";
    std::cout << "Hadron-like: "   << fHadronLikeCount << " ("
              << (100.0*fHadronLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount)) << "%)\n\n";

    std::cout << "PARTICLE TYPE BREAKDOWN:\n";
    for (const auto& kv : fParticleTypeCounts){
        std::cout << "  " << kv.first << ": " << kv.second << " events\n";
    }
    std::cout << "==========================================\n";

    if (fLogFile.is_open()){
        fLogFile << "\nFinal Performance Metrics:\n";
        fLogFile << "Accuracy: "  << accuracy  << "%\n";
        fLogFile << "Precision: " << precision << "%\n";
        fLogFile << "Recall: "    << recall    << "%\n";
        fLogFile << "F1-Score: "  << f1        << "%\n";
        fLogFile.flush();
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    std::cout << "\n==========================================\n"
              << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n"
              << "==========================================\n";
    std::cout << "Events processed: "  << fEventCount   << "\n";
    std::cout << "Stations analyzed: " << fStationCount << "\n";

    CalculatePerformanceMetrics();

    // save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    if (fOutputFile){
        fOutputFile->cd();
        if (fMLTree) fMLTree->Write();
        if (hPhotonScore)           hPhotonScore->Write();
        if (hPhotonScorePhotons)    hPhotonScorePhotons->Write();
        if (hPhotonScoreHadrons)    hPhotonScoreHadrons->Write();
        if (hConfidence)            hConfidence->Write();
        if (hRisetime)              hRisetime->Write();
        if (hAsymmetry)             hAsymmetry->Write();
        if (hKurtosis)              hKurtosis->Write();
        if (hScoreVsEnergy)         hScoreVsEnergy->Write();
        if (hScoreVsDistance)       hScoreVsDistance->Write();
        if (hConfusionMatrix)       hConfusionMatrix->Write();
        if (hTrainingLoss)          hTrainingLoss->Write();
        if (hValidationLoss)        hValidationLoss->Write();
        if (hAccuracyHistory)       hAccuracyHistory->Write();
        fOutputFile->Close();
        delete fOutputFile; fOutputFile=nullptr;
    }

    // separate summary ROOT
    {
        TFile fsum("photon_trigger_summary.root","RECREATE");
        if (!fsum.IsZombie()){
            if (hConfusionMatrix) hConfusionMatrix->Write("ConfusionMatrix");
            if (hPhotonScore)     hPhotonScore->Write("PhotonScore_All");
            if (hPhotonScorePhotons) hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
            if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
            if (hConfidence)      hConfidence->Write("Confidence");
            fsum.Close();
        }
    }
    // legacy text summary (kept)
    {
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()){
            const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
            const int total = TP+FP+TN+FN;
            const double acc  = (total>0)? 100.0*(TP+TN)/total : 0.0;
            const double prec = (TP+FP>0)? 100.0*TP/(TP+FP)   : 0.0;
            const double rec  = (TP+FN>0)? 100.0*TP/(TP+FN)   : 0.0;
            const double f1   = (prec+rec>0)? (2.0*prec*rec/(prec+rec)) : 0.0;
            txt << "PhotonTriggerML Summary\n";
            txt << "BaseThreshold=" << fPhotonThreshold << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy="  << acc  << "%  Precision=" << prec << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
        }
    }

    // In-place annotation of existing trace files (skip in signal)
    if (!gCalledFromSignal){
        auto strip_ml = [](const std::string& t)->std::string{
            size_t p=t.find(" [ML:"); return (p==std::string::npos)?t:t.substr(0,p);
        };
        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool{
            if (!h) return false;
            const int n = h->GetNbinsX(); if (n<=0) return false;
            std::vector<double> tr(n); for (int i=1;i<=n;++i) tr[i-1]=h->GetBinContent(i);
            EnhancedFeatures F = ExtractEnhancedFeatures(tr);
            std::vector<double> X = NormalizeFeatures(F);
            if (X.size()>=17){ X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5; }
            double ml = fNeuralNetwork->Predict(X,false);
            // physics penalty as in runtime
            double penalty=0.0;
            if (F.pulse_width<80.0 && F.peak_charge_ratio>0.40) penalty-=0.15;
            if (F.risetime_10_90<40.0 && F.peak_charge_ratio>0.35) penalty-=0.10;
            score = clamp01(ml + penalty);
            isPhoton = (score > fPhotonThreshold);
            return true;
        };
        auto annotate_dir = [&](TDirectory* dir, auto&& self)->void{
            if (!dir) return;
            TIter next(dir->GetListOfKeys()); TKey* key;
            while ((key=(TKey*)next())){
                TObject* obj = dir->Get(key->GetName());
                if (!obj) continue;
                if (obj->InheritsFrom(TDirectory::Class())){
                    self((TDirectory*)obj, self);
                } else if (obj->InheritsFrom(TH1::Class())){
                    TH1* h=(TH1*)obj; double sc=0.0; bool ph=false;
                    if (score_from_hist(h, sc, ph)){
                        std::ostringstream t; t << strip_ml(h->GetTitle()?h->GetTitle():"")
                            << " [ML: " << std::fixed << std::setprecision(3) << sc
                            << " " << (ph?"ML-Photon":"ML-Hadron") << "]";
                        h->SetTitle(t.str().c_str());
                        dir->cd(); h->Write(h->GetName(), TObject::kOverwrite);
                    }
                } else if (obj->InheritsFrom(TCanvas::Class())){
                    TCanvas* c=(TCanvas*)obj; TH1* h=nullptr;
                    if (TList* prim=c->GetListOfPrimitives()){
                        TIter nx(prim); while (TObject* o=nx()){ if (o->InheritsFrom(TH1::Class())){ h=(TH1*)o; break; } }
                    }
                    if (h){
                        double sc=0.0; bool ph=false;
                        if (score_from_hist(h, sc, ph)){
                            std::ostringstream tc; tc << strip_ml(c->GetTitle()?c->GetTitle():"")
                                << " [ML: " << std::fixed << std::setprecision(3) << sc
                                << " " << (ph?"ML-Photon":"ML-Hadron") << "]";
                            c->SetTitle(tc.str().c_str());
                            c->Modified(); c->Update();
                            dir->cd(); c->Write(c->GetName(), TObject::kOverwrite);
                        }
                    }
                }
                delete obj;
            }
        };
        const char* names[]={"pmt_traces_1EeV.root","pmt_Traces_1EeV.root","pmt_traces_1eev.root","pmt_Traces_1eev.root"};
        TFile* f=nullptr;
        for (const char* nm:names){
            f = TFile::Open(nm,"UPDATE");
            if (f && !f->IsZombie()) break;
            if (f){ delete f; f=nullptr; }
        }
        if (f && !f->IsZombie()){
            if (!f->TestBit(TFile::kRecovered)){
                annotate_dir(f, annotate_dir);
                f->Write("", TObject::kOverwrite);
                std::cout << "Annotated ML tags in " << f->GetName() << "\n";
            } else {
                std::cout << "Recovered ROOT file detected; skip annotation.\n";
            }
            f->Close(); delete f;
        } else {
            std::cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    if (fLogFile.is_open()) fLogFile.close();
    std::cout << "==========================================\n";
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish()");
    SaveAndDisplaySummary();
    return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& result)
{
    auto it = fMLResultsMap.find(stationId);
    if (it != fMLResultsMap.end()){ result = it->second; return true; }
    return false;
}

void PhotonTriggerML::ClearMLResults(){ fMLResultsMap.clear(); }

