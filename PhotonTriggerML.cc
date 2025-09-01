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

void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << endl;
        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); } catch (...) {}
        }
        std::signal(signal, SIG_DFL);
        std::raise(signal);
        _Exit(0);
    }
}

// ============================================================================
// Neural Network (same topology; tuned class weights)
// ============================================================================
PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.1),
    fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize = input_size; fHidden1Size = hidden1_size; fHidden2Size = hidden2_size;
    cout << "Initializing Physics-Based Neural Network: " << input_size << " -> "
         << hidden1_size << " -> " << hidden2_size << " -> 1" << endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.resize(hidden1_size, std::vector<double>(input_size));
    for (int i = 0; i < hidden1_size; ++i)
        for (int j = 0; j < input_size; ++j)
            fWeights1[i][j] = dist(gen) / sqrt(input_size);

    fWeights2.resize(hidden2_size, std::vector<double>(hidden1_size));
    for (int i = 0; i < hidden2_size; ++i)
        for (int j = 0; j < hidden1_size; ++j)
            fWeights2[i][j] = dist(gen) / sqrt(hidden1_size);

    fWeights3.resize(1, std::vector<double>(hidden2_size));
    for (int j = 0; j < hidden2_size; ++j)
        fWeights3[0][j] = dist(gen) / sqrt(hidden2_size);

    fBias1.assign(hidden1_size, 0.0);
    fBias2.assign(hidden2_size, 0.0);
    fBias3 = 0.0;

    fMomentum1_w1.assign(hidden1_size, std::vector<double>(input_size, 0));
    fMomentum2_w1.assign(hidden1_size, std::vector<double>(input_size, 0));
    fMomentum1_w2.assign(hidden2_size, std::vector<double>(hidden1_size, 0));
    fMomentum2_w2.assign(hidden2_size, std::vector<double>(hidden1_size, 0));
    fMomentum1_w3.assign(1, std::vector<double>(hidden2_size, 0));
    fMomentum2_w3.assign(1, std::vector<double>(hidden2_size, 0));

    fMomentum1_b1.assign(hidden1_size, 0);
    fMomentum2_b1.assign(hidden1_size, 0);
    fMomentum1_b2.assign(hidden2_size, 0);
    fMomentum2_b2.assign(hidden2_size, 0);
    fMomentum1_b3 = 0;
    fMomentum2_b3 = 0;
    fTimeStep = 0;

    cout << "Neural Network initialized for physics-based discrimination!" << endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& features, bool training)
{
    if ((int)features.size() != fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size);
    for (int i = 0; i < fHidden1Size; ++i) {
        double s = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * features[j];
        h1[i] = 1.0 / (1.0 + exp(-s));
        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) h1[i] = 0;
    }

    std::vector<double> h2(fHidden2Size);
    for (int i = 0; i < fHidden2Size; ++i) {
        double s = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
        h2[i] = 1.0 / (1.0 + exp(-s));
        if (training && (rand() / double(RAND_MAX)) < fDropoutRate) h2[i] = 0;
    }

    double out = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) out += fWeights3[0][j] * h2[j];
    return 1.0 / (1.0 + exp(-out));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y,
                                             double lr)
{
    if (X.empty() || X.size() != y.size()) return -1.0;

    int B = (int)X.size();
    int nP = std::count(y.begin(), y.end(), 1);

    // Softer class weights (precision-friendly)
    double wP = (nP > 0) ? 1.5 : 1.0;
    double wN = 1.0;

    std::vector<std::vector<double>> gw1(fHidden1Size, std::vector<double>(fInputSize, 0));
    std::vector<std::vector<double>> gw2(fHidden2Size, std::vector<double>(fHidden1Size, 0));
    std::vector<std::vector<double>> gw3(1, std::vector<double>(fHidden2Size, 0));
    std::vector<double> gb1(fHidden1Size, 0), gb2(fHidden2Size, 0);
    double gb3 = 0, total = 0;

    for (int n = 0; n < B; ++n) {
        const auto& in = X[n];
        int label = y[n];
        double w = (label == 1) ? wP : wN;

        std::vector<double> h1(fHidden1Size), h1r(fHidden1Size);
        for (int i = 0; i < fHidden1Size; ++i) {
            double s = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * in[j];
            h1r[i] = s; h1[i] = 1.0 / (1.0 + exp(-s));
        }

        std::vector<double> h2(fHidden2Size), h2r(fHidden2Size);
        for (int i = 0; i < fHidden2Size; ++i) {
            double s = fBias2[i];
            for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
            h2r[i] = s; h2[i] = 1.0 / (1.0 + exp(-s));
        }

        double oraw = fBias3; for (int j=0;j<fHidden2Size;++j) oraw += fWeights3[0][j]*h2[j];
        double p = 1.0 / (1.0 + exp(-oraw));

        total += -w * (label*log(p+1e-7) + (1-label)*log(1-p+1e-7));
        double dOut = w * (p - label);

        for (int j=0;j<fHidden2Size;++j) gw3[0][j] += dOut * h2[j];
        gb3 += dOut;

        std::vector<double> dh2(fHidden2Size);
        for (int j=0;j<fHidden2Size;++j) {
            dh2[j] = fWeights3[0][j]*dOut;
            dh2[j] *= h2[j]*(1-h2[j]);
        }

        for (int i=0;i<fHidden2Size;++i) {
            for (int j=0;j<fHidden1Size;++j) gw2[i][j] += dh2[i]*h1[j];
            gb2[i] += dh2[i];
        }

        std::vector<double> dh1(fHidden1Size, 0);
        for (int j=0;j<fHidden1Size;++j) {
            for (int i=0;i<fHidden2Size;++i) dh1[j] += fWeights2[i][j]*dh2[i];
            dh1[j] *= h1[j]*(1-h1[j]);
        }
        for (int i=0;i<fHidden1Size;++i) {
            for (int j=0;j<fInputSize;++j) gw1[i][j] += dh1[i]*in[j];
            gb1[i] += dh1[i];
        }
    }

    fTimeStep++; double m = 0.9;

    // *** FIXED: proper loop bound prevents infinite loop / segfault ***
    for (int i=0;i<fHidden1Size;++i) {
        for (int j=0; j < fInputSize; ++j) {
            double g = gw1[i][j]/B;
            fMomentum1_w1[i][j] = m*fMomentum1_w1[i][j] - lr*g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        double g = gb1[i]/B;
        fMomentum1_b1[i] = m*fMomentum1_b1[i] - lr*g;
        fBias1[i] += fMomentum1_b1[i];
    }
    for (int i=0;i<fHidden2Size;++i) {
        for (int j=0;j<fHidden1Size;++j) {
            double g = gw2[i][j]/B;
            fMomentum1_w2[i][j] = m*fMomentum1_w2[i][j] - lr*g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        double g = gb2[i]/B;
        fMomentum1_b2[i] = m*fMomentum1_b2[i] - lr*g;
        fBias2[i] += fMomentum1_b2[i];
    }
    for (int j=0;j<fHidden2Size;++j) {
        double g = gw3[0][j]/B;
        fMomentum1_w3[0][j] = m*fMomentum1_w3[0][j] - lr*g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }
    double g = gb3/B;
    fMomentum1_b3 = m*fMomentum1_b3 - lr*g;
    fBias3 += fMomentum1_b3;

    return total / B;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        cout << "Error: Could not save weights to " << filename << endl;
        return;
    }

    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";

    for (const auto& row : fWeights1) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    {
        for (double b : fBias1) { file << b << " "; }
        file << "\n";
    }

    for (const auto& row : fWeights2) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    {
        for (double b : fBias2) { file << b << " "; }
        file << "\n";
    }

    {
        for (double w : fWeights3[0]) { file << w << " "; }
        file << "\n";
    }
    file << fBias3 << "\n";

    file.close();
    cout << "Weights saved to " << filename << endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        cout << "Warning: Could not load weights from " << filename << endl;
        return false;
    }

    file >> fInputSize >> fHidden1Size >> fHidden2Size;
    fWeights1.resize(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.resize(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.resize(1, std::vector<double>(fHidden2Size));
    fBias1.resize(fHidden1Size);
    fBias2.resize(fHidden2Size);

    for (auto& row : fWeights1) for (double& w : row) file >> w;
    for (double& b : fBias1) file >> b;
    for (auto& row : fWeights2) for (double& w : row) file >> w;
    for (double& b : fBias2) file >> b;
    for (double& w : fWeights3[0]) file >> w;
    file >> fBias3;

    file.close();
    cout << "Weights loaded from " << filename << endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights(){ fIsQuantized = true; }

// ============================================================================
// PhotonTriggerML (precision-first + recall fixes)
// ============================================================================
PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
    fIsTraining(true),
    fTrainingEpochs(500),
    fTrainingStep(0),
    fBestValidationLoss(1e9),
    fEpochsSinceImprovement(0),
    fEventCount(0), fStationCount(0),
    fPhotonLikeCount(0), fHadronLikeCount(0),
    fEnergy(0), fCoreX(0), fCoreY(0),
    fPrimaryId(0), fPrimaryType("Unknown"),
    fPhotonScore(0), fConfidence(0), fDistance(0),
    fStationId(0), fIsActualPhoton(false),
    fOutputFile(nullptr), fMLTree(nullptr),
    fLogFileName("photon_trigger_ml_physics.log"),
    hPhotonScore(nullptr), hPhotonScorePhotons(nullptr), hPhotonScoreHadrons(nullptr),
    hConfidence(nullptr), hRisetime(nullptr), hAsymmetry(nullptr), hKurtosis(nullptr),
    hScoreVsEnergy(nullptr), hScoreVsDistance(nullptr), gROCCurve(nullptr),
    hConfusionMatrix(nullptr), hTrainingLoss(nullptr), hValidationLoss(nullptr), hAccuracyHistory(nullptr),
    fTruePositives(0), fFalsePositives(0), fTrueNegatives(0), fFalseNegatives(0),
    fPhotonThreshold(0.65),
    fEnergyMin(1e18), fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_physics.root"),
    fWeightsFileName("photon_trigger_weights_physics.txt"),
    fLoadPretrainedWeights(true)
{
    fInstance = this;
    fFeatureMeans.resize(17, 0.0);
    fFeatureStdDevs.resize(17, 1.0);

    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Constructor (precision-first + recall-safe boosts)" << endl;
    cout << "Output file: " << fOutputFileName << endl;
    cout << "Log file: " << fLogFileName << endl;
    cout << "==========================================" << endl;
}

PhotonTriggerML::~PhotonTriggerML(){ cout << "PhotonTriggerML Destructor called" << endl; }

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization");

    cout << "\n==========================================" << endl;
    cout << "PhotonTriggerML Initialization (PHYSICS)" << endl;
    cout << "==========================================" << endl;

    fLogFile.open(fLogFileName.c_str());
    if (!fLogFile.is_open()) { ERROR("Failed to open log file: " + fLogFileName); return eFailure; }

    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML Physics-Based Version Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << endl << endl;

    fNeuralNetwork->Initialize(17, 8, 4);

    // Warm-start from weights but keep training (fine-tuning)
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        cout << "Loaded pre-trained weights from " << fWeightsFileName << " (fine-tuning enabled)\n";
        fIsTraining = true;
    } else {
        cout << "Starting with random weights (training mode)" << endl;
    }

    // ROOT outputs
    cout << "Creating output file: " << fOutputFileName << endl;
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) { ERROR("Failed to create output file"); return eFailure; }

    cout << "Creating ROOT tree..." << endl;
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

    cout << "Creating histograms..." << endl;
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

    hTrainingLoss = new TH1D("hTrainingLoss", "Training Loss;Batch;Loss", 10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss", 10000, 0, 10000);
    hAccuracyHistory = new TH1D("hAccuracyHistory", "Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    cout << "Initialization complete!" << endl;
    cout << "==========================================" << endl << endl;
    INFO("PhotonTriggerML initialized successfully");
    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    if (fEventCount % 50 == 1) {
        cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐" << endl;
        cout << "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│" << endl;
        cout << "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤" << endl;
    }

    fEnergy = 0; fCoreX = 0; fCoreY = 0; fPrimaryId = 0; fPrimaryType = "Unknown"; fIsActualPhoton = false;

    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy(); fPrimaryId = shower.GetPrimaryParticle();
        switch (fPrimaryId) {
            case 22: fPrimaryType = "photon"; fIsActualPhoton = true; break;
            case 11: case -11: fPrimaryType = "electron"; break;
            case 2212: fPrimaryType = "proton"; break;
            case 1000026056: fPrimaryType = "iron"; break;
            default: fPrimaryType = (fPrimaryId > 1000000000) ? "nucleus" : "unknown";
        }
        fParticleTypeCounts[fPrimaryType]++;
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS); fCoreY = core.GetY(siteCS);
        }
        if (fEventCount <= 5) {
            cout << "\nEvent " << fEventCount
                 << ": Energy=" << fEnergy/1e18 << " EeV"
                 << ", Primary=" << fPrimaryType
                 << " (ID=" << fPrimaryId << ")" << endl;
        }
    }

    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }

    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1, 1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2, 1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1, 2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2, 2, fTruePositives);
    }

    // Training / calibration pass
    if (fIsTraining && fTrainingFeatures.size() >= 20 && fEventCount % 5 == 0) {
        double val_loss = TrainNetwork();
        int batch_num = fEventCount / 5;
        hTrainingLoss->SetBinContent(batch_num, val_loss);

        int correct = fTruePositives + fTrueNegatives;
        int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total > 0) hAccuracyHistory->SetBinContent(batch_num, 100.0 * correct / total);
        if (val_loss < fBestValidationLoss) {
            fBestValidationLoss = val_loss; fEpochsSinceImprovement = 0;
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

    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        double sx = detStation.GetPosition().GetX(siteCS);
        double sy = detStation.GetPosition().GetY(siteCS);
        fDistance = sqrt((sx - fCoreX)*(sx - fCoreX) + (sy - fCoreY)*(sy - fCoreY));
    } catch (...) {
        fDistance = -1; return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();
    for (int p = 0; p < 3; p++) {
        const int pmtId = p + firstPMT;
        if (!station.HasPMT(pmtId)) continue;
        const sevt::PMT& pmt = station.GetPMT(pmtId);

        vector<double> trace_data;
        if (!ExtractTraceData(pmt, trace_data) || trace_data.size() != 2048) continue;

        double maxVal = *max_element(trace_data.begin(), trace_data.end());
        double minVal = *min_element(trace_data.begin(), trace_data.end());
        if (maxVal - minVal < 10.0) continue;

        // Main feature set
        fFeatures = ExtractEnhancedFeatures(trace_data);

        // --------- Additional shape measures ----------
        double baseline = 0; for (int i=0;i<100;i++) baseline += trace_data[i]; baseline/=100.0;
        double peakVal = 0;
        for (int i=0;i<(int)trace_data.size();++i) {
            double s = trace_data[i] - baseline; if (s < 0) s = 0;
            if (s > peakVal) { peakVal = s; }
        }
        const double NS_PER_BIN = 25.0;

        int tot10_bins = 0, totAbs_bins = 0;
        const double thrFrac10 = 0.10 * peakVal;
        const double thrAbs = 15.0;
        for (int i=0;i<(int)trace_data.size();++i) {
            double s = trace_data[i] - baseline; if (s < 0) s = 0;
            if (s >= thrFrac10) tot10_bins++;
            if (s >= thrAbs)    totAbs_bins++;
        }
        double TOT10_ns = tot10_bins * NS_PER_BIN;
        double TOTabs_ns = totAbs_bins * NS_PER_BIN;

        std::vector<int> peakIdx;
        double thrPeak = 0.25 * peakVal;
        for (int i=1;i<(int)trace_data.size()-1;++i) {
            double s = trace_data[i] - baseline; if (s < 0) s = 0;
            double sL = trace_data[i-1] - baseline; if (sL < 0) sL = 0;
            double sR = trace_data[i+1] - baseline; if (sR < 0) sR = 0;
            if (s > thrPeak && s > sL && s > sR) peakIdx.push_back(i);
        }
        double interMean=0, interStd=0;
        if (peakIdx.size() >= 2) {
            std::vector<double> dts; dts.reserve(peakIdx.size()-1);
            for (size_t k=1;k<peakIdx.size();++k) dts.push_back((peakIdx[k]-peakIdx[k-1])*NS_PER_BIN);
            double sum=0; for (double v: dts) sum += v; interMean = (dts.empty()?0:sum/dts.size());
            double vs=0; for (double v: dts) { double d=v-interMean; vs += d*d; }
            interStd = (dts.size()>1)? sqrt(vs/(dts.size()-1)):0;
        }
        double burstiness = (interMean>0)? interStd/(interMean+1e-9) : 0;

        // Feature hists / stats
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        UpdateFeatureStatistics(fFeatures);

        // Normalize + emphasize key physics features
        std::vector<double> normalized = NormalizeFeatures(fFeatures);
        normalized[1]  *= 2.0;  // risetime_10_90
        normalized[3]  *= 2.0;  // pulse_width
        normalized[7]  *= 1.5;  // peak_charge_ratio
        normalized[16] *= 1.5;  // secondary_peak_ratio

        // Raw ML score
        double ml_score = fNeuralNetwork->Predict(normalized, false);

        // Precision-safe muon-spike penalty (guarded by very short TOT)
        double penalty = 0.0;
        if (fFeatures.pulse_width < 100.0 && fFeatures.peak_charge_ratio > 0.38 && TOTabs_ns < 120.0)
            penalty -= 0.15;
        if (fFeatures.num_peaks <= 2 && fFeatures.secondary_peak_ratio > 0.60 && TOTabs_ns < 150.0)
            penalty -= 0.10;

        // Recall boost for EM-like multi-peak clusters
        double boost = 0.0;
        if (fFeatures.num_peaks >= 3 && fFeatures.pulse_width >= 150.0 && fFeatures.pulse_width <= 800.0)
            boost += 0.10;
        if (TOT10_ns >= 200.0) boost += 0.05;
        if (burstiness > 0.35) boost += 0.05;

        fPhotonScore = std::max(0.0, std::min(1.0, ml_score + penalty + boost));
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // Local, distance-aware decision threshold
        double thr = fPhotonThreshold;
        if (fDistance > 1200.0) thr -= 0.05;
        if (fDistance > 2000.0) thr -= 0.08;
        if (thr < 0.35) thr = 0.35;
        if (thr > 0.80) thr = 0.80;

        // Training buffers (reduced oversampling; photons only)
        bool isValidation = (fStationCount % 10 == 0);
        if (fIsTraining) {
            if (!isValidation) {
                if (fIsActualPhoton) {
                    for (int copy = 0; copy < 2; ++copy) {
                        std::vector<double> varied = normalized;
                        static std::mt19937 gen{1234567u};
                        std::normal_distribution<> noise(0, 0.02);
                        for (auto& v : varied) v += noise(gen);
                        fTrainingFeatures.push_back(varied);
                        fTrainingLabels.push_back(1);
                    }
                } else {
                    fTrainingFeatures.push_back(normalized);
                    fTrainingLabels.push_back(0);
                }
            } else {
                fValidationFeatures.push_back(normalized);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }

        fStationCount++;
        bool identifiedAsPhoton = (fPhotonScore > thr);

        MLResult mlResult;
        mlResult.photonScore = fPhotonScore;
        mlResult.identifiedAsPhoton = identifiedAsPhoton;
        mlResult.isActualPhoton = fIsActualPhoton;
        mlResult.vemCharge = fFeatures.total_charge;
        mlResult.features = fFeatures;
        mlResult.primaryType = fPrimaryType;
        mlResult.confidence = fConfidence;
        fMLResultsMap[fStationId] = mlResult;

        if (identifiedAsPhoton) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++; else {
                fFalseNegatives++;
            }
        }

        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);
        fMLTree->Fill();
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace_data)
{
    trace_data.clear();
    if (pmt.HasFADCTrace()) {
        try {
            const auto& tr = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            for (int i=0;i<2048;i++) trace_data.push_back(tr[i]);
            return true;
        } catch (...) {}
    }
    if (pmt.HasSimData()) {
        try {
            const sevt::PMTSimData& sim = pmt.GetSimData();
            if (sim.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& tr = sim.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                  sevt::StationConstants::eTotal);
                for (int i=0;i<2048;i++) trace_data.push_back(tr[i]);
                return true;
            }
        } catch (...) {}
    }
    return false;
}

PhotonTriggerML::EnhancedFeatures PhotonTriggerML::ExtractEnhancedFeatures(
    const std::vector<double>& trace, double)
{
    EnhancedFeatures F;
    const int N = trace.size();
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN = 25.0;

    int peak_bin = 0; double peak_val = 0, total = 0;
    std::vector<double> signal(N);

    double base = 0; for (int i=0;i<100;i++) base += trace[i]; base /= 100.0;

    for (int i=0;i<N;i++) {
        signal[i] = trace[i] - base; if (signal[i] < 0) signal[i] = 0;
        if (signal[i] > peak_val) { peak_val = signal[i]; peak_bin = i; }
        total += signal[i];
    }
    if (peak_val < 5.0 || total < 10.0) return F;

    F.peak_amplitude = peak_val / ADC_PER_VEM;
    F.total_charge   = total    / ADC_PER_VEM;
    F.peak_charge_ratio = F.peak_amplitude / (F.total_charge + 0.001);

    double v10 = 0.1*peak_val, v50 = 0.5*peak_val, v90 = 0.9*peak_val;
    int b10r=0, b50r=0, b90r=peak_bin;
    for (int i=peak_bin; i>=0; --i) {
        if (signal[i] <= v90 && b90r==peak_bin) b90r = i;
        if (signal[i] <= v50 && b50r==0)       b50r = i;
        if (signal[i] <= v10) { b10r = i; break; }
    }
    int b90f=peak_bin, b10f=N-1;
    for (int i=peak_bin; i<N; ++i) {
        if (signal[i] <= v90 && b90f==peak_bin) b90f = i;
        if (signal[i] <= v10) { b10f = i; break; }
    }

    F.risetime_10_50 = std::abs(b50r-b10r)*NS_PER_BIN;
    F.risetime_10_90 = std::abs(b90r-b10r)*NS_PER_BIN;
    F.falltime_90_10 = std::abs(b10f-b90f)*NS_PER_BIN;

    double half = peak_val/2.0;
    int bHalfR=b10r, bHalfF=b10f;
    for (int i=b10r; i<=peak_bin; ++i) if (signal[i] >= half) { bHalfR = i; break; }
    for (int i=peak_bin; i<N; ++i) if (signal[i] <= half) { bHalfF = i; break; }
    F.pulse_width = std::abs(bHalfF-bHalfR)*NS_PER_BIN;

    double rise = F.risetime_10_90, fall = F.falltime_90_10;
    F.asymmetry = (fall - rise) / (fall + rise + 0.001);

    double mean_t = 0; for (int i=0;i<N;i++) mean_t += i*signal[i]; mean_t /= (total + 0.001);
    double var=0, skew=0, kurt=0;
    for (int i=0;i<N;i++) {
        double d = i - mean_t;
        double w = signal[i] / (total + 0.001);
        var  += d*d*w; skew += d*d*d*w; kurt += d*d*d*d*w;
    }
    double sd = sqrt(var + 0.001);
    F.time_spread = sd*NS_PER_BIN;
    F.skewness = skew / (sd*sd*sd + 0.001);
    F.kurtosis = kurt / (var*var + 0.001) - 3.0;

    int q = N/4; double early=0, late=0;
    for (int i=0;i<q;i++) early += signal[i];
    for (int i=3*q;i<N;i++) late  += signal[i];
    F.early_fraction = early/(total+0.001);
    F.late_fraction  = late /(total+0.001);

    double sumsq=0; int cnt=0;
    for (int i=1;i<N-1;i++) if (signal[i] > 0.1*peak_val) { double d2 = signal[i+1] - 2*signal[i] + signal[i-1]; sumsq += d2*d2; cnt++; }
    F.smoothness = sqrt(sumsq/(cnt+1));

    double hf=0; for (int i=1;i<N-1;i++){ double d = signal[i+1]-signal[i-1]; hf += d*d; }
    F.high_freq_content = hf / (total*total + 0.001);

    F.num_peaks = 0; double sec=0; double thr = 0.25*peak_val; // stricter
    for (int i=1;i<N-1;i++) if (signal[i]>thr && signal[i]>signal[i-1] && signal[i]>signal[i+1]) {
        F.num_peaks++; if (i!=peak_bin && signal[i]>sec) sec=signal[i];
    }
    F.secondary_peak_ratio = sec/(peak_val + 0.001);

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

    static int n = 0; n++;
    for (size_t i=0;i<raw.size();++i) {
        double d  = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += d / n;
        double d2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += d * d2;
    }
    if (n>1) {
        for (size_t i=0;i<raw.size();++i) {
            fFeatureStdDevs[i] = sqrt(fFeatureStdDevs[i] / (n-1));
            if (fFeatureStdDevs[i] < 0.001) fFeatureStdDevs[i] = 1.0;
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
    std::vector<double> mins = {0, 0, 0, 0, -1, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0};
    std::vector<double> maxs = {500, 1000, 1000, 1000, 1, 10, 100, 1, 100, 20, 5, 1, 1, 1000, 10, 10, 1};

    std::vector<double> z; z.reserve(raw.size());
    for (size_t i=0;i<raw.size();++i) {
        double v = (raw[i] - mins[i]) / (maxs[i] - mins[i] + 0.001);
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        z.push_back(v);
    }
    return z;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;

    cout << "\n  Training with " << fTrainingFeatures.size() << " samples...";

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    int max_samples = std::min(100, (int)fTrainingFeatures.size());
    for (int i=0;i<max_samples;++i) { X.push_back(fTrainingFeatures[i]); y.push_back(fTrainingLabels[i]); }

    int nP = std::count(y.begin(), y.end(), 1);
    int nN = (int)y.size() - nP;
    cout << " (P:" << nP << " H:" << nN << ")";

    double lr = 0.01, total_loss=0;
    for (int e=0;e<10;++e) total_loss += fNeuralNetwork->Train(X, y, lr);
    double train_loss = total_loss / 10.0;
    cout << " Loss: " << fixed << setprecision(4) << train_loss;

    // ----- Validation & auto-threshold -----
    double val_loss = 0;
    if (!fValidationFeatures.empty()) {
        int N = (int)fValidationFeatures.size();
        std::vector<double> p(N);
        for (int i=0;i<N;++i) p[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

        int correct=0;
        for (int i=0;i<N;++i) {
            int label = fValidationLabels[i];
            val_loss -= label*log(p[i]+1e-7) + (1-label)*log(1-p[i]+1e-7);
            int pred = (p[i] > fPhotonThreshold)? 1:0; if (pred==label) correct++;
        }
        val_loss /= N;
        double acc = 100.0*correct/N;
        cout << " Val: " << val_loss << " (Acc: " << acc << "%)";

        double bestF1=-1, bestThr=fPhotonThreshold;
        for (double thr=0.35; thr<=0.80+1e-9; thr+=0.01) {
            int TP=0,FP=0,TN=0,FN=0;
            for (int i=0;i<N;++i) {
                int label = fValidationLabels[i];
                int pred  = (p[i] > thr)? 1:0;
                if (pred==1 && label==1) TP++; else if (pred==1) FP++;
                else if (label==0) TN++; else FN++;
            }
            double prec=(TP+FP>0)? (double)TP/(TP+FP):0.0;
            double rec =(TP+FN>0)? (double)TP/(TP+FN):0.0;
            double F1  =(prec+rec>0)? 2.0*prec*rec/(prec+rec):0.0;
            double posRate = (double)(TP+FP)/N;
            if (posRate <= 0.65 && F1 > bestF1) { bestF1 = F1; bestThr = thr; }
        }
        double old = fPhotonThreshold;
        fPhotonThreshold = 0.8*fPhotonThreshold + 0.2*bestThr; // smooth
        cout << " | Recalibrated threshold to " << fixed << setprecision(3) << fPhotonThreshold
             << " (best=" << bestThr << ", F1=" << bestF1 << ")";
        int batch_num = fEventCount / 5;
        hValidationLoss->SetBinContent(batch_num, val_loss);
        if (fLogFile.is_open()) {
            fLogFile << "Recalibrated threshold: " << old << " -> " << fPhotonThreshold
                     << " (bestF1=" << bestF1 << ", bestThr=" << bestThr << ")\n";
        }
    }
    cout << endl;

    if (fTrainingFeatures.size() > 1000) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin()+200);
        fTrainingLabels.erase(fTrainingLabels.begin(), fTrainingLabels.begin()+200);
    }
    fTrainingStep++;
    return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) return;

    double acc = 100.0*(fTruePositives + fTrueNegatives) / total;
    double prec = (fTruePositives + fFalsePositives > 0)? 100.0*fTruePositives/(fTruePositives + fFalsePositives):0;
    double rec  = (fTruePositives + fFalseNegatives > 0)? 100.0*fTruePositives/(fTruePositives + fFalseNegatives):0;
    double f1   = (prec+rec>0)? 2*prec*rec/(prec+rec):0;

    double pfrac = (fPhotonLikeCount + fHadronLikeCount > 0)?
                   100.0 * fPhotonLikeCount / (fPhotonLikeCount + fHadronLikeCount) : 0;

    cout << "│ " << setw(6) << fEventCount
         << " │ " << setw(8) << fStationCount
         << " │ " << fixed << setprecision(1) << setw(7) << pfrac << "%"
         << " │ " << setw(7) << acc << "%"
         << " │ " << setw(8) << prec << "%"
         << " │ " << setw(7) << rec << "%"
         << " │ " << setw(7) << f1 << "%│" << endl;

    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount
                 << " - Acc: " << acc << "% Prec: " << prec << "% Rec: " << rec << "% F1: " << f1 << "%\n";
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    cout << "\n==========================================" << endl;
    cout << "PERFORMANCE METRICS" << endl;
    cout << "==========================================" << endl;

    int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total == 0) { cout << "No predictions made yet!" << endl; return; }

    double acc = 100.0*(fTruePositives + fTrueNegatives) / total;
    double prec = (fTruePositives + fFalsePositives > 0)? 100.0*fTruePositives/(fTruePositives + fFalsePositives):0;
    double rec  = (fTruePositives + fFalseNegatives > 0)? 100.0*fTruePositives/(fTruePositives + fFalseNegatives):0;
    double f1   = (prec+rec>0)? 2*prec*rec/(prec+rec):0;

    cout << "Accuracy:  " << fixed << setprecision(1) << acc << "%" << endl;
    cout << "Precision: " << prec << "%" << endl;
    cout << "Recall:    " << rec << "%" << endl;
    cout << "F1-Score:  " << f1 << "%" << endl << endl;

    cout << "CONFUSION MATRIX:" << endl;
    cout << "                Predicted" << endl;
    cout << "             Hadron   Photon" << endl;
    cout << "Actual Hadron  " << setw(6) << fTrueNegatives << "   " << setw(6) << fFalsePositives << endl;
    cout << "       Photon  " << setw(6) << fFalseNegatives << "   " << setw(6) << fTruePositives << endl << endl;

    cout << "Total Stations: " << fStationCount << endl;
    cout << "Photon-like: " << fPhotonLikeCount << " ("
         << 100.0 * fPhotonLikeCount / max(1, fPhotonLikeCount + fHadronLikeCount) << "%)" << endl;
    cout << "Hadron-like: " << fHadronLikeCount << " ("
         << 100.0 * fHadronLikeCount / max(1, fPhotonLikeCount + fHadronLikeCount) << "%)" << endl;

    cout << "\nPARTICLE TYPE BREAKDOWN:" << endl;
    for (const auto& kv : fParticleTypeCounts) cout << "  " << kv.first << ": " << kv.second << " events" << endl;
    cout << "==========================================" << endl;

    if (fLogFile.is_open()) {
        fLogFile << "\nFinal Performance Metrics:\n";
        fLogFile << "Accuracy: " << acc << "%\nPrecision: " << prec
                 << "%\nRecall: " << rec << "%\nF1-Score: " << f1 << "%\n";
    }
}

void PhotonTriggerML::SaveAndDisplaySummary()
{
    cout << "\n==========================================" << endl;
    cout << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)" << endl;
    cout << "==========================================" << endl;

    cout << "Events processed: " << fEventCount << endl;
    cout << "Stations analyzed: " << fStationCount << endl;

    CalculatePerformanceMetrics();

    fNeuralNetwork->SaveWeights(fWeightsFileName);

    if (fOutputFile) {
        fOutputFile->cd();
        if (fMLTree) { fMLTree->Write(); cout << "Wrote " << fMLTree->GetEntries() << " entries to tree" << endl; }
        hPhotonScore->Write(); hPhotonScorePhotons->Write(); hPhotonScoreHadrons->Write();
        hConfidence->Write(); hRisetime->Write(); hAsymmetry->Write(); hKurtosis->Write();
        hScoreVsEnergy->Write(); hScoreVsDistance->Write(); hConfusionMatrix->Write();
        hTrainingLoss->Write(); hValidationLoss->Write(); hAccuracyHistory->Write();
        cout << "Histograms written to " << fOutputFileName << endl;
        fOutputFile->Close(); delete fOutputFile; fOutputFile = nullptr;
    }

    // Quick summary artifacts
    { TFile fsum("photon_trigger_summary.root", "RECREATE");
      if (!fsum.IsZombie()) {
        if (hConfusionMatrix) hConfusionMatrix->Write("ConfusionMatrix");
        if (hPhotonScore)     hPhotonScore->Write("PhotonScore_All");
        if (hPhotonScorePhotons) hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
        if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
        if (hConfidence)      hConfidence->Write("Confidence");
        fsum.Close();
      } }
    { std::ofstream txt("photon_trigger_summary.txt");
      if (txt.is_open()) {
        const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
        const int total=TP+FP+TN+FN;
        const double acc=(total>0)?100.0*(TP+TN)/total:0.0;
        const double prec=(TP+FP>0)?100.0*TP/(TP+FP):0.0;
        const double rec =(TP+FN>0)?100.0*TP/(TP+FN):0.0;
        const double f1  =(prec+rec>0)?(2.0*prec*rec/(prec+rec)):0.0;
        txt << "PhotonTriggerML Summary\nThreshold(final)=" << fPhotonThreshold
            << "\nTP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN
            << "\nAccuracy="<<acc<<"% Precision="<<prec<<"% Recall="<<rec<<"% F1="<<f1<<"%\n";
        txt.close();
      } }

    // In-place annotation of existing trace file (skip on signal)
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& t)->std::string {
            size_t p = t.find(" [ML:"); return (p==std::string::npos)? t : t.substr(0,p);
        };
        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool {
            if (!h) { return false; }
            int n = h->GetNbinsX();
            if (n <= 0) { return false; }

            std::vector<double> tr(n);
            for (int i=1;i<=n;++i) tr[i-1]=h->GetBinContent(i);
            EnhancedFeatures F = ExtractEnhancedFeatures(tr);
            std::vector<double> X = NormalizeFeatures(F);
            if (X.size()>=17) { X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5; }
            double ml = fNeuralNetwork->Predict(X,false);

            double penalty = 0.0;
            if (F.pulse_width < 100.0 && F.peak_charge_ratio > 0.38) { penalty -= 0.15; }
            if (F.num_peaks <= 2 && F.secondary_peak_ratio > 0.60)  { penalty -= 0.10; }

            double boost = 0.0;
            if (F.num_peaks >= 3 && F.pulse_width >= 150.0 && F.pulse_width <= 800.0) { boost += 0.10; }

            score = std::max(0.0, std::min(1.0, ml + penalty + boost));
            isPhoton = (score > fPhotonThreshold);
            return true;
        };
        auto annotate_dir = [&](TDirectory* dir, auto&& self)->void {
            if (!dir) { return; }
            TIter next(dir->GetListOfKeys());
            TKey* key;
            while ((key=(TKey*)next())) {
                TObject* obj = dir->Get(key->GetName()); if (!obj) continue;
                if (obj->InheritsFrom(TDirectory::Class())) {
                    self((TDirectory*)obj, self);
                } else if (obj->InheritsFrom(TH1::Class())) {
                    TH1* h=(TH1*)obj; double sc=0; bool ph=false;
                    if (score_from_hist(h,sc,ph)) {
                        std::string base = strip_ml(h->GetTitle()?h->GetTitle():"");
                        std::ostringstream t; t<<base<<" [ML: "<<fixed<<setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                        h->SetTitle(t.str().c_str()); dir->cd(); h->Write(h->GetName(), TObject::kOverwrite);
                    }
                } else if (obj->InheritsFrom(TCanvas::Class())) {
                    TCanvas* c=(TCanvas*)obj; TH1* h=nullptr;
                    if (TList* prim=c->GetListOfPrimitives()) {
                        TIter nx(prim); while (TObject* po=nx()) { if (po->InheritsFrom(TH1::Class())) { h=(TH1*)po; break; } }
                    }
                    if (h){ double sc=0; bool ph=false;
                        if (score_from_hist(h,sc,ph)) {
                            std::string baseC=strip_ml(c->GetTitle()?c->GetTitle():"");
                            std::string baseH=strip_ml(h->GetTitle()?h->GetTitle():"");
                            std::ostringstream tc,th;
                            tc<<baseC<<" [ML: "<<fixed<<setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                            th<<baseH<<" [ML: "<<fixed<<setprecision(3)<<sc<<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                            c->SetTitle(tc.str().c_str()); h->SetTitle(th.str().c_str());
                            c->Modified(); c->Update(); dir->cd(); c->Write(c->GetName(), TObject::kOverwrite);
                        } }
                }
                // NOTE: do NOT delete obj; ROOT owns objects returned by Get()
            }
        };

        const char* candidates[] = {"pmt_traces_1EeV.root","pmt_Traces_1EeV.root","pmt_traces_1eev.root","pmt_Traces_1eev.root"};
        TFile* f = nullptr; for (const char* n : candidates){ f = TFile::Open(n,"UPDATE"); if (f && !f->IsZombie()) break; if (f){delete f; f=nullptr;}}
        if (f && !f->IsZombie()) {
            if (!f->TestBit(TFile::kRecovered)) { annotate_dir(f, annotate_dir); f->Write("", TObject::kOverwrite); cout<<"Annotated ML tags in existing trace file: "<<f->GetName()<<endl; }
            else cout << "Trace file appears recovered; skipping ML annotation for safety.\n";
            f->Close(); delete f;
        } else {
            cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    if (fLogFile.is_open()) fLogFile.close();
    cout << "==========================================" << endl;
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish() - Normal completion");
    SaveAndDisplaySummary();
    return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& result)
{
    auto it = fMLResultsMap.find(stationId);
    if (it != fMLResultsMap.end()) { result = it->second; return true; }
    return false;
}

void PhotonTriggerML::ClearMLResults(){ fMLResultsMap.clear(); }

