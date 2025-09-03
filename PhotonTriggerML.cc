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
// Static globals (module-local)
// ============================================================================

PhotonTriggerML* PhotonTriggerML::fInstance = 0;
std::map<int, PhotonTriggerML::MLResult> PhotonTriggerML::fMLResultsMap;

// Safe flag for signal context (skip external file annotation there)
static volatile sig_atomic_t gCalledFromSignal = 0;

// Per-run TSV diagnostics stream (avoids header change)
static std::ofstream gScoresTSV;
static bool gScoresOpen = false;

// Forward: tiny guard to flush log safely
static inline void SafeLog(std::ofstream& os, const std::string& s)
{
    if (os.is_open()) {
        os << s;
        os.flush();
    }
}

// ============================================================================
// Minimal signal handler
// ============================================================================
void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        std::cout << "\n\nInterrupt signal received. Saving PhotonTriggerML data...\n" << std::endl;

        gCalledFromSignal = 1;
        if (PhotonTriggerML::fInstance) {
            try { PhotonTriggerML::fInstance->SaveAndDisplaySummary(); }
            catch (...) { /* swallow to guarantee termination */ }
        }
        std::signal(signal, SIG_DFL);
        std::raise(signal);
        _Exit(0);
    }
}

// ============================================================================
// Small physics‑informed NN (same interface as before)
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.1),
    fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int h1, int h2)
{
    fInputSize    = input_size;
    fHidden1Size  = h1;
    fHidden2Size  = h2;

    std::cout << "Initializing Physics-Based Neural Network: "
              << input_size << " -> " << h1 << " -> " << h2 << " -> 1" << std::endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.assign(h1, std::vector<double>(input_size));
    for (int i = 0; i < h1; ++i)
        for (int j = 0; j < input_size; ++j)
            fWeights1[i][j] = dist(gen) / std::sqrt((double)input_size);

    fWeights2.assign(h2, std::vector<double>(h1));
    for (int i = 0; i < h2; ++i)
        for (int j = 0; j < h1; ++j)
            fWeights2[i][j] = dist(gen) / std::sqrt((double)h1);

    fWeights3.assign(1, std::vector<double>(h2));
    for (int j = 0; j < h2; ++j)
        fWeights3[0][j] = dist(gen) / std::sqrt((double)h2);

    fBias1.assign(h1, 0.0);
    fBias2.assign(h2, 0.0);
    fBias3 = 0.0;

    fMomentum1_w1.assign(h1, std::vector<double>(input_size, 0.0));
    fMomentum2_w1.assign(h1, std::vector<double>(input_size, 0.0));
    fMomentum1_w2.assign(h2, std::vector<double>(h1, 0.0));
    fMomentum2_w2.assign(h2, std::vector<double>(h1, 0.0));
    fMomentum1_w3.assign(1, std::vector<double>(h2, 0.0));
    fMomentum2_w3.assign(1, std::vector<double>(h2, 0.0));
    fMomentum1_b1.assign(h1, 0.0);
    fMomentum2_b1.assign(h1, 0.0);
    fMomentum1_b2.assign(h2, 0.0);
    fMomentum2_b2.assign(h2, 0.0);
    fMomentum1_b3 = 0.0;
    fMomentum2_b3 = 0.0;

    fTimeStep = 0;
    std::cout << "Neural Network initialized for physics-based discrimination!" << std::endl;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if ((int)x.size() != fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size);
    for (int i = 0; i < fHidden1Size; ++i) {
        double s = fBias1[i];
        for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * x[j];
        double a = 1.0 / (1.0 + std::exp(-s));
        if (training && (std::rand() / double(RAND_MAX)) < fDropoutRate) a = 0.0;
        h1[i] = a;
    }

    std::vector<double> h2(fHidden2Size);
    for (int i = 0; i < fHidden2Size; ++i) {
        double s = fBias2[i];
        for (int j = 0; j < fHidden1Size; ++j) s += fWeights2[i][j] * h1[j];
        double a = 1.0 / (1.0 + std::exp(-s));
        if (training && (std::rand() / double(RAND_MAX)) < fDropoutRate) a = 0.0;
        h2[i] = a;
    }

    double out = fBias3;
    for (int j = 0; j < fHidden2Size; ++j) out += fWeights3[0][j] * h2[j];
    return 1.0 / (1.0 + std::exp(-out));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y,
                                             double lr)
{
    if (X.empty() || X.size() != y.size()) return -1.0;

    const int n  = (int)X.size();
    const int h1 = fHidden1Size;
    const int h2 = fHidden2Size;

    // Light class weighting (stabilize precision)
    int nPho = std::count(y.begin(), y.end(), 1);
    double wPho = (nPho > 0 ? 1.5 : 1.0);
    double wHad = 1.0;

    std::vector<std::vector<double>> gw1(h1, std::vector<double>(fInputSize, 0.0));
    std::vector<std::vector<double>> gw2(h2, std::vector<double>(h1, 0.0));
    std::vector<std::vector<double>> gw3(1, std::vector<double>(h2, 0.0));
    std::vector<double> gb1(h1, 0.0), gb2(h2, 0.0);
    double gb3 = 0.0;

    double total = 0.0;

    for (int t = 0; t < n; ++t) {
        const std::vector<double>& x = X[t];
        const int label = y[t];
        const double w = (label == 1 ? wPho : wHad);

        // forward
        std::vector<double> h1a(h1), h1s(h1);
        for (int i = 0; i < h1; ++i) {
            double s = fBias1[i];
            for (int j = 0; j < fInputSize; ++j) s += fWeights1[i][j] * x[j];
            h1s[i] = s;
            h1a[i] = 1.0 / (1.0 + std::exp(-s));
        }

        std::vector<double> h2a(h2), h2s(h2);
        for (int i = 0; i < h2; ++i) {
            double s = fBias2[i];
            for (int j = 0; j < h1; ++j) s += fWeights2[i][j] * h1a[j];
            h2s[i] = s;
            h2a[i] = 1.0 / (1.0 + std::exp(-s));
        }

        double os = fBias3;
        for (int j = 0; j < h2; ++j) os += fWeights3[0][j] * h2a[j];
        double p = 1.0 / (1.0 + std::exp(-os));

        total += -w * (label * std::log(p + 1e-7) + (1 - label) * std::log(1 - p + 1e-7));

        // backward
        double dOut = w * (p - label);

        for (int j = 0; j < h2; ++j) gw3[0][j] += dOut * h2a[j];
        gb3 += dOut;

        std::vector<double> d2(h2, 0.0);
        for (int j = 0; j < h2; ++j) {
            d2[j] = fWeights3[0][j] * dOut;
            d2[j] *= (h2a[j] * (1.0 - h2a[j]));
        }
        for (int i = 0; i < h2; ++i) {
            for (int j = 0; j < h1; ++j) gw2[i][j] += d2[i] * h1a[j];
            gb2[i] += d2[i];
        }

        std::vector<double> d1(h1, 0.0);
        for (int j = 0; j < h1; ++j) {
            double acc = 0.0;
            for (int i = 0; i < h2; ++i) acc += fWeights2[i][j] * d2[i];
            d1[j] = acc * (h1a[j] * (1.0 - h1a[j]));
        }
        for (int i = 0; i < h1; ++i) {
            for (int j = 0; j < fInputSize; ++j) gw1[i][j] += d1[i] * x[j];
            gb1[i] += d1[i];
        }
    }

    // SGD + momentum
    fTimeStep++;
    const double mom = 0.9;

    for (int i = 0; i < h1; ++i) {
        for (int j = 0; j < fInputSize; ++j) {
            double g = gw1[i][j] / n;
            fMomentum1_w1[i][j] = mom * fMomentum1_w1[i][j] - lr * g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        double g = gb1[i] / n;
        fMomentum1_b1[i] = mom * fMomentum1_b1[i] - lr * g;
        fBias1[i] += fMomentum1_b1[i];
    }

    for (int i = 0; i < h2; ++i) {
        for (int j = 0; j < h1; ++j) {
            double g = gw2[i][j] / n;
            fMomentum1_w2[i][j] = mom * fMomentum1_w2[i][j] - lr * g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        double g = gb2[i] / n;
        fMomentum1_b2[i] = mom * fMomentum1_b2[i] - lr * g;
        fBias2[i] += fMomentum1_b2[i];
    }

    for (int j = 0; j < h2; ++j) {
        double g = gw3[0][j] / n;
        fMomentum1_w3[0][j] = mom * fMomentum1_w3[0][j] - lr * g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }

    double g = gb3 / n;
    fMomentum1_b3 = mom * fMomentum1_b3 - lr * g;
    fBias3 += fMomentum1_b3;

    return total / n;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Error: Could not save weights to " << filename << std::endl;
        return;
    }

    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";

    for (const auto& row : fWeights1) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias1) file << b << " ";
    file << "\n";

    for (const auto& row : fWeights2) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias2) file << b << " ";
    file << "\n";

    for (double w : fWeights3[0]) file << w << " ";
    file << "\n";
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

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.assign(1, std::vector<double>(fHidden2Size));
    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);

    for (auto& row : fWeights1)
        for (double& w : row) file >> w;
    for (double& b : fBias1) file >> b;

    for (auto& row : fWeights2)
        for (double& w : row) file >> w;
    for (double& b : fBias2) file >> b;

    for (double& w : fWeights3[0]) file >> w;
    file >> fBias3;

    file.close();
    std::cout << "Weights loaded from " << filename << std::endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights()
{
    fIsQuantized = true;
}

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
    fPhotonThreshold(0.65),
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_physics.root"),
    fWeightsFileName("photon_trigger_weights_physics.txt"),
    fLoadPretrainedWeights(true)
{
    fInstance = this;
    fFeatureMeans.assign(17, 0.0);
    fFeatureStdDevs.assign(17, 1.0);

    std::cout << "\n==========================================" << std::endl;
    std::cout << "PhotonTriggerML Constructor (restored logging + diagnostics)" << std::endl;
    std::cout << "Output file: " << fOutputFileName << std::endl;
    std::cout << "Log file: "    << fLogFileName     << std::endl;
    std::cout << "==========================================" << std::endl;
}

PhotonTriggerML::~PhotonTriggerML()
{
    std::cout << "PhotonTriggerML Destructor called" << std::endl;
}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init() - Starting initialization");

    // Open classic run log
    fLogFile.open(fLogFileName.c_str(), std::ios::out);
    if (!fLogFile.is_open()) {
        ERROR("Failed to open log file: " + fLogFileName);
        return eFailure;
    }
    time_t now = time(0);
    fLogFile << "==========================================" << std::endl;
    fLogFile << "PhotonTriggerML Physics-Based Version Log" << std::endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << std::endl << std::endl;
    fLogFile.flush();

    // TSV diagnostics file (append across runs in same directory)
    if (!gScoresOpen) {
        gScoresTSV.open("photon_trigger_scores.tsv", std::ios::out);
        if (gScoresTSV.is_open()) {
            gScoresTSV << "event\tstation\tpmt\tenergy_eV\tdistance_m"
                       << "\tscore\tthreshold\tisPhoton"
                       << "\trisetime_10_90_ns\twidth_ns\tpeak_ratio\tsec_peak_ratio"
                       << "\tasymmetry\tn_peaks\tcharge_VEM\tTOT10_ns\n";
            gScoresTSV.flush();
            gScoresOpen = true;
        }
    }

    // NN init
    fNeuralNetwork->Initialize(17, 8, 4);
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        std::cout << "Loaded pre-trained weights from " << fWeightsFileName << std::endl;
        fIsTraining = false;
    } else {
        std::cout << "Starting with random weights (training mode)" << std::endl;
    }

    // ROOT output and objects
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create ROOT output: " + fOutputFileName);
        return eFailure;
    }

    fMLTree = new TTree("MLTree", "PhotonTriggerML Physics Tree");
    fMLTree->Branch("eventId",        &fEventCount,    "eventId/I");
    fMLTree->Branch("stationId",      &fStationId,     "stationId/I");
    fMLTree->Branch("energy",         &fEnergy,        "energy/D");
    fMLTree->Branch("distance",       &fDistance,      "distance/D");
    fMLTree->Branch("photonScore",    &fPhotonScore,   "photonScore/D");
    fMLTree->Branch("confidence",     &fConfidence,    "confidence/D");
    fMLTree->Branch("primaryId",      &fPrimaryId,     "primaryId/I");
    fMLTree->Branch("primaryType",    &fPrimaryType);
    fMLTree->Branch("isActualPhoton", &fIsActualPhoton,"isActualPhoton/O");

    hPhotonScore         = new TH1D("hPhotonScore","ML Photon Score (All);Score;Count",50,0,1);
    hPhotonScorePhotons  = new TH1D("hPhotonScorePhotons","ML Score (True Photons);Score;Count",50,0,1);
    hPhotonScoreHadrons  = new TH1D("hPhotonScoreHadrons","ML Score (True Hadrons);Score;Count",50,0,1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons->SetLineColor(kRed);

    hConfidence          = new TH1D("hConfidence","ML Confidence;|Score - 0.5|;Count",50,0,0.5);
    hRisetime            = new TH1D("hRisetime","Rise Time 10-90%;Time [ns];Count",50,0,1000);
    hAsymmetry           = new TH1D("hAsymmetry","Pulse Asymmetry;(fall-rise)/(fall+rise);Count",50,-1,1);
    hKurtosis            = new TH1D("hKurtosis","Signal Kurtosis;Kurtosis;Count",50,-5,20);
    hScoreVsEnergy       = new TH2D("hScoreVsEnergy","Score vs Energy;Energy [eV];Score",50,1e17,1e20,50,0,1);
    hScoreVsDistance     = new TH2D("hScoreVsDistance","Score vs Distance;Distance [m];Score",50,0,3000,50,0,1);

    hConfusionMatrix     = new TH2D("hConfusionMatrix","Confusion Matrix;Predicted;Actual",2,-0.5,1.5,2,-0.5,1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2,"Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2,"Photon");

    hTrainingLoss   = new TH1D("hTrainingLoss",  "Training Loss;Batch;Loss", 10000, 0, 10000);
    hValidationLoss = new TH1D("hValidationLoss","Validation Loss;Batch;Loss",10000, 0, 10000);
    hAccuracyHistory= new TH1D("hAccuracyHistory","Accuracy History;Batch;Accuracy [%]",10000,0,10000);

    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    std::cout << "Initialization complete!" << std::endl;
    INFO("PhotonTriggerML initialized successfully");

    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    // header block every 50 events
    if (fEventCount % 50 == 1) {
        std::cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n";
        std::cout <<   "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│\n";
        std::cout <<   "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n";
    }

    // Shower info
    fEnergy = 0.0; fCoreX = fCoreY = 0.0; fPrimaryId = 0;
    fPrimaryType = "Unknown"; fIsActualPhoton = false;

    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        fPrimaryId = shower.GetPrimaryParticle();
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
            fCoreX = core.GetX(siteCS);
            fCoreY = core.GetY(siteCS);
        }

        if (fEventCount <= 5) {
            std::cout << "\nEvent " << fEventCount
                      << ": Energy=" << fEnergy/1e18 << " EeV"
                      << ", Primary=" << fPrimaryType
                      << " (ID=" << fPrimaryId << ")\n";
        }
    }

    // Stations
    int stationsInEvent = 0;
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
            stationsInEvent++;
        }
    }

    // Running metrics
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1,1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2,1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1,2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2,2, fTruePositives);
    }

    // Periodic training + threshold sweep
    if (fIsTraining && fTrainingFeatures.size() >= 20 && (fEventCount % 5 == 0)) {
        double val_loss = TrainNetwork();

        int batch_num = fEventCount / 5;
        hTrainingLoss->SetBinContent(batch_num, val_loss);

        const int correct = fTruePositives + fTrueNegatives;
        const int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total > 0) {
            const double acc = 100.0 * (double)correct / (double)total;
            hAccuracyHistory->SetBinContent(batch_num, acc);
        }

        if (val_loss < fBestValidationLoss) {
            fBestValidationLoss = val_loss;
            fEpochsSinceImprovement = 0;
            fNeuralNetwork->SaveWeights("best_" + fWeightsFileName);
        } else {
            fEpochsSinceImprovement++;
        }
        (void)val_loss; // used above; keep to satisfy -Werror in all toolchains
    }

    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // position and core distance
    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        const double sx = detStation.GetPosition().GetX(siteCS);
        const double sy = detStation.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((sx - fCoreX)*(sx - fCoreX) + (sy - fCoreY)*(sy - fCoreY));
    } catch (...) {
        fDistance = -1;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    for (int p = 0; p < 3; ++p) {
        const int pmtId = p + firstPMT;
        if (!station.HasPMT(pmtId)) continue;

        const sevt::PMT& pmt = station.GetPMT(pmtId);

        // raw trace
        std::vector<double> trace;
        if (!ExtractTraceData(pmt, trace) || trace.size() != 2048) continue;

        // signal range sanity
        const double maxVal = *std::max_element(trace.begin(), trace.end());
        const double minVal = *std::min_element(trace.begin(), trace.end());
        if ((maxVal - minVal) < 10.0) continue;

        // baseline + simple TOT(>10% of peak) proxy (for diagnostics/penalty)
        double baseline = 0.0;
        for (int i = 0; i < 100; ++i) baseline += trace[i];
        baseline /= 100.0;
        double peak = 0.0;
        int peakBin = 0;
        for (int i = 0; i < (int)trace.size(); ++i) {
            const double s = trace[i] - baseline;
            if (s > peak) { peak = s; peakBin = i; }
        }
        (void)peakBin;
        const double thr10 = 0.1 * peak;
        int tot10_bins = 0;
        for (size_t i = 0; i < trace.size(); ++i) {
            double s = trace[i] - baseline;
            if (s >= thr10) tot10_bins++;
        }
        const double TOT10_ns = 25.0 * (double)tot10_bins; // 25 ns/bin

        // features
        fFeatures = ExtractEnhancedFeatures(trace);

        // feature hists
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);

        UpdateFeatureStatistics(fFeatures);

        // normalization
        std::vector<double> X = NormalizeFeatures(fFeatures);
        // emphasize discriminants
        if (X.size() >= 17) {
            X[1]  *= 2.0; // risetime_10_90
            X[3]  *= 2.0; // pulse_width
            X[7]  *= 1.5; // peak_charge_ratio
            X[16] *= 1.5; // secondary_peak_ratio
        }

        // NN
        double score = fNeuralNetwork->Predict(X, false);

        // spike vs. extended pulse tweaks (small, bounded)
        double adj = 0.0;
        if (fFeatures.pulse_width < 120.0 && fFeatures.peak_charge_ratio > 0.35) adj -= 0.18;
        if (fFeatures.num_peaks <= 2 && fFeatures.secondary_peak_ratio > 0.60)  adj -= 0.08;

        if (TOT10_ns >= 260.0) adj += 0.06;
        else if (TOT10_ns <= 120.0) adj -= 0.06;

        fPhotonScore = std::max(0.0, std::min(1.0, score + adj));
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // training/validation buffers (reduced photon oversampling)
        const bool isValidation = ((fStationCount % 10) == 0);
        if (fIsTraining) {
            if (!isValidation) {
                if (fIsActualPhoton) {
                    for (int k = 0; k < 2; ++k) {
                        std::vector<double> v = X;
                        static std::mt19937 gen(1234567u);
                        std::normal_distribution<> n(0.0, 0.02);
                        for (double& t : v) t += n(gen);
                        fTrainingFeatures.push_back(v);
                        fTrainingLabels.push_back(1);
                    }
                } else {
                    fTrainingFeatures.push_back(X);
                    fTrainingLabels.push_back(0);
                }
            } else {
                fValidationFeatures.push_back(X);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }

        fStationCount++;

        // decision (single threshold; the auto-recalibration happens in TrainNetwork)
        const bool identifiedAsPhoton = (fPhotonScore > fPhotonThreshold);

        // result map (1 entry per station – keep last PMT evaluated)
        MLResult mlr;
        mlr.photonScore        = fPhotonScore;
        mlr.identifiedAsPhoton = identifiedAsPhoton;
        mlr.isActualPhoton     = fIsActualPhoton;
        mlr.vemCharge          = fFeatures.total_charge;
        mlr.features           = fFeatures;
        mlr.primaryType        = fPrimaryType;
        mlr.confidence         = fConfidence;
        fMLResultsMap[fStationId] = mlr;

        // counters
        if (identifiedAsPhoton) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
        }

        // hists
        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);

        // tree
        if (fMLTree) fMLTree->Fill();

        // per-trace diagnostics (TSV)
        if (gScoresOpen && gScoresTSV.is_open()) {
            gScoresTSV << fEventCount << '\t'
                       << fStationId  << '\t'
                       << pmtId       << '\t'
                       << std::setprecision(12) << fEnergy << '\t'
                       << std::setprecision(6)  << fDistance << '\t'
                       << std::setprecision(6)  << fPhotonScore << '\t'
                       << std::setprecision(6)  << fPhotonThreshold << '\t'
                       << (identifiedAsPhoton ? 1 : 0) << '\t'
                       << fFeatures.risetime_10_90 << '\t'
                       << fFeatures.pulse_width    << '\t'
                       << fFeatures.peak_charge_ratio << '\t'
                       << fFeatures.secondary_peak_ratio << '\t'
                       << fFeatures.asymmetry      << '\t'
                       << fFeatures.num_peaks      << '\t'
                       << fFeatures.total_charge   << '\t'
                       << TOT10_ns                 << '\n';
            gScoresTSV.flush();
        }

        // debug print for photon-tag-ish
        if ((fIsActualPhoton || fPhotonScore > 0.6) && fStationCount <= 120) {
            std::cout << "  Station " << fStationId
                      << " PMT " << pmtId
                      << ": Score=" << std::fixed << std::setprecision(3) << fPhotonScore
                      << " (True: " << fPrimaryType << ")"
                      << " Rise="  << fFeatures.risetime_10_90 << "ns"
                      << " Width=" << fFeatures.pulse_width   << "ns"
                      << " Peaks=" << fFeatures.num_peaks
                      << " TOT10=" << TOT10_ns << "ns" << std::endl;
        }
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& trace)
{
    trace.clear();

    if (pmt.HasFADCTrace()) {
        try {
            const auto& t = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            for (int i = 0; i < 2048; ++i) trace.push_back(t[i]);
            return true;
        } catch (...) {}
    }

    if (pmt.HasSimData()) {
        try {
            const sevt::PMTSimData& sim = pmt.GetSimData();
            if (sim.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& t = sim.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                 sevt::StationConstants::eTotal);
                for (int i = 0; i < 2048; ++i) trace.push_back(t[i]);
                return true;
            }
        } catch (...) {}
    }
    return false;
}

PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
    EnhancedFeatures F;
    const int N = (int)trace.size();
    if (N <= 0) return F;

    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    // baseline from first 100
    double base = 0.0;
    const int nb = std::min(100, N);
    for (int i = 0; i < nb; ++i) base += trace[i];
    base /= std::max(1, nb);

    // signal, peak, integral
    int    pk_bin = 0;
    double pk_val = 0.0;
    double sum    = 0.0;
    std::vector<double> sig(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double s = trace[i] - base;
        if (s < 0) s = 0.0;
        sig[i] = s;
        if (s > pk_val) { pk_val = s; pk_bin = i; }
        sum += s;
    }
    if (pk_val < 5.0 || sum < 10.0) return F;

    F.peak_amplitude    = pk_val / ADC_PER_VEM;
    F.total_charge      = sum    / ADC_PER_VEM;
    F.peak_charge_ratio = F.peak_amplitude / (F.total_charge + 1e-3);

    // thresholds
    const double s10 = 0.1 * pk_val;
    const double s50 = 0.5 * pk_val;
    const double s90 = 0.9 * pk_val;

    // rise edges
    int r10 = 0, r50 = 0, r90 = pk_bin;
    for (int i = pk_bin; i >= 0; --i) {
        if (sig[i] <= s90 && r90 == pk_bin) r90 = i;
        if (sig[i] <= s50 && r50 == 0)      r50 = i;
        if (sig[i] <= s10) { r10 = i; break; }
    }

    // fall edges
    int f90 = pk_bin, f10 = N - 1;
    for (int i = pk_bin; i < N; ++i) {
        if (sig[i] <= s90 && f90 == pk_bin) f90 = i;
        if (sig[i] <= s10) { f10 = i; break; }
    }

    F.risetime_10_50   = std::abs(r50 - r10) * NS_PER_BIN;
    F.risetime_10_90   = std::abs(r90 - r10) * NS_PER_BIN;
    F.falltime_90_10   = std::abs(f10 - f90) * NS_PER_BIN;

    // FWHM
    int hL = r10, hR = f10;
    for (int i = r10; i <= pk_bin; ++i) { if (sig[i] >= s50) { hL = i; break; } }
    for (int i = pk_bin; i < N;   ++i) { if (sig[i] <= s50) { hR = i; break; } }
    F.pulse_width = std::abs(hR - hL) * NS_PER_BIN;

    // asymmetry
    const double rise = F.risetime_10_90;
    const double fall = F.falltime_90_10;
    F.asymmetry = (fall - rise) / (fall + rise + 1e-3);

    // moments
    double mu = 0.0;
    for (int i = 0; i < N; ++i) mu += i * sig[i];
    mu /= (sum + 1e-3);

    double var = 0.0, skew = 0.0, kurt = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = i - mu;
        const double w = sig[i] / (sum + 1e-3);
        var  += d*d   * w;
        skew += d*d*d * w;
        kurt += d*d*d*d * w;
    }
    const double sd = std::sqrt(var + 1e-3);
    F.time_spread = sd * NS_PER_BIN;
    F.skewness    = skew / (sd*sd*sd + 1e-3);
    F.kurtosis    = kurt / (var*var + 1e-3) - 3.0;

    // early/late
    const int q = N / 4;
    double early = 0.0, late = 0.0;
    for (int i = 0; i < q; ++i) early += sig[i];
    for (int i = 3*q; i < N; ++i) late += sig[i];
    F.early_fraction = early / (sum + 1e-3);
    F.late_fraction  = late  / (sum + 1e-3);

    // smoothness (2nd deriv energy) over >10% region
    double ssd = 0.0;
    int    cnt = 0;
    for (int i = 1; i < N-1; ++i) {
        if (sig[i] > 0.1*pk_val) {
            const double d2 = sig[i+1] - 2*sig[i] + sig[i-1];
            ssd += d2*d2; cnt++;
        }
    }
    F.smoothness = std::sqrt(ssd / (cnt + 1));

    // high‑freq proxy
    double hf_sum = 0.0;
    for (int i = 1; i < N-1; ++i) {
        const double d = sig[i+1] - sig[i-1];
        hf_sum += d*d;
    }
    F.high_freq_content = hf_sum / (sum*sum + 1e-3);

    // peak counting (stricter to reject noise multi-peaks)
    F.num_peaks = 0;
    double sec = 0.0;
    const double thr = 0.25 * pk_val; // stricter than 0.15
    for (int i = 1; i < N-1; ++i) {
        if (sig[i] > thr && sig[i] > sig[i-1] && sig[i] > sig[i+1]) {
            F.num_peaks++;
            if (i != pk_bin && sig[i] > sec) sec = sig[i];
        }
    }
    F.secondary_peak_ratio = sec / (pk_val + 1e-3);

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

    static int nObs = 0;
    nObs++;

    for (size_t i = 0; i < raw.size(); ++i) {
        double d = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += d / nObs;
        double d2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += d * d2;
    }

    if (nObs > 1) {
        for (size_t i = 0; i < raw.size(); ++i) {
            fFeatureStdDevs[i] = std::sqrt(fFeatureStdDevs[i] / (nObs - 1));
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

    std::vector<double> z;
    z.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        double v = (raw[i] - mins[i]) / (maxs[i] - mins[i] + 1e-3);
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        z.push_back(v);
    }
    return z;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;

    std::ostringstream hdr;
    hdr << "\n  Training with " << fTrainingFeatures.size() << " samples...";
    std::cout << hdr.str();

    std::vector<std::vector<double>> Bx;
    std::vector<int> By;

    const int maxN = std::min(100, (int)fTrainingFeatures.size());
    for (int i = 0; i < maxN; ++i) {
        Bx.push_back(fTrainingFeatures[i]);
        By.push_back(fTrainingLabels[i]);
    }

    const int nPho = std::count(By.begin(), By.end(), 1);
    const int nHad = (int)By.size() - nPho;
    std::cout << " (P:" << nPho << " H:" << nHad << ")";

    const double lr = 0.01;

    double sumLoss = 0.0;
    for (int epoch = 0; epoch < 10; ++epoch) {
        double loss = fNeuralNetwork->Train(Bx, By, lr);
        sumLoss += loss;
    }
    const double train_loss = sumLoss / 10.0;
    std::cout << " Loss: " << std::fixed << std::setprecision(4) << train_loss;

    // validation + threshold sweep for F1
    double val_loss = 0.0;
    if (!fValidationFeatures.empty()) {
        const int N = (int)fValidationFeatures.size();
        std::vector<double> preds(N, 0.5);
        for (int i = 0; i < N; ++i) preds[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

        int correct = 0;
        for (int i = 0; i < N; ++i) {
            const int lab = fValidationLabels[i];
            const double p = preds[i];
            val_loss += - (lab * std::log(p + 1e-7) + (1 - lab) * std::log(1 - p + 1e-7));
            const int predLab = (p > fPhotonThreshold) ? 1 : 0;
            if (predLab == lab) correct++;
        }
        val_loss /= std::max(1, N);
        const double val_acc = 100.0 * (double)correct / std::max(1, N);
        std::cout << " Val: " << val_loss << " (Acc: " << val_acc << "%)";

        // sweep
        double bestF1 = -1.0;
        double bestThr = fPhotonThreshold;
        for (double thr = 0.45; thr <= 0.80 + 1e-12; thr += 0.02) {
            int TP=0,FP=0,TN=0,FN=0;
            for (int i = 0; i < N; ++i) {
                const int lab = fValidationLabels[i];
                const int pr  = (preds[i] > thr) ? 1 : 0;
                if (pr==1 && lab==1) TP++;
                else if (pr==1 && lab==0) FP++;
                else if (pr==0 && lab==0) TN++;
                else FN++;
            }
            const double precision = (TP+FP>0)? (double)TP/(TP+FP) : 0.0;
            const double recall    = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            const double F1        = (precision+recall>0)? 2.0*precision*recall/(precision+recall) : 0.0;
            if (F1 > bestF1) { bestF1 = F1; bestThr = thr; }
        }

        const double oldThr = fPhotonThreshold;
        fPhotonThreshold = 0.8 * fPhotonThreshold + 0.2 * bestThr; // smooth update
        std::cout << " | Threshold-> " << std::fixed << std::setprecision(3)
                  << fPhotonThreshold << " (best=" << bestThr
                  << ", F1=" << std::setprecision(3) << bestF1 << ")";

        // keep loss history in hist
        const int b = fEventCount / 5;
        hValidationLoss->SetBinContent(b, val_loss);

        // log file
        std::ostringstream ln;
        ln << "Recalibrated threshold from " << oldThr
           << " -> " << fPhotonThreshold
           << " using bestF1=" << bestF1 << " at thr=" << bestThr << "\n";
        SafeLog(fLogFile, ln.str());
    }

    std::cout << std::endl;

    // memory cap
    if (fTrainingFeatures.size() > 1000) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin()+200);
        fTrainingLabels.erase(fTrainingLabels.begin(),   fTrainingLabels.begin()+200);
    }

    fTrainingStep++;
    return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total <= 0) return;

    const double accuracy  = 100.0 * (double)(fTruePositives + fTrueNegatives) / (double)total;
    const double precision = (fTruePositives + fFalsePositives > 0) ? 100.0 * (double)fTruePositives / (double)(fTruePositives + fFalsePositives) : 0.0;
    const double recall    = (fTruePositives + fFalseNegatives > 0) ? 100.0 * (double)fTruePositives / (double)(fTruePositives + fFalseNegatives) : 0.0;
    const double f1        = (precision + recall > 0) ? (2.0 * precision * recall) / (precision + recall) : 0.0;

    const double photonFrac = (fPhotonLikeCount + fHadronLikeCount > 0) ?
                               100.0 * (double)fPhotonLikeCount / (double)(fPhotonLikeCount + fHadronLikeCount) : 0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << photonFrac << "%"
              << " │ " << std::setw(7) << accuracy  << "%"
              << " │ " << std::setw(8) << precision << "%"
              << " │ " << std::setw(7) << recall    << "%"
              << " │ " << std::setw(7) << f1        << "%│" << std::endl;

    std::ostringstream ln;
    ln << "Event " << fEventCount
       << " - Acc: " << accuracy
       << "% Prec: " << precision
       << "% Rec: "  << recall
       << "% F1: "   << f1 << "%\n";
    SafeLog(fLogFile, ln.str());
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    std::cout << "\n==========================================\n";
    std::cout << "PERFORMANCE METRICS\n";
    std::cout << "==========================================\n";

    const int total = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
    if (total <= 0) {
        std::cout << "No predictions made yet!\n";
        return;
    }

    const double accuracy  = 100.0 * (double)(fTruePositives + fTrueNegatives) / (double)total;
    const double precision = (fTruePositives + fFalsePositives > 0) ? 100.0 * (double)fTruePositives / (double)(fTruePositives + fFalsePositives) : 0.0;
    const double recall    = (fTruePositives + fFalseNegatives > 0) ? 100.0 * (double)fTruePositives / (double)(fTruePositives + fFalseNegatives) : 0.0;
    const double f1        = (precision + recall > 0) ? (2.0 * precision * recall) / (precision + recall) : 0.0;

    std::cout << "Accuracy:  " << std::fixed << std::setprecision(1) << accuracy << "%\n";
    std::cout << "Precision: " << precision << "%\n";
    std::cout << "Recall:    " << recall    << "%\n";
    std::cout << "F1-Score:  " << f1        << "%\n\n";

    std::cout << "CONFUSION MATRIX:\n";
    std::cout << "                Predicted\n";
    std::cout << "             Hadron   Photon\n";
    std::cout << "Actual Hadron  " << std::setw(6) << fTrueNegatives << "   " << std::setw(6) << fFalsePositives << "\n";
    std::cout << "       Photon  " << std::setw(6) << fFalseNegatives << "   " << std::setw(6) << fTruePositives  << "\n\n";

    std::cout << "Total Stations: " << fStationCount << "\n";
    std::cout << "Photon-like: " << fPhotonLikeCount << " ("
              << 100.0 * (double)fPhotonLikeCount / std::max(1.0, (double)(fPhotonLikeCount + fHadronLikeCount)) << "%)\n";
    std::cout << "Hadron-like: " << fHadronLikeCount << " ("
              << 100.0 * (double)fHadronLikeCount / std::max(1.0, (double)(fPhotonLikeCount + fHadronLikeCount)) << "%)\n";

    std::cout << "\nPARTICLE TYPE BREAKDOWN:\n";
    for (const auto& kv : fParticleTypeCounts)
        std::cout << "  " << kv.first << ": " << kv.second << " events\n";
    std::cout << "==========================================\n";

    if (fLogFile.is_open()) {
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
    std::cout << "\n==========================================\n";
    std::cout << "PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n";
    std::cout << "==========================================\n";
    std::cout << "Events processed: "  << fEventCount  << "\n";
    std::cout << "Stations analyzed: " << fStationCount << "\n";

    CalculatePerformanceMetrics();

    // persist weights (so you can compare runs)
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // ROOT outputs
    if (fOutputFile) {
        fOutputFile->cd();
        if (fMLTree) {
            fMLTree->Write();
            std::cout << "Wrote " << fMLTree->GetEntries() << " entries to tree\n";
        }
        if (hPhotonScore)         hPhotonScore->Write();
        if (hPhotonScorePhotons)  hPhotonScorePhotons->Write();
        if (hPhotonScoreHadrons)  hPhotonScoreHadrons->Write();
        if (hConfidence)          hConfidence->Write();
        if (hRisetime)            hRisetime->Write();
        if (hAsymmetry)           hAsymmetry->Write();
        if (hKurtosis)            hKurtosis->Write();
        if (hScoreVsEnergy)       hScoreVsEnergy->Write();
        if (hScoreVsDistance)     hScoreVsDistance->Write();
        if (hConfusionMatrix)     hConfusionMatrix->Write();
        if (hTrainingLoss)        hTrainingLoss->Write();
        if (hValidationLoss)      hValidationLoss->Write();
        if (hAccuracyHistory)     hAccuracyHistory->Write();

        std::cout << "Histograms written to " << fOutputFileName << "\n";
        fOutputFile->Close();
        delete fOutputFile; fOutputFile = nullptr;
    }

    // compact ROOT summary
    {
        TFile fsum("photon_trigger_summary.root", "RECREATE");
        if (!fsum.IsZombie()) {
            if (hConfusionMatrix)     hConfusionMatrix->Write("ConfusionMatrix");
            if (hPhotonScore)         hPhotonScore->Write("PhotonScore_All");
            if (hPhotonScorePhotons)  hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
            if (hPhotonScoreHadrons)  hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
            if (hConfidence)          hConfidence->Write("Confidence");
            fsum.Close();
        }
    }

    // classic plaintext summary
    {
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
            const int total = TP+FP+TN+FN;
            const double acc  = (total>0)? 100.0*(TP+TN)/total : 0.0;
            const double prec = (TP+FP>0)? 100.0*TP/(TP+FP) : 0.0;
            const double rec  = (TP+FN>0)? 100.0*TP/(TP+FN) : 0.0;
            const double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;

            txt << "PhotonTriggerML Summary\n";
            txt << "BaseThreshold=" << fPhotonThreshold << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy="  << acc  << "%  Precision=" << prec
                << "%  Recall=" << rec  << "%  F1=" << f1 << "%\n";
            txt.close();
        }
    }

    // In-place annotate existing PMT trace ROOT file (skip if signal context)
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& t)->std::string {
            size_t p = t.find(" [ML:");
            return (p==std::string::npos)? t : t.substr(0, p);
        };

        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool {
            if (!h) return false;
            const int n = h->GetNbinsX();
            if (n <= 0) return false;

            std::vector<double> trace(n, 0.0);
            for (int i = 1; i <= n; ++i) trace[i-1] = h->GetBinContent(i);

            EnhancedFeatures F = ExtractEnhancedFeatures(trace);
            std::vector<double> X = NormalizeFeatures(F);
            if (X.size() >= 17) {
                X[1]  *= 2.0; X[3]  *= 2.0; X[7]  *= 1.5; X[16] *= 1.5;
            }
            double ml = fNeuralNetwork->Predict(X, false);

            double adj = 0.0;
            if (F.pulse_width < 120.0 && F.peak_charge_ratio > 0.35) adj -= 0.18;
            if (F.num_peaks <= 2 && F.secondary_peak_ratio > 0.60)  adj -= 0.08;

            score = std::max(0.0, std::min(1.0, ml + adj));
            isPhoton = (score > fPhotonThreshold);
            return true;
        };

        auto annotate_dir = [&](TDirectory* dir, auto&& self)->void {
            if (!dir) return;
            TIter next(dir->GetListOfKeys());
            TKey* key = nullptr;
            while ((key = (TKey*)next())) {
                TObject* obj = dir->Get(key->GetName());
                if (!obj) continue;

                if (obj->InheritsFrom(TDirectory::Class())) {
                    self((TDirectory*)obj, self);
                } else if (obj->InheritsFrom(TH1::Class())) {
                    TH1* h = (TH1*)obj;
                    double sc=0.0; bool ph=false;
                    if (score_from_hist(h, sc, ph)) {
                        const std::string base = strip_ml(h->GetTitle() ? h->GetTitle() : "");
                        std::ostringstream t;
                        t << base << " [ML: " << std::fixed << std::setprecision(3) << sc
                          << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                        h->SetTitle(t.str().c_str());
                        dir->cd();
                        h->Write(h->GetName(), TObject::kOverwrite);
                    }
                } else if (obj->InheritsFrom(TCanvas::Class())) {
                    TCanvas* c = (TCanvas*)obj;
                    TH1* h = nullptr;
                    if (TList* prim = c->GetListOfPrimitives()) {
                        TIter itp(prim);
                        while (TObject* po = itp()) {
                            if (po->InheritsFrom(TH1::Class())) { h = (TH1*)po; break; }
                        }
                    }
                    if (h) {
                        double sc=0.0; bool ph=false;
                        if (score_from_hist(h, sc, ph)) {
                            const std::string baseC = strip_ml(c->GetTitle()?c->GetTitle():"");
                            const std::string baseH = strip_ml(h->GetTitle()?h->GetTitle():"");
                            std::ostringstream tc, th;
                            tc << baseC << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                            th << baseH << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph ? "ML-Photon" : "ML-Hadron") << "]";
                            c->SetTitle(tc.str().c_str());
                            h->SetTitle(th.str().c_str());
                            c->Modified(); c->Update();
                            dir->cd();
                            c->Write(c->GetName(), TObject::kOverwrite);
                        }
                    }
                }
                delete obj;
            }
        };

        const char* candidates[] = {
            "pmt_traces_1EeV.root", "pmt_Traces_1EeV.root",
            "pmt_traces_1eev.root", "pmt_Traces_1eev.root"
        };

        TFile* f = nullptr;
        for (const char* name : candidates) {
            f = TFile::Open(name, "UPDATE");
            if (f && !f->IsZombie()) break;
            if (f) { delete f; f = nullptr; }
        }
        if (f && !f->IsZombie()) {
            if (!f->TestBit(TFile::kRecovered)) {
                annotate_dir(f, annotate_dir);
                f->Write("", TObject::kOverwrite);
                std::cout << "Annotated ML tags in existing trace file: " << f->GetName() << "\n";
            } else {
                std::cout << "Trace file appears recovered; skipping ML annotation for safety.\n";
            }
            f->Close(); delete f;
        } else {
            std::cout << "Trace file not found for ML annotation (skipping).\n";
        }
    }

    if (fLogFile.is_open()) fLogFile.close();
    if (gScoresOpen && gScoresTSV.is_open()) { gScoresTSV.flush(); gScoresTSV.close(); gScoresOpen=false; }

    std::cout << "==========================================\n";
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
    if (it != fMLResultsMap.end()) {
        result = it->second;
        return true;
    }
    return false;
}

void PhotonTriggerML::ClearMLResults()
{
    fMLResultsMap.clear();
}

