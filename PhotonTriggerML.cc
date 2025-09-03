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

// When finishing from a signal, do not try to reopen/annotate external ROOT files.
static volatile sig_atomic_t gCalledFromSignal = 0;

// Diagnostics TSV (opened in Init, used across module)
static std::ofstream gScoresOut;
static const char* gScoresFileName = "photon_trigger_scores.tsv";

// ============================================================================
// Signal handler: best‑effort summary, then clean termination
// ============================================================================
static void PhotonTriggerMLSignalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTSTP) {
        std::cout << "\n\n[PhotonTriggerML] Caught signal. Saving summary ...\n" << std::endl;
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
// Neural network (small, physics‑guided)
// ============================================================================
PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.10),
    fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int hidden1_size, int hidden2_size)
{
    fInputSize   = input_size;
    fHidden1Size = hidden1_size;
    fHidden2Size = hidden2_size;

    std::cout << "Initializing NN: " << input_size << " -> "
              << hidden1_size << " -> " << hidden2_size << " -> 1" << std::endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    for (int i=0;i<fHidden1Size;++i)
        for (int j=0;j<fInputSize;++j)
            fWeights1[i][j] = dist(gen)/std::sqrt((double)fInputSize);

    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    for (int i=0;i<fHidden2Size;++i)
        for (int j=0;j<fHidden1Size;++j)
            fWeights2[i][j] = dist(gen)/std::sqrt((double)fHidden1Size);

    fWeights3.assign(1, std::vector<double>(fHidden2Size, 0.0));
    for (int j=0;j<fHidden2Size;++j)
        fWeights3[0][j] = dist(gen)/std::sqrt((double)fHidden2Size);

    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);
    fBias3 = 0.0;

    // momentum buffers
    fMomentum1_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fMomentum2_w1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fMomentum1_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fMomentum2_w2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fMomentum1_w3.assign(1, std::vector<double>(fHidden2Size, 0.0));
    fMomentum2_w3.assign(1, std::vector<double>(fHidden2Size, 0.0));

    fMomentum1_b1.assign(fHidden1Size, 0.0);
    fMomentum2_b1.assign(fHidden1Size, 0.0);
    fMomentum1_b2.assign(fHidden2Size, 0.0);
    fMomentum2_b2.assign(fHidden2Size, 0.0);
    fMomentum1_b3 = 0.0;
    fMomentum2_b3 = 0.0;

    fTimeStep = 0;
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if ((int)x.size()!=fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size,0.0);
    for (int i=0;i<fHidden1Size;++i) {
        double s = fBias1[i];
        for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
        double a = 1.0/(1.0+std::exp(-s));
        if (training && (rand()/double(RAND_MAX))<fDropoutRate) a=0.0;
        h1[i]=a;
    }

    std::vector<double> h2(fHidden2Size,0.0);
    for (int i=0;i<fHidden2Size;++i) {
        double s = fBias2[i];
        for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
        double a = 1.0/(1.0+std::exp(-s));
        if (training && (rand()/double(RAND_MAX))<fDropoutRate) a=0.0;
        h2[i]=a;
    }

    double out = fBias3;
    for (int j=0;j<fHidden2Size;++j) out += fWeights3[0][j]*h2[j];
    return 1.0/(1.0+std::exp(-out));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& y,
                                             double lr)
{
    if (X.empty() || X.size()!=y.size()) return -1.0;
    const int n = (int)X.size();

    // light class weighting
    const int n_ph = std::count(y.begin(), y.end(), 1);
    const double w_ph = (n_ph>0)? 1.5 : 1.0;
    const double w_hd = 1.0;

    std::vector<std::vector<double>> gW1(fHidden1Size, std::vector<double>(fInputSize,0.0));
    std::vector<std::vector<double>> gW2(fHidden2Size, std::vector<double>(fHidden1Size,0.0));
    std::vector<std::vector<double>> gW3(1, std::vector<double>(fHidden2Size,0.0));
    std::vector<double> gB1(fHidden1Size,0.0), gB2(fHidden2Size,0.0);
    double gB3 = 0.0;

    double totalLoss=0.0;

    for (int k=0;k<n;++k) {
        const std::vector<double>& x = X[k];
        const int t = y[k];
        const double w = (t==1)? w_ph : w_hd;

        // forward
        std::vector<double> h1r(fHidden1Size,0.0), h1(fHidden1Size,0.0);
        for (int i=0;i<fHidden1Size;++i) {
            double s=fBias1[i];
            for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
            h1r[i]=s; h1[i]=1.0/(1.0+std::exp(-s));
        }
        std::vector<double> h2r(fHidden2Size,0.0), h2(fHidden2Size,0.0);
        for (int i=0;i<fHidden2Size;++i) {
            double s=fBias2[i];
            for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
            h2r[i]=s; h2[i]=1.0/(1.0+std::exp(-s));
        }
        double zr=fBias3;
        for (int j=0;j<fHidden2Size;++j) zr += fWeights3[0][j]*h2[j];
        double yhat = 1.0/(1.0+std::exp(-zr));

        // loss
        totalLoss += -w*( t*std::log(yhat+1e-7) + (1-t)*std::log(1-yhat+1e-7) );

        // backward
        double dOut = w*(yhat - t);
        for (int j=0;j<fHidden2Size;++j) gW3[0][j] += dOut*h2[j];
        gB3 += dOut;

        std::vector<double> dH2(fHidden2Size,0.0);
        for (int j=0;j<fHidden2Size;++j) {
            double tmp = fWeights3[0][j]*dOut;
            dH2[j] = tmp * h2[j]*(1-h2[j]);
        }
        for (int i=0;i<fHidden2Size;++i) {
            for (int j=0;j<fHidden1Size;++j) gW2[i][j] += dH2[i]*h1[j];
            gB2[i] += dH2[i];
        }

        std::vector<double> dH1(fHidden1Size,0.0);
        for (int j=0;j<fHidden1Size;++j) {
            double acc=0.0;
            for (int i=0;i<fHidden2Size;++i) acc += fWeights2[i][j]*dH2[i];
            dH1[j] = acc * h1[j]*(1-h1[j]);
        }
        for (int i=0;i<fHidden1Size;++i) {
            for (int j=0;j<fInputSize;++j) gW1[i][j] += dH1[i]*x[j];
            gB1[i] += dH1[i];
        }
    }

    // momentum SGD
    const double m=0.9;
    for (int i=0;i<fHidden1Size;++i) {
        for (int j=0;j<fInputSize;++j) {
            double g = gW1[i][j]/n;
            fMomentum1_w1[i][j] = m*fMomentum1_w1[i][j] - lr*g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        double gb = gB1[i]/n;
        fMomentum1_b1[i] = m*fMomentum1_b1[i] - lr*gb;
        fBias1[i] += fMomentum1_b1[i];
    }
    for (int i=0;i<fHidden2Size;++i) {
        for (int j=0;j<fHidden1Size;++j) {
            double g = gW2[i][j]/n;
            fMomentum1_w2[i][j] = m*fMomentum1_w2[i][j] - lr*g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        double gb = gB2[i]/n;
        fMomentum1_b2[i] = m*fMomentum1_b2[i] - lr*gb;
        fBias2[i] += fMomentum1_b2[i];
    }
    for (int j=0;j<fHidden2Size;++j) {
        double g = gW3[0][j]/n;
        fMomentum1_w3[0][j] = m*fMomentum1_w3[0][j] - lr*g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }
    double gb = gB3/n;
    fMomentum1_b3 = m*fMomentum1_b3 - lr*gb;
    fBias3 += fMomentum1_b3;

    return totalLoss/n;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& filename)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Error: could not save weights to " << filename << std::endl;
        return;
    }
    file << fInputSize << " " << fHidden1Size << " " << fHidden2Size << "\n";
    for (const auto& row : fWeights1) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias1) { file << b << " "; }
    file << "\n";

    for (const auto& row : fWeights2) {
        for (double w : row) file << w << " ";
        file << "\n";
    }
    for (double b : fBias2) { file << b << " "; }
    file << "\n";

    for (double w : fWeights3[0]) { file << w << " "; }
    file << "\n";
    file << fBias3 << "\n";
    file.close();
    std::cout << "Weights saved to " << filename << std::endl;
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& filename)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cout << "Warning: could not open weights file " << filename << std::endl;
        return false;
    }
    file >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize, 0.0));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size, 0.0));
    fWeights3.assign(1, std::vector<double>(fHidden2Size, 0.0));
    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);

    for (auto& row: fWeights1) for (double& w: row) file >> w;
    for (double& b: fBias1) file >> b;
    for (auto& row: fWeights2) for (double& w: row) file >> w;
    for (double& b: fBias2) file >> b;
    for (double& w: fWeights3[0]) file >> w;
    file >> fBias3;
    file.close();
    std::cout << "Weights loaded from " << filename << std::endl;
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights() { fIsQuantized = true; }

// ============================================================================
// PhotonTriggerML
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
    // Base threshold is auto‑tuned; we also keep near/mid/far for reporting/experiments
    fPhotonThreshold(0.65),
    fEnergyMin(1e18),
    fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_physics.root"),
    fWeightsFileName("photon_trigger_weights_physics.txt"),
    fLoadPretrainedWeights(true)
{
    fInstance = this;
    fFeatureMeans.assign(17,0.0);
    fFeatureStdDevs.assign(17,1.0);
}

PhotonTriggerML::~PhotonTriggerML() {}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init()");

    // open log (restore older style header)
    fLogFile.open(fLogFileName.c_str());
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

    // NN
    fNeuralNetwork->Initialize(17,8,4);
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        std::cout << "Loaded pre-trained weights: " << fWeightsFileName << std::endl;
        fIsTraining = false; // allow pure inference if desired
    } else {
        std::cout << "Random weights: training enabled" << std::endl;
    }

    // ROOT out
    fOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Cannot create output ROOT file");
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

    // histos
    hPhotonScore           = new TH1D("hPhotonScore", "ML Photon Score (All);Score;Count", 50, 0, 1);
    hPhotonScorePhotons    = new TH1D("hPhotonScorePhotons", "ML Score (True Photons);Score;Count", 50, 0, 1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons    = new TH1D("hPhotonScoreHadrons", "ML Score (True Hadrons);Score;Count", 50, 0, 1);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence            = new TH1D("hConfidence", "ML Confidence;|Score - 0.5|;Count", 50, 0, 0.5);
    hRisetime              = new TH1D("hRisetime", "Rise Time 10-90%;Time [ns];Count", 50, 0, 1000);
    hAsymmetry             = new TH1D("hAsymmetry", "Pulse Asymmetry;(fall-rise)/(fall+rise);Count", 50, -1, 1);
    hKurtosis              = new TH1D("hKurtosis", "Signal Kurtosis;Kurtosis;Count", 50, -5, 20);
    hScoreVsEnergy         = new TH2D("hScoreVsEnergy", "Score vs Energy;Energy [eV];Score", 50, 1e17, 1e20, 50, 0, 1);
    hScoreVsDistance       = new TH2D("hScoreVsDistance", "Score vs Distance;Distance [m];Score", 50, 0, 3000, 50, 0, 1);
    hConfusionMatrix       = new TH2D("hConfusionMatrix", "Confusion Matrix;Predicted;Actual", 2, -0.5, 1.5, 2, -0.5, 1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2, "Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1, "Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2, "Photon");
    hTrainingLoss          = new TH1D("hTrainingLoss", "Training Loss;Batch;Loss", 10000, 0, 10000);
    hValidationLoss        = new TH1D("hValidationLoss", "Validation Loss;Batch;Loss", 10000, 0, 10000);
    hAccuracyHistory       = new TH1D("hAccuracyHistory", "Accuracy History;Batch;Accuracy [%]", 10000, 0, 10000);

    // diagnostics TSV
    gScoresOut.open(gScoresFileName, std::ios::out | std::ios::trunc);
    if (gScoresOut.is_open()) {
        gScoresOut << "event\tstation\tpmt\tdistance_m\tprimary\tisPhoton\tml_score\t"
                   << "rise10_90_ns\twidth_ns\tpeakChargeRatio\tsecPeakRatio\tnPeaks\ttotalChargeVEM\n";
        gScoresOut.flush();
    }

    // signals
    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    return eSuccess;
}

VModule::ResultFlag PhotonTriggerML::Run(Event& event)
{
    fEventCount++;
    ClearMLResults();

    // header every 50 events (console table)
    if (fEventCount % 50 == 1) {
        std::cout << "\n┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n";
        std::cout <<   "│ Event  │ Stations │ Photon% │ Accuracy│ Precision│ Recall  │ F1-Score│\n";
        std::cout <<   "├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n";
    }

    // shower info
    fEnergy=0; fCoreX=0; fCoreY=0; fPrimaryId=0; fPrimaryType="Unknown"; fIsActualPhoton=false;
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy     = shower.GetEnergy();
        fPrimaryId  = shower.GetPrimaryParticle();

        switch (fPrimaryId) {
            case 22:             fPrimaryType="photon"; fIsActualPhoton=true; break;
            case 11: case -11:   fPrimaryType="electron"; break;
            case 2212:           fPrimaryType="proton";   break;
            case 1000026056:     fPrimaryType="iron";     break;
            default:             fPrimaryType = (fPrimaryId>1000000000) ? "nucleus" : "unknown";
        }
        fParticleTypeCounts[fPrimaryType]++;

        if (shower.GetNSimCores()>0) {
            const Detector& det = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS); fCoreY = core.GetY(siteCS);
        }
        if (fEventCount<=5) {
            std::cout << "\nEvent " << fEventCount
                      << ": Energy=" << fEnergy/1e18 << " EeV, Primary=" << fPrimaryType
                      << " (ID=" << fPrimaryId << ")\n";
        }
    }

    // stations
    if (event.HasSEvent()) {
        const sevt::SEvent& sev = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it=sev.StationsBegin(); it!=sev.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }

    // every 10 events: update metrics/histos + log/flush
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1,1, fTrueNegatives);
        hConfusionMatrix->SetBinContent(2,1, fFalsePositives);
        hConfusionMatrix->SetBinContent(1,2, fFalseNegatives);
        hConfusionMatrix->SetBinContent(2,2, fTruePositives);
        if (fLogFile.is_open()) fLogFile.flush();
        if (gScoresOut.is_open()) gScoresOut.flush();
    }

    // light training (optional)
    if (fIsTraining && fTrainingFeatures.size() >= 20 && (fEventCount % 5 == 0)) {
        const double val_loss = TrainNetwork();

        const int batch = fEventCount/5;
        hTrainingLoss->SetBinContent(batch, val_loss);

        // accuracy trend
        const int correct = fTruePositives + fTrueNegatives;
        const int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total>0) {
            const double acc = 100.0*correct/total;
            hAccuracyHistory->SetBinContent(batch, acc);
        }

        // best weights
        if (val_loss < fBestValidationLoss) {
            fBestValidationLoss = val_loss;
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

    // station position -> core distance
    try {
        const Detector& det = Detector::GetInstance();
        const sdet::SDetector& sd = det.GetSDetector();
        const sdet::Station& ds = sd.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
        const double sx = ds.GetPosition().GetX(siteCS);
        const double sy = ds.GetPosition().GetY(siteCS);
        fDistance = std::sqrt( (sx-fCoreX)*(sx-fCoreX) + (sy-fCoreY)*(sy-fCoreY) );
    } catch (...) {
        fDistance = -1.0;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    // 3 PMTs
    for (int p=0;p<3;++p) {
        const int pmtId = firstPMT + p;
        if (!station.HasPMT(pmtId)) continue;

        const sevt::PMT& pmt = station.GetPMT(pmtId);

        std::vector<double> trace;
        if (!ExtractTraceData(pmt, trace) || trace.size()!=2048) continue;

        // very small signals → skip
        const auto mm = std::minmax_element(trace.begin(), trace.end());
        if ((*mm.second - *mm.first) < 10.0) continue;

        // features
        fFeatures = ExtractEnhancedFeatures(trace);

        // (debug phys tag – not used in score)
        const bool physicsPhotonLike =
            (fFeatures.risetime_10_90 < 150.0) &&
            (fFeatures.pulse_width     < 300.0) &&
            (fFeatures.secondary_peak_ratio < 0.30) &&
            (fFeatures.peak_charge_ratio    > 0.15);

        // fill feature hists
        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);

        // update running means (for debug only, normalization below is fixed min‑max)
        UpdateFeatureStatistics(fFeatures);

        // normalized input
        std::vector<double> x = NormalizeFeatures(fFeatures);
        // emphasize a few physics‑telling coords
        if (x.size()>=17) {
            x[1]  *= 2.0;   // risetime_10_90
            x[3]  *= 2.0;   // width
            x[7]  *= 1.5;   // peak_charge_ratio
            x[16] *= 1.5;   // secondary_peak_ratio
        }

        // NN score
        double score = fNeuralNetwork->Predict(x, false);

        // simple spike penalties (narrow + peaky)
        double penalty = 0.0;
        if (fFeatures.pulse_width < 120.0 && fFeatures.peak_charge_ratio > 0.35) penalty -= 0.20;
        if (fFeatures.num_peaks <= 2   && fFeatures.secondary_peak_ratio > 0.60) penalty -= 0.10;

        fPhotonScore = std::max(0.0, std::min(1.0, score + penalty));
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // distance‑aware threshold (reported in summary; light static mapping)
        double thr = fPhotonThreshold;
        if      (fDistance >= 1600.0) thr = std::max(0.55, fPhotonThreshold - 0.05); // far: slightly easier
        else if (fDistance < 800.0)   thr = std::min(0.80, fPhotonThreshold + 0.03); // near: slightly stricter

        // training buffers (light; no aggressive oversampling)
        const bool isVal = (fStationCount % 10 == 0);
        if (fIsTraining) {
            if (!isVal) {
                if (fIsActualPhoton) {
                    // 2x copies with tiny Gaussian jitter
                    static std::mt19937 gen(1234567u);
                    std::normal_distribution<> n(0,0.02);
                    for (int k=0;k<2;++k) {
                        std::vector<double> v = x;
                        for (double& z : v) z += n(gen);
                        fTrainingFeatures.push_back(v);
                        fTrainingLabels.push_back(1);
                    }
                } else {
                    fTrainingFeatures.push_back(x);
                    fTrainingLabels.push_back(0);
                }
            } else {
                fValidationFeatures.push_back(x);
                fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);
            }
        }

        fStationCount++;
        const bool isPhoton = (fPhotonScore > thr);

        // store result (map by station)
        MLResult res;
        res.photonScore        = fPhotonScore;
        res.identifiedAsPhoton = isPhoton;
        res.isActualPhoton     = fIsActualPhoton;
        res.vemCharge          = fFeatures.total_charge;
        res.features           = fFeatures;
        res.primaryType        = fPrimaryType;
        res.confidence         = fConfidence;
        fMLResultsMap[fStationId] = res;

        // confusion counters
        if (isPhoton) {
            fPhotonLikeCount++;
            if (fIsActualPhoton) fTruePositives++; else fFalsePositives++;
        } else {
            fHadronLikeCount++;
            if (!fIsActualPhoton) fTrueNegatives++; else fFalseNegatives++;
        }

        // histos
        hPhotonScore->Fill(fPhotonScore);
        hConfidence ->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy  ->Fill(fEnergy,   fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);

        // tree
        if (fMLTree) fMLTree->Fill();

        // diagnostics TSV (flush periodically)
        if (gScoresOut.is_open()) {
            gScoresOut << fEventCount << "\t" << fStationId << "\t" << pmtId << "\t"
                       << fDistance << "\t" << fPrimaryType << "\t"
                       << (fIsActualPhoton?1:0) << "\t" << std::fixed << std::setprecision(6) << fPhotonScore << "\t"
                       << fFeatures.risetime_10_90 << "\t" << fFeatures.pulse_width << "\t"
                       << fFeatures.peak_charge_ratio << "\t" << fFeatures.secondary_peak_ratio << "\t"
                       << fFeatures.num_peaks << "\t" << fFeatures.total_charge << "\n";
        }

        // optional console debug for early events
        if ((fIsActualPhoton || physicsPhotonLike) && fStationCount<=100) {
            std::cout << "  Station " << fStationId << " PMT " << pmtId
                      << "  Score=" << std::fixed << std::setprecision(3) << fPhotonScore
                      << "  Thr=" << thr
                      << "  True=" << fPrimaryType
                      << "  Rise=" << fFeatures.risetime_10_90
                      << "ns  Width=" << fFeatures.pulse_width << "ns"
                      << "  PhysTag=" << (physicsPhotonLike?"YES":"NO") << "\n";
        }
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& out)
{
    out.clear();
    // try FADC
    if (pmt.HasFADCTrace()) {
        try {
            const auto& tr = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            out.reserve(2048);
            for (int i=0;i<2048;++i) out.push_back(tr[i]);
            return true;
        } catch (...) {}
    }
    // try sim data
    if (pmt.HasSimData()) {
        try {
            const sevt::PMTSimData& sd = pmt.GetSimData();
            if (sd.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& tr = sd.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                 sevt::StationConstants::eTotal);
                out.reserve(2048);
                for (int i=0;i<2048;++i) out.push_back(tr[i]);
                return true;
            }
        } catch (...) {}
    }
    return false;
}

PhotonTriggerML::EnhancedFeatures
PhotonTriggerML::ExtractEnhancedFeatures(const std::vector<double>& trace, double)
{
    EnhancedFeatures F; // default zeros
    const int N = (int)trace.size();
    if (N<=0) return F;

    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN  = 25.0;

    // baseline from first 100 bins
    double base=0.0; const int B=std::min(100,N);
    for (int i=0;i<B;++i) base += trace[i];
    base /= (B>0? B:1);

    int peakBin=0; double peakVal=0.0, total=0.0;
    std::vector<double> s(N,0.0);
    for (int i=0;i<N;++i) {
        double v = trace[i]-base; if (v<0) v=0.0;
        s[i]=v; total += v;
        if (v>peakVal) { peakVal=v; peakBin=i; }
    }
    if (peakVal<5.0 || total<10.0) return F;

    F.peak_amplitude   = peakVal/ADC_PER_VEM;
    F.total_charge     = total/ADC_PER_VEM;
    F.peak_charge_ratio= F.peak_amplitude/(F.total_charge+1e-3);

    // rise/fall
    const double p10=0.10*peakVal, p50=0.50*peakVal, p90=0.90*peakVal;
    int r10=0, r50=0, r90=peakBin;
    for (int i=peakBin;i>=0;--i) {
        if (s[i]<=p90 && r90==peakBin) r90=i;
        if (s[i]<=p50 && r50==0)       r50=i;
        if (s[i]<=p10) { r10=i; break; }
    }
    int f90=peakBin, f10=N-1;
    for (int i=peakBin;i<N;++i) {
        if (s[i]<=p90 && f90==peakBin) f90=i;
        if (s[i]<=p10) { f10=i; break; }
    }
    F.risetime_10_50  = std::abs(r50-r10)*NS_PER_BIN;
    F.risetime_10_90  = std::abs(r90-r10)*NS_PER_BIN;
    F.falltime_90_10  = std::abs(f10-f90)*NS_PER_BIN;

    // FWHM
    const double half=0.5*peakVal;
    int whi=r10, wlo=f10;
    for (int i=r10;i<=peakBin;++i)  if (s[i]>=half) { whi=i; break; }
    for (int i=peakBin;i<N;++i)     if (s[i]<=half) { wlo=i; break; }
    F.pulse_width = std::abs(wlo-whi)*NS_PER_BIN;

    // asymmetry
    const double rise = F.risetime_10_90, fall = F.falltime_90_10;
    F.asymmetry = (fall - rise) / (fall + rise + 1e-3);

    // moments
    double mean_t=0.0;
    for (int i=0;i<N;++i) mean_t += i*s[i];
    mean_t /= (total+1e-3);
    double var=0.0, sk=0.0, ku=0.0;
    for (int i=0;i<N;++i) {
        const double d = i-mean_t;
        const double w = s[i]/(total+1e-3);
        var += d*d*w; sk += d*d*d*w; ku += d*d*d*d*w;
    }
    const double sd = std::sqrt(var+1e-3);
    F.time_spread = sd*NS_PER_BIN;
    F.skewness    = sk/std::max(1e-6, sd*sd*sd);
    F.kurtosis    = ku/std::max(1e-6, var*var) - 3.0;

    // early/late fractions
    const int Q=N/4;
    double early=0.0, late=0.0;
    for (int i=0;i<Q;++i)           early += s[i];
    for (int i=3*Q;i<N;++i)         late  += s[i];
    F.early_fraction = early/(total+1e-3);
    F.late_fraction  = late /(total+1e-3);

    // smoothness / HF content
    double sum2=0.0; int cnt=0; double hf=0.0;
    for (int i=1;i<N-1;++i) {
        if (s[i] > 0.1*peakVal) {
            const double sec = s[i+1] - 2*s[i] + s[i-1];
            sum2 += sec*sec; cnt++;
        }
        const double d = s[i+1] - s[i-1];
        hf += d*d;
    }
    F.smoothness        = std::sqrt( sum2 / (cnt + 1) );
    F.high_freq_content = hf / std::max(1e-6, total*total);

    // peak counting (stricter threshold to avoid noise peaks)
    F.num_peaks = 0;
    double secPeak=0.0;
    const double thr = 0.25*peakVal; // stricter than earlier 0.15
    for (int i=1;i<N-1;++i) {
        if (s[i] > thr && s[i] > s[i-1] && s[i] > s[i+1]) {
            F.num_peaks++;
            if (i!=peakBin && s[i]>secPeak) secPeak=s[i];
        }
    }
    F.secondary_peak_ratio = secPeak/(peakVal+1e-3);
    return F;
}

void PhotonTriggerML::UpdateFeatureStatistics(const EnhancedFeatures& F)
{
    std::vector<double> raw = {
        F.risetime_10_50, F.risetime_10_90, F.falltime_90_10, F.pulse_width,
        F.asymmetry, F.peak_amplitude, F.total_charge, F.peak_charge_ratio,
        F.smoothness, F.kurtosis, F.skewness, F.early_fraction, F.late_fraction,
        F.time_spread, F.high_freq_content, (double)F.num_peaks, F.secondary_peak_ratio
    };
    static long long n=0; ++n;

    for (size_t i=0;i<raw.size();++i) {
        const double delta = raw[i] - fFeatureMeans[i];
        fFeatureMeans[i] += delta / (double)n;
        const double delta2 = raw[i] - fFeatureMeans[i];
        fFeatureStdDevs[i] += delta * delta2; // temporarily hold sum of squares
    }
    if (n>1) {
        for (size_t i=0;i<raw.size();++i) {
            double v = fFeatureStdDevs[i] / (double)(n-1);
            fFeatureStdDevs[i] = (v>1e-6) ? std::sqrt(v) : 1.0;
        }
    }
}

std::vector<double> PhotonTriggerML::NormalizeFeatures(const EnhancedFeatures& F)
{
    std::vector<double> raw = {
        F.risetime_10_50, F.risetime_10_90, F.falltime_90_10, F.pulse_width,
        F.asymmetry, F.peak_amplitude, F.total_charge, F.peak_charge_ratio,
        F.smoothness, F.kurtosis, F.skewness, F.early_fraction, F.late_fraction,
        F.time_spread, F.high_freq_content, (double)F.num_peaks, F.secondary_peak_ratio
    };
    const std::vector<double> minv = {0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
    const std::vector<double> maxv = {500,1000,1000,1000,1,10,100,1,100,20,5,1,1,1000,10,10,1};

    std::vector<double> z; z.reserve(raw.size());
    for (size_t i=0;i<raw.size();++i) {
        double v = (raw[i]-minv[i]) / (maxv[i]-minv[i] + 1e-3);
        if (v<0) v=0; if (v>1) v=1;
        z.push_back(v);
    }
    return z;
}

double PhotonTriggerML::TrainNetwork()
{
    if (fTrainingFeatures.empty()) return 1e9;

    std::cout << "\n  Training with " << fTrainingFeatures.size() << " samples...";
    std::vector<std::vector<double>> X;
    std::vector<int> Y;

    const int useN = std::min(100, (int)fTrainingFeatures.size());
    X.reserve(useN); Y.reserve(useN);
    for (int i=0;i<useN;++i) { X.push_back(fTrainingFeatures[i]); Y.push_back(fTrainingLabels[i]); }

    const int nPh = std::count(Y.begin(), Y.end(), 1);
    const int nHd = useN - nPh;
    std::cout << " (P:" << nPh << " H:" << nHd << ")";

    const double lr = 0.01;
    double accLoss=0.0;
    for (int e=0;e<10;++e) accLoss += fNeuralNetwork->Train(X, Y, lr);
    const double trainLoss = accLoss/10.0;
    std::cout << " Loss: " << std::fixed << std::setprecision(4) << trainLoss;

    // validation + mild threshold retune (F1 sweep)
    double val_loss=0.0;
    if (!fValidationFeatures.empty()) {
        const int N=(int)fValidationFeatures.size();
        std::vector<double> pred(N,0.0);
        for (int i=0;i<N;++i) pred[i] = fNeuralNetwork->Predict(fValidationFeatures[i], false);

        int correct=0;
        for (int i=0;i<N;++i) {
            const int t = fValidationLabels[i];
            const double p = pred[i];
            val_loss += - ( t*std::log(p+1e-7) + (1-t)*std::log(1-p+1e-7) );
            if ( (p>fPhotonThreshold && t==1) || (p<=fPhotonThreshold && t==0) ) correct++;
        }
        val_loss /= N;
        const double valAcc = 100.0*correct/N;
        std::cout << " Val: " << val_loss << " (Acc: " << valAcc << "%)";

        double bestF1=-1.0, bestThr=fPhotonThreshold;
        for (double thr=0.40; thr<=0.80+1e-9; thr+=0.02) {
            int TP=0,FP=0,TN=0,FN=0;
            for (int i=0;i<N;++i) {
                const int t = fValidationLabels[i];
                const int y = (pred[i]>thr)?1:0;
                if (y==1 && t==1) TP++;
                else if (y==1 && t==0) FP++;
                else if (y==0 && t==0) TN++;
                else FN++;
            }
            const double P = (TP+FP>0)? (double)TP/(TP+FP) : 0.0;
            const double R = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            const double F = (P+R>0)? 2.0*P*R/(P+R) : 0.0;
            if (F>bestF1) { bestF1=F; bestThr=thr; }
        }
        const double old = fPhotonThreshold;
        fPhotonThreshold = 0.8*fPhotonThreshold + 0.2*bestThr; // smooth move
        std::cout << " | Thr->" << std::setprecision(3) << fPhotonThreshold
                  << " (best=" << bestThr << " F1=" << bestF1 << ")";
        const int b = fEventCount/5;
        hValidationLoss->SetBinContent(b, val_loss);

        if (fLogFile.is_open()) {
            fLogFile << "Recalibrated threshold from " << old << " -> " << fPhotonThreshold
                     << " using validation F1=" << bestF1 << " at thr=" << bestThr << "\n";
            fLogFile.flush();
        }
    }

    // keep buffers bounded
    if (fTrainingFeatures.size()>1000) {
        fTrainingFeatures.erase(fTrainingFeatures.begin(), fTrainingFeatures.begin()+200);
        fTrainingLabels.erase  (fTrainingLabels.begin(),   fTrainingLabels.begin()+200);
    }
    fTrainingStep++;
    std::cout << std::endl;
    return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    const int total = fTruePositives+fTrueNegatives+fFalsePositives+fFalseNegatives;
    if (total==0) return;

    const double accuracy  = 100.0*(fTruePositives+fTrueNegatives)/total;
    const double precision = (fTruePositives+fFalsePositives>0)? 100.0*fTruePositives/(fTruePositives+fFalsePositives) : 0.0;
    const double recall    = (fTruePositives+fFalseNegatives>0)? 100.0*fTruePositives/(fTruePositives+fFalseNegatives) : 0.0;
    const double f1        = (precision+recall>0)? 2.0*precision*recall/(precision+recall) : 0.0;
    const double photonFrac= (fPhotonLikeCount+fHadronLikeCount>0)? 100.0*fPhotonLikeCount/(fPhotonLikeCount+fHadronLikeCount) : 0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << photonFrac << "%"
              << " │ " << std::setw(7) << accuracy << "%"
              << " │ " << std::setw(8) << precision << "%"
              << " │ " << std::setw(7) << recall << "%"
              << " │ " << std::setw(7) << f1 << "%│" << std::endl;

    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount
                 << " - Acc: "  << accuracy
                 << "% Prec: "  << precision
                 << "% Rec: "   << recall
                 << "% F1: "    << f1 << "%\n";
        fLogFile.flush();
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    std::cout << "\n==========================================\n";
    std::cout << "PERFORMANCE METRICS\n";
    std::cout << "==========================================\n";

    const int total = fTruePositives+fTrueNegatives+fFalsePositives+fFalseNegatives;
    if (total==0) { std::cout << "No predictions made yet!\n"; return; }

    const double accuracy  = 100.0*(fTruePositives+fTrueNegatives)/total;
    const double precision = (fTruePositives+fFalsePositives>0)? 100.0*fTruePositives/(fTruePositives+fFalsePositives) : 0.0;
    const double recall    = (fTruePositives+fFalseNegatives>0)? 100.0*fTruePositives/(fTruePositives+fFalseNegatives) : 0.0;
    const double f1        = (precision+recall>0)? 2.0*precision*recall/(precision+recall) : 0.0;

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
    std::cout << "Photon-like: " << fPhotonLikeCount << " ("
              << 100.0*fPhotonLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount) << "%)\n";
    std::cout << "Hadron-like: " << fHadronLikeCount << " ("
              << 100.0*fHadronLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount) << "%)\n\n";

    std::cout << "PARTICLE TYPE BREAKDOWN:\n";
    for (const auto& kv : fParticleTypeCounts) {
        std::cout << "  " << kv.first << ": " << kv.second << " events\n";
    }
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
    std::cout << "Events processed: "  << fEventCount   << "\n";
    std::cout << "Stations analyzed: " << fStationCount << "\n";

    CalculatePerformanceMetrics();

    // always save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // ROOT objects
    if (fOutputFile) {
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

        fOutputFile->Write("", TObject::kOverwrite);
        fOutputFile->Close();
        delete fOutputFile; fOutputFile=nullptr;
    }

    // separate ROOT summary
    {
        TFile fsum("photon_trigger_summary.root", "RECREATE");
        if (!fsum.IsZombie()) {
            if (hConfusionMatrix) hConfusionMatrix->Write("ConfusionMatrix");
            if (hPhotonScore)     hPhotonScore->Write("PhotonScore_All");
            if (hPhotonScorePhotons) hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
            if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
            if (hConfidence)      hConfidence->Write("Confidence");
            fsum.Write("", TObject::kOverwrite);
            fsum.Close();
        }
    }

    // human‑readable text summary (old format preserved)
    {
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
            const int total = TP+FP+TN+FN;
            const double acc  = (total>0)? 100.0*(TP+TN)/total : 0.0;
            const double prec = (TP+FP>0)? 100.0*TP/(TP+FP) : 0.0;
            const double rec  = (TP+FN>0)? 100.0*TP/(TP+FN) : 0.0;
            const double f1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;

            // also report near/mid/far thresholds (simple offsets around base)
            const double thrNear = std::min(0.80, fPhotonThreshold + 0.03);
            const double thrMid  = fPhotonThreshold;
            const double thrFar  = std::max(0.55, fPhotonThreshold - 0.05);

            txt << "PhotonTriggerML Summary\n";
            txt << "BaseThreshold=" << std::setprecision(6) << fPhotonThreshold << "\n";
            txt << "ThresholdNear=" << thrNear
                << " ThresholdMid=" << thrMid
                << " ThresholdFar=" << thrFar << "\n";
            txt << "TP=" << TP << " FP=" << FP << " TN=" << TN << " FN=" << FN << "\n";
            txt << std::fixed << std::setprecision(2);
            txt << "Accuracy=" << acc << "%  Precision=" << prec
                << "%  Recall=" << rec << "%  F1=" << f1 << "%\n";
            txt.close();
        }
    }

    // annotate existing external trace files (only on normal finish)
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& t)->std::string {
            size_t p = t.find(" [ML:"); return (p==std::string::npos)? t : t.substr(0,p);
        };
        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool {
            if (!h) return false;
            const int n = h->GetNbinsX();
            if (n<=0) return false;
            std::vector<double> tr(n,0.0);
            for (int i=1;i<=n;++i) tr[i-1] = h->GetBinContent(i);

            EnhancedFeatures F = ExtractEnhancedFeatures(tr);
            std::vector<double> X = NormalizeFeatures(F);
            if (X.size()>=17) {
                X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5;
            }
            double s = fNeuralNetwork->Predict(X, false);
            double pen=0.0;
            if (F.pulse_width<120.0 && F.peak_charge_ratio>0.35) pen -= 0.20;
            if (F.num_peaks<=2   && F.secondary_peak_ratio>0.60) pen -= 0.10;
            score = std::max(0.0, std::min(1.0, s+pen));
            isPhoton = (score > fPhotonThreshold);
            return true;
        };
        std::function<void(TDirectory*)> annotate_dir = [&](TDirectory* dir){
            if (!dir) return;
            TIter next(dir->GetListOfKeys());
            TKey* key=nullptr;
            while ((key=(TKey*)next())) {
                TObject* obj = dir->Get(key->GetName());
                if (!obj) continue;

                if (obj->InheritsFrom(TDirectory::Class())) {
                    annotate_dir((TDirectory*)obj);
                } else if (obj->InheritsFrom(TH1::Class())) {
                    TH1* h = (TH1*)obj;
                    double sc=0.0; bool ph=false;
                    if (score_from_hist(h,sc,ph)) {
                        const std::string base = strip_ml(h->GetTitle()?h->GetTitle():"");
                        std::ostringstream t;
                        t << base << " [ML: " << std::fixed << std::setprecision(3) << sc
                          << " " << (ph? "ML-Photon":"ML-Hadron") << "]";
                        h->SetTitle(t.str().c_str());
                        dir->cd();
                        h->Write(h->GetName(), TObject::kOverwrite);
                    }
                } else if (obj->InheritsFrom(TCanvas::Class())) {
                    TCanvas* c = (TCanvas*)obj;
                    TH1* h=nullptr;
                    if (TList* prim=c->GetListOfPrimitives()) {
                        TIter nx(prim);
                        while (TObject* po=nx()) {
                            if (po->InheritsFrom(TH1::Class())) { h=(TH1*)po; break; }
                        }
                    }
                    if (h) {
                        double sc=0.0; bool ph=false;
                        if (score_from_hist(h,sc,ph)) {
                            const std::string baseC = strip_ml(c->GetTitle()?c->GetTitle():"");
                            const std::string baseH = strip_ml(h->GetTitle()?h->GetTitle():"");
                            std::ostringstream tc, th;
                            tc << baseC << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph? "ML-Photon":"ML-Hadron") << "]";
                            th << baseH << " [ML: " << std::fixed << std::setprecision(3) << sc
                               << " " << (ph? "ML-Photon":"ML-Hadron") << "]";
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

        const char* cands[] = {"pmt_traces_1EeV.root", "pmt_Traces_1EeV.root",
                               "pmt_traces_1eev.root", "pmt_Traces_1eev.root"};
        TFile* f=nullptr;
        for (const char* nm : cands) {
            f = TFile::Open(nm, "UPDATE");
            if (f && !f->IsZombie()) break;
            if (f) { delete f; f=nullptr; }
        }
        if (f && !f->IsZombie()) {
            if (!f->TestBit(TFile::kRecovered)) {
                annotate_dir(f);
                f->Write("", TObject::kOverwrite);
                std::cout << "Annotated ML tags in " << f->GetName() << "\n";
            } else {
                std::cout << "Recovered ROOT file detected; skip annotation.\n";
            }
            f->Close(); delete f;
        } else {
            std::cout << "Trace file for annotation not found (skipping).\n";
        }
    }

    if (gScoresOut.is_open()) { gScoresOut.flush(); gScoresOut.close(); }

    if (fLogFile.is_open()) { fLogFile.flush(); fLogFile.close(); }

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
    if (it!=fMLResultsMap.end()) { result = it->second; return true; }
    return false;
}

void PhotonTriggerML::ClearMLResults()
{
    fMLResultsMap.clear();
}

