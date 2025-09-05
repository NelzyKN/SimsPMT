// PhotonTriggerML.cc  —  precision-first, auto-calibrated threshold + restored logs
// Drop-in replacement compatible with the existing PhotonTriggerML.h

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

// Append-only scores TSV (kept outside the class to avoid header changes)
static std::ofstream gScoresTSV;
static bool gScoresTSVOpen = false;

// Small helper to safely open TSV once (idempotent)
static void EnsureScoresTSV()
{
    if (!gScoresTSVOpen) {
        gScoresTSV.open("photon_trigger_scores.tsv", std::ios::out | std::ios::app);
        if (gScoresTSV.good()) {
            // Write a header only if file is empty
            gScoresTSV.seekp(0, std::ios::end);
            if (gScoresTSV.tellp() == std::streampos(0)) {
                gScoresTSV
                    << "event\tstation\tpmt\tprimary\tlabel\t"
                    << "score\tthreshold\tconf\t"
                    << "rise10_90\twidth\tpeak_amp\tcharge\tpeak_charge_ratio\t"
                    << "num_peaks\tsecond_peak_ratio\tkurtosis\tasymmetry\n";
                gScoresTSV.flush();
            }
            gScoresTSVOpen = true;
        }
    }
}

// ============================================================================
// Signal handler
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
// Neural Network (unchanged math; safer I/O)
// ============================================================================

PhotonTriggerML::NeuralNetwork::NeuralNetwork() :
    fInputSize(0), fHidden1Size(0), fHidden2Size(0),
    fTimeStep(0), fDropoutRate(0.1),
    fIsQuantized(false), fQuantizationScale(127.0)
{}

void PhotonTriggerML::NeuralNetwork::Initialize(int input_size, int h1, int h2)
{
    fInputSize = input_size; fHidden1Size = h1; fHidden2Size = h2;
    std::cout << "Initializing Physics-Based Neural Network: "
              << input_size << " -> " << h1 << " -> " << h2 << " -> 1\n";

    std::mt19937 gen(12345);
    std::uniform_real_distribution<> dist(-0.5, 0.5);

    fWeights1.assign(h1, std::vector<double>(input_size));
    for (int i=0;i<h1;++i) for (int j=0;j<input_size;++j)
        fWeights1[i][j] = dist(gen)/std::sqrt((double)input_size);

    fWeights2.assign(h2, std::vector<double>(h1));
    for (int i=0;i<h2;++i) for (int j=0;j<h1;++j)
        fWeights2[i][j] = dist(gen)/std::sqrt((double)h1);

    fWeights3.assign(1, std::vector<double>(h2));
    for (int j=0;j<h2;++j) fWeights3[0][j] = dist(gen)/std::sqrt((double)h2);

    fBias1.assign(h1, 0.0);
    fBias2.assign(h2, 0.0);
    fBias3 = 0.0;

    fMomentum1_w1.assign(h1, std::vector<double>(input_size,0.0));
    fMomentum2_w1.assign(h1, std::vector<double>(input_size,0.0));
    fMomentum1_w2.assign(h2, std::vector<double>(h1,0.0));
    fMomentum2_w2.assign(h2, std::vector<double>(h1,0.0));
    fMomentum1_w3.assign(1, std::vector<double>(h2,0.0));
    fMomentum2_w3.assign(1, std::vector<double>(h2,0.0));
    fMomentum1_b1.assign(h1, 0.0);
    fMomentum2_b1.assign(h1, 0.0);
    fMomentum1_b2.assign(h2, 0.0);
    fMomentum2_b2.assign(h2, 0.0);
    fMomentum1_b3 = 0.0; fMomentum2_b3 = 0.0;

    fTimeStep = 0;
    std::cout << "Neural Network initialized.\n";
}

double PhotonTriggerML::NeuralNetwork::Predict(const std::vector<double>& x, bool training)
{
    if ((int)x.size()!=fInputSize) return 0.5;

    std::vector<double> h1(fHidden1Size);
    for (int i=0;i<fHidden1Size;++i) {
        double s = fBias1[i];
        for (int j=0;j<fInputSize;++j) s += fWeights1[i][j]*x[j];
        h1[i] = 1.0/(1.0+std::exp(-s));
        if (training && (rand()/double(RAND_MAX))<fDropoutRate) h1[i]=0.0;
    }

    std::vector<double> h2(fHidden2Size);
    for (int i=0;i<fHidden2Size;++i) {
        double s = fBias2[i];
        for (int j=0;j<fHidden1Size;++j) s += fWeights2[i][j]*h1[j];
        h2[i] = 1.0/(1.0+std::exp(-s));
        if (training && (rand()/double(RAND_MAX))<fDropoutRate) h2[i]=0.0;
    }

    double o = fBias3;
    for (int j=0;j<fHidden2Size;++j) o += fWeights3[0][j]*h2[j];
    return 1.0/(1.0+std::exp(-o));
}

double PhotonTriggerML::NeuralNetwork::Train(const std::vector<std::vector<double>>& X,
                                             const std::vector<int>& Y,
                                             double lr)
{
    if (X.empty() || X.size()!=Y.size()) return -1.0;

    const int B = (int)X.size();
    int num_photons = std::count(Y.begin(),Y.end(),1);
    const double wP = (num_photons>0) ? 1.5 : 1.0; // lighter than before
    const double wH = 1.0;

    std::vector<std::vector<double>> gw1(fHidden1Size, std::vector<double>(fInputSize,0.0));
    std::vector<std::vector<double>> gw2(fHidden2Size, std::vector<double>(fHidden1Size,0.0));
    std::vector<std::vector<double>> gw3(1, std::vector<double>(fHidden2Size,0.0));
    std::vector<double> gb1(fHidden1Size,0.0), gb2(fHidden2Size,0.0);
    double gb3=0.0, total_loss=0.0;

    for (int n=0;n<B;++n) {
        const auto& x = X[n];
        const int y = Y[n];
        const double w = (y==1? wP : wH);

        // forward
        std::vector<double> h1(fHidden1Size), r1(fHidden1Size);
        for (int i=0;i<fHidden1Size;++i) {
            double s=fBias1[i];
            for (int j=0;j<fInputSize;++j) s+=fWeights1[i][j]*x[j];
            r1[i]=s; h1[i]=1.0/(1.0+std::exp(-s));
        }

        std::vector<double> h2(fHidden2Size), r2(fHidden2Size);
        for (int i=0;i<fHidden2Size;++i) {
            double s=fBias2[i];
            for (int j=0;j<fHidden1Size;++j) s+=fWeights2[i][j]*h1[j];
            r2[i]=s; h2[i]=1.0/(1.0+std::exp(-s));
        }

        double ro=fBias3;
        for (int j=0;j<fHidden2Size;++j) ro+=fWeights3[0][j]*h2[j];
        const double p = 1.0/(1.0+std::exp(-ro));

        total_loss += -w*( y*std::log(p+1e-7) + (1-y)*std::log(1-p+1e-7) );

        // backward
        const double go = w*(p - y);
        for (int j=0;j<fHidden2Size;++j) gw3[0][j] += go*h2[j];
        gb3 += go;

        std::vector<double> g2(fHidden2Size,0.0);
        for (int j=0;j<fHidden2Size;++j) {
            g2[j] = fWeights3[0][j]*go * (h2[j]*(1.0-h2[j]));
        }
        for (int i=0;i<fHidden2Size;++i) {
            for (int j=0;j<fHidden1Size;++j) gw2[i][j] += g2[i]*h1[j];
            gb2[i] += g2[i];
        }

        std::vector<double> g1(fHidden1Size,0.0);
        for (int j=0;j<fHidden1Size;++j) {
            for (int i=0;i<fHidden2Size;++i) g1[j]+=fWeights2[i][j]*g2[i];
            g1[j] *= (h1[j]*(1.0-h1[j]));
        }
        for (int i=0;i<fHidden1Size;++i) {
            for (int j=0;j<fInputSize;++j) gw1[i][j] += g1[i]*x[j];
            gb1[i] += g1[i];
        }
    }

    const double mom=0.9;
    for (int i=0;i<fHidden1Size;++i) {
        for (int j=0;j<fInputSize;++j) {
            const double g = gw1[i][j]/B;
            fMomentum1_w1[i][j] = mom*fMomentum1_w1[i][j] - lr*g;
            fWeights1[i][j] += fMomentum1_w1[i][j];
        }
        const double g = gb1[i]/B;
        fMomentum1_b1[i] = mom*fMomentum1_b1[i] - lr*g;
        fBias1[i] += fMomentum1_b1[i];
    }
    for (int i=0;i<fHidden2Size;++i) {
        for (int j=0;j<fHidden1Size;++j) {
            const double g = gw2[i][j]/B;
            fMomentum1_w2[i][j] = mom*fMomentum1_w2[i][j] - lr*g;
            fWeights2[i][j] += fMomentum1_w2[i][j];
        }
        const double g = gb2[i]/B;
        fMomentum1_b2[i] = mom*fMomentum1_b2[i] - lr*g;
        fBias2[i] += fMomentum1_b2[i];
    }
    for (int j=0;j<fHidden2Size;++j) {
        const double g = gw3[0][j]/B;
        fMomentum1_w3[0][j] = mom*fMomentum1_w3[0][j] - lr*g;
        fWeights3[0][j] += fMomentum1_w3[0][j];
    }
    const double g = gb3/B;
    fMomentum1_b3 = mom*fMomentum1_b3 - lr*g;
    fBias3 += fMomentum1_b3;

    return total_loss/B;
}

void PhotonTriggerML::NeuralNetwork::SaveWeights(const std::string& fn)
{
    std::ofstream file(fn.c_str());
    if (!file.is_open()) { std::cout<<"Error: could not save weights to "<<fn<<"\n"; return; }

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
    file << "\n" << fBias3 << "\n";
    file.close();
    std::cout << "Weights saved to " << fn << "\n";
}

bool PhotonTriggerML::NeuralNetwork::LoadWeights(const std::string& fn)
{
    std::ifstream file(fn.c_str());
    if (!file.is_open()) { std::cout<<"Warning: could not load weights from "<<fn<<"\n"; return false; }

    file >> fInputSize >> fHidden1Size >> fHidden2Size;

    fWeights1.assign(fHidden1Size, std::vector<double>(fInputSize));
    fWeights2.assign(fHidden2Size, std::vector<double>(fHidden1Size));
    fWeights3.assign(1, std::vector<double>(fHidden2Size));
    fBias1.assign(fHidden1Size, 0.0);
    fBias2.assign(fHidden2Size, 0.0);

    for (auto& row : fWeights1) for (double& w : row) file >> w;
    for (double& b : fBias1) file >> b;
    for (auto& row : fWeights2) for (double& w : row) file >> w;
    for (double& b : fBias2) file >> b;
    for (double& w : fWeights3[0]) file >> w;
    file >> fBias3;
    file.close();
    std::cout<<"Weights loaded from "<<fn<<"\n";
    return true;
}

void PhotonTriggerML::NeuralNetwork::QuantizeWeights() { fIsQuantized = true; }

// ============================================================================
// PhotonTriggerML
// ============================================================================

PhotonTriggerML::PhotonTriggerML() :
    fNeuralNetwork(std::make_unique<NeuralNetwork>()),
    fIsTraining(true), fTrainingEpochs(500), fTrainingStep(0),
    fBestValidationLoss(1e9), fEpochsSinceImprovement(0),
    fEventCount(0), fStationCount(0),
    fPhotonLikeCount(0), fHadronLikeCount(0),
    fEnergy(0), fCoreX(0), fCoreY(0), fPrimaryId(0),
    fPrimaryType("Unknown"),
    fPhotonScore(0), fConfidence(0), fDistance(0), fStationId(0),
    fIsActualPhoton(false),
    fOutputFile(nullptr), fMLTree(nullptr),
    fLogFileName("photon_trigger_ml_physics.log"),
    hPhotonScore(nullptr), hPhotonScorePhotons(nullptr), hPhotonScoreHadrons(nullptr),
    hConfidence(nullptr), hRisetime(nullptr), hAsymmetry(nullptr), hKurtosis(nullptr),
    hScoreVsEnergy(nullptr), hScoreVsDistance(nullptr),
    gROCCurve(nullptr), hConfusionMatrix(nullptr),
    hTrainingLoss(nullptr), hValidationLoss(nullptr), hAccuracyHistory(nullptr),
    fTruePositives(0), fFalsePositives(0), fTrueNegatives(0), fFalseNegatives(0),
    fPhotonThreshold(0.65), // will be auto-calibrated
    fEnergyMin(1e18), fEnergyMax(1e19),
    fOutputFileName("photon_trigger_ml_physics.root"),
    fWeightsFileName("photon_trigger_weights_physics.txt"),
    fLoadPretrainedWeights(true)
{
    fInstance = this;
    fFeatureMeans.assign(17,0.0);
    fFeatureStdDevs.assign(17,1.0);

    std::cout<<"\n==========================================\n"
             <<"PhotonTriggerML Constructor (precision-first)\n"
             <<"Output file: "<<fOutputFileName<<"\n"
             <<"Log file: "<<fLogFileName<<"\n"
             <<"==========================================\n";
}

PhotonTriggerML::~PhotonTriggerML() {}

VModule::ResultFlag PhotonTriggerML::Init()
{
    INFO("PhotonTriggerML::Init()");
    // log file
    fLogFile.open(fLogFileName.c_str(), std::ios::out);
    if (!fLogFile.is_open()) { ERROR("Failed to open log file"); return eFailure; }

    time_t now = time(0);
    fLogFile << "==========================================" << endl;
    fLogFile << "PhotonTriggerML Physics-Based Version Log" << endl;
    fLogFile << "Date: " << ctime(&now);
    fLogFile << "==========================================" << endl << endl;
    fLogFile.flush();

    // NN
    fNeuralNetwork->Initialize(17,8,4);
    if (fLoadPretrainedWeights && fNeuralNetwork->LoadWeights(fWeightsFileName)) {
        std::cout<<"Loaded pre-trained weights.\n";
        // keep fIsTraining as configured; calibration will still run even if we don't train
    } else {
        std::cout<<"Starting with random weights (training mode)\n";
        fIsTraining = true;
    }

    // ROOT out
    fOutputFile = new TFile(fOutputFileName.c_str(),"RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) { ERROR("Failed to create ROOT output"); return eFailure; }

    fMLTree = new TTree("MLTree","PhotonTriggerML Physics Tree");
    fMLTree->Branch("eventId",&fEventCount,"eventId/I");
    fMLTree->Branch("stationId",&fStationId,"stationId/I");
    fMLTree->Branch("energy",&fEnergy,"energy/D");
    fMLTree->Branch("distance",&fDistance,"distance/D");
    fMLTree->Branch("photonScore",&fPhotonScore,"photonScore/D");
    fMLTree->Branch("confidence",&fConfidence,"confidence/D");
    fMLTree->Branch("primaryId",&fPrimaryId,"primaryId/I");
    fMLTree->Branch("primaryType",&fPrimaryType);
    fMLTree->Branch("isActualPhoton",&fIsActualPhoton,"isActualPhoton/O");

    hPhotonScore = new TH1D("hPhotonScore","ML Photon Score (All);Score;Count",50,0,1);
    hPhotonScorePhotons = new TH1D("hPhotonScorePhotons","ML Score (True Photons);Score;Count",50,0,1);
    hPhotonScorePhotons->SetLineColor(kBlue);
    hPhotonScoreHadrons = new TH1D("hPhotonScoreHadrons","ML Score (True Hadrons);Score;Count",50,0,1);
    hPhotonScoreHadrons->SetLineColor(kRed);
    hConfidence = new TH1D("hConfidence","ML Confidence;|Score-0.5|;Count",50,0,0.5);
    hRisetime = new TH1D("hRisetime","Rise Time 10-90%;Time [ns];Count",50,0,1000);
    hAsymmetry = new TH1D("hAsymmetry","Pulse Asymmetry;(fall-rise)/(fall+rise);Count",50,-1,1);
    hKurtosis = new TH1D("hKurtosis","Signal Kurtosis;Kurtosis;Count",50,-5,20);

    hScoreVsEnergy = new TH2D("hScoreVsEnergy","Score vs Energy;Energy [eV];Score",50,1e17,1e20,50,0,1);
    hScoreVsDistance = new TH2D("hScoreVsDistance","Score vs Distance;Distance [m];Score",50,0,3000,50,0,1);

    hConfusionMatrix = new TH2D("hConfusionMatrix","Confusion Matrix;Predicted;Actual",2,-0.5,1.5,2,-0.5,1.5);
    hConfusionMatrix->GetXaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetXaxis()->SetBinLabel(2,"Photon");
    hConfusionMatrix->GetYaxis()->SetBinLabel(1,"Hadron");
    hConfusionMatrix->GetYaxis()->SetBinLabel(2,"Photon");

    hTrainingLoss = new TH1D("hTrainingLoss","Training Loss;Batch;Loss",10000,0,10000);
    hValidationLoss = new TH1D("hValidationLoss","Validation Loss;Batch;Loss",10000,0,10000);
    hAccuracyHistory = new TH1D("hAccuracyHistory","Accuracy History;Batch;Accuracy [%]",10000,0,10000);

    signal(SIGINT,  PhotonTriggerMLSignalHandler);
    signal(SIGTSTP, PhotonTriggerMLSignalHandler);

    EnsureScoresTSV();

    std::cout<<"Initialization complete!\n";
    INFO("PhotonTriggerML initialized.");
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

    // shower info
    fEnergy=0; fCoreX=0; fCoreY=0; fPrimaryId=0; fPrimaryType="Unknown"; fIsActualPhoton=false;
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        fPrimaryId = shower.GetPrimaryParticle();
        switch (fPrimaryId) {
            case 22: fPrimaryType="photon"; fIsActualPhoton=true; break;
            case 11: case -11: fPrimaryType="electron"; break;
            case 2212: fPrimaryType="proton"; break;
            case 1000026056: fPrimaryType="iron"; break;
            default: fPrimaryType = (fPrimaryId>1000000000) ? "nucleus" : "unknown";
        }
        fParticleTypeCounts[fPrimaryType]++;
        if (shower.GetNSimCores()>0) {
            const Detector& det = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = det.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            fCoreX = core.GetX(siteCS); fCoreY = core.GetY(siteCS);
        }
        if (fEventCount<=5) {
            std::cout<<"\nEvent "<<fEventCount<<": Energy="<<fEnergy/1e18<<" EeV, Primary="<<fPrimaryType
                     <<" (ID="<<fPrimaryId<<")\n";
        }
    }

    // stations
    if (event.HasSEvent()) {
        const sevt::SEvent& sevent = event.GetSEvent();
        for (sevt::SEvent::ConstStationIterator it = sevent.StationsBegin();
             it != sevent.StationsEnd(); ++it) {
            ProcessStation(*it);
        }
    }

    // metrics
    if (fEventCount % 10 == 0) {
        CalculateAndDisplayMetrics();
        hConfusionMatrix->SetBinContent(1,1,fTrueNegatives);
        hConfusionMatrix->SetBinContent(2,1,fFalsePositives);
        hConfusionMatrix->SetBinContent(1,2,fFalseNegatives);
        hConfusionMatrix->SetBinContent(2,2,fTruePositives);
    }

    // --- Always run calibration against validation cache ---
    if ( (fEventCount % 5)==0 && (!fValidationFeatures.empty()) ) {
        double val_loss = TrainNetwork(); // will only train if training data exists; always calibrates
        int batch_num = fEventCount/5;
        hValidationLoss->SetBinContent(batch_num, val_loss);

        int correct = fTruePositives + fTrueNegatives;
        int total   = fTruePositives + fTrueNegatives + fFalsePositives + fFalseNegatives;
        if (total>0) {
            double acc = 100.0*correct/total;
            hAccuracyHistory->SetBinContent(batch_num, acc);
        }
    }

    return eSuccess;
}

void PhotonTriggerML::ProcessStation(const sevt::Station& station)
{
    fStationId = station.GetId();

    // geometry
    try {
        const Detector& detector = Detector::GetInstance();
        const sdet::SDetector& sdetector = detector.GetSDetector();
        const sdet::Station& detStation = sdetector.GetStation(fStationId);
        const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
        double sx = detStation.GetPosition().GetX(siteCS);
        double sy = detStation.GetPosition().GetY(siteCS);
        fDistance = std::sqrt((sx-fCoreX)*(sx-fCoreX) + (sy-fCoreY)*(sy-fCoreY));
    } catch (...) {
        fDistance = -1;
        return;
    }

    const int firstPMT = sdet::Station::GetFirstPMTId();

    for (int p=0;p<3;++p) {
        const int pmtId = firstPMT + p;
        if (!station.HasPMT(pmtId)) continue;
        const sevt::PMT& pmt = station.GetPMT(pmtId);

        std::vector<double> trace;
        bool ok = ExtractTraceData(pmt, trace);
        if (!ok || trace.size()!=2048) continue;

        double maxV=*std::max_element(trace.begin(),trace.end());
        double minV=*std::min_element(trace.begin(),trace.end());
        if (maxV-minV<10.0) continue;

        fFeatures = ExtractEnhancedFeatures(trace);

        // physics-only tag (no bias)
        bool physicsPhotonLike = false;
        if (fFeatures.risetime_10_90<150.0 &&
            fFeatures.pulse_width<300.0 &&
            fFeatures.secondary_peak_ratio<0.30 &&
            fFeatures.peak_charge_ratio>0.15) physicsPhotonLike = true;

        hRisetime->Fill(fFeatures.risetime_10_90);
        hAsymmetry->Fill(fFeatures.asymmetry);
        hKurtosis->Fill(fFeatures.kurtosis);
        UpdateFeatureStatistics(fFeatures);

        std::vector<double> x = NormalizeFeatures(fFeatures);
        // emphasize physics
        if (x.size()>=17) {
            x[1]*=2.0; x[3]*=2.0; x[7]*=1.5; x[16]*=1.5;
        }

        double score = fNeuralNetwork->Predict(x,false);

        // muon-spike penalties
        double penalty = 0.0;
        if (fFeatures.pulse_width<120.0 && fFeatures.peak_charge_ratio>0.35) penalty -= 0.20;
        if (fFeatures.num_peaks<=2 && fFeatures.secondary_peak_ratio>0.60)  penalty -= 0.10;

        fPhotonScore = std::max(0.0, std::min(1.0, score + penalty));
        fConfidence  = std::fabs(fPhotonScore - 0.5);

        // validation cache (always collect so threshold can adapt, even if not training)
        fValidationFeatures.push_back(x);
        fValidationLabels.push_back(fIsActualPhoton ? 1 : 0);

        // training cache (reduced oversampling)
        if (fIsTraining) {
            if (fIsActualPhoton) {
                for (int c=0;c<2;++c) {
                    std::vector<double> v = x;
                    static std::mt19937 gen(1234567u);
                    std::normal_distribution<> n(0,0.02);
                    for (double& t : v) t += n(gen);
                    fTrainingFeatures.push_back(v);
                    fTrainingLabels.push_back(1);
                }
            } else {
                fTrainingFeatures.push_back(x);
                fTrainingLabels.push_back(0);
            }
        }

        ++fStationCount;
        const double thr = fPhotonThreshold;
        const bool isPhoton = (fPhotonScore>thr);

        // record result
        MLResult r;
        r.photonScore=fPhotonScore; r.identifiedAsPhoton=isPhoton; r.isActualPhoton=fIsActualPhoton;
        r.vemCharge=fFeatures.total_charge; r.features=fFeatures; r.primaryType=fPrimaryType; r.confidence=fConfidence;
        fMLResultsMap[fStationId]=r;

        if (isPhoton) { ++fPhotonLikeCount; if (fIsActualPhoton) ++fTruePositives; else ++fFalsePositives; }
        else          { ++fHadronLikeCount; if (!fIsActualPhoton) ++fTrueNegatives; else ++fFalseNegatives; }

        hPhotonScore->Fill(fPhotonScore);
        hConfidence->Fill(fConfidence);
        if (fIsActualPhoton) hPhotonScorePhotons->Fill(fPhotonScore);
        else                 hPhotonScoreHadrons->Fill(fPhotonScore);
        hScoreVsEnergy->Fill(fEnergy, fPhotonScore);
        hScoreVsDistance->Fill(fDistance, fPhotonScore);
        fMLTree->Fill();

        // append TSV
        EnsureScoresTSV();
        if (gScoresTSVOpen) {
            gScoresTSV << fEventCount << "\t" << fStationId << "\t" << pmtId << "\t"
                       << fPrimaryType << "\t" << (fIsActualPhoton?1:0) << "\t"
                       << std::fixed << std::setprecision(6) << fPhotonScore << "\t"
                       << fPhotonThreshold << "\t" << fConfidence << "\t"
                       << fFeatures.risetime_10_90 << "\t" << fFeatures.pulse_width << "\t"
                       << fFeatures.peak_amplitude << "\t" << fFeatures.total_charge << "\t"
                       << fFeatures.peak_charge_ratio << "\t"
                       << fFeatures.num_peaks << "\t" << fFeatures.secondary_peak_ratio << "\t"
                       << fFeatures.kurtosis << "\t" << fFeatures.asymmetry << "\n";
            gScoresTSV.flush();
        }

        if ((fIsActualPhoton || physicsPhotonLike) && fStationCount<=100) {
            std::cout<<"  Station "<<fStationId<<" PMT "<<pmtId
                     <<": Score="<<std::fixed<<std::setprecision(3)<<fPhotonScore
                     <<" (True: "<<fPrimaryType<<")"
                     <<" Rise="<<fFeatures.risetime_10_90<<"ns"
                     <<" Width="<<fFeatures.pulse_width<<"ns"
                     <<" PhysicsTag="<<(physicsPhotonLike?"YES":"NO")<<"\n";
        }
    }
}

bool PhotonTriggerML::ExtractTraceData(const sevt::PMT& pmt, std::vector<double>& out)
{
    out.clear();

    // Preferred: recorded FADC (HG)
    if (pmt.HasFADCTrace()) {
        try {
            const auto& tr = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
            for (int i=0;i<2048;++i) out.push_back(tr[i]);
            return true;
        } catch (...) {}
    }

    // Fallback: simulation data
    if (pmt.HasSimData()) {
        try {
            const sevt::PMTSimData& sim = pmt.GetSimData();
            if (sim.HasFADCTrace(sevt::StationConstants::eTotal)) {
                const auto& tr = sim.GetFADCTrace(sdet::PMTConstants::eHighGain,
                                                  sevt::StationConstants::eTotal);
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
    EnhancedFeatures F;
    const int N = (int)trace.size();
    const double ADC_PER_VEM = 180.0;
    const double NS_PER_BIN = 25.0;

    int peak_bin=0; double peak=0.0; double total=0.0;
    std::vector<double> sig(N,0.0);

    double base=0.0;
    for (int i=0;i<100 && i<N;++i) base += trace[i];
    base /= 100.0;

    for (int i=0;i<N;++i) {
        sig[i] = trace[i]-base;
        if (sig[i]<0) sig[i]=0.0;
        if (sig[i]>peak) { peak=sig[i]; peak_bin=i; }
        total += sig[i];
    }
    if (peak<5.0 || total<10.0) return F;

    F.peak_amplitude = peak / ADC_PER_VEM;
    F.total_charge   = total / ADC_PER_VEM;
    F.peak_charge_ratio = F.peak_amplitude / (F.total_charge + 1e-3);

    const double p10 = 0.10*peak;
    const double p50 = 0.50*peak;
    const double p90 = 0.90*peak;

    int r10=0, r50=0, r90=peak_bin;
    for (int i=peak_bin;i>=0;--i) {
        if (sig[i]<=p90 && r90==peak_bin) r90=i;
        if (sig[i]<=p50 && r50==0)        r50=i;
        if (sig[i]<=p10) { r10=i; break; }
    }
    int f90=peak_bin, f10=N-1;
    for (int i=peak_bin;i<N;++i) {
        if (sig[i]<=p90 && f90==peak_bin) f90=i;
        if (sig[i]<=p10) { f10=i; break; }
    }

    F.risetime_10_50 = std::abs(r50-r10)*NS_PER_BIN;
    F.risetime_10_90 = std::abs(r90-r10)*NS_PER_BIN;
    F.falltime_90_10 = std::abs(f10-f90)*NS_PER_BIN;

    // FWHM
    const double half = 0.5*peak;
    int hr=r10, hf=f10;
    for (int i=r10;i<=peak_bin;++i) if (sig[i]>=half) { hr=i; break; }
    for (int i=peak_bin;i<N;++i)    if (sig[i]<=half) { hf=i; break; }
    F.pulse_width = std::abs(hf-hr)*NS_PER_BIN;

    const double rise = F.risetime_10_90;
    const double fall = F.falltime_90_10;
    F.asymmetry = (fall-rise)/(fall+rise+1e-3);

    double mean_t=0.0;
    for (int i=0;i<N;++i) mean_t += i*sig[i];
    mean_t /= (total+1e-3);

    double var=0.0, sk=0.0, ku=0.0;
    for (int i=0;i<N;++i) {
        double d = i-mean_t;
        double w = sig[i]/(total+1e-3);
        var += d*d*w; sk += d*d*d*w; ku += d*d*d*d*w;
    }
    const double sd = std::sqrt(var+1e-3);
    F.time_spread = sd*NS_PER_BIN;
    F.skewness = sk/(sd*sd*sd+1e-3);
    F.kurtosis = ku/(var*var+1e-3) - 3.0;

    // early/late
    const int Q = N/4;
    double early=0.0, late=0.0;
    for (int i=0;i<Q;++i) early += sig[i];
    for (int i=3*Q;i<N;++i) late += sig[i];
    F.early_fraction = early/(total+1e-3);
    F.late_fraction  = late /(total+1e-3);

    // smoothness / HF content
    double ssd=0.0; int cnt=0;
    for (int i=1;i<N-1;++i) if (sig[i]>0.1*peak) {
        double sec = sig[i+1]-2*sig[i]+sig[i-1];
        ssd += sec*sec; ++cnt;
    }
    F.smoothness = std::sqrt( ssd / (cnt+1) );

    double HF=0.0;
    for (int i=1;i<N-1;++i) {
        double d = sig[i+1]-sig[i-1];
        HF += d*d;
    }
    F.high_freq_content = HF/((total*total)+1e-3);

    // peaks (stricter threshold to reduce fake multi-peak)
    F.num_peaks = 0;
    double sec_peak=0.0;
    const double th = 0.25*peak;
    for (int i=1;i<N-1;++i) {
        if (sig[i]>th && sig[i]>sig[i-1] && sig[i]>sig[i+1]) {
            ++F.num_peaks;
            if (i!=peak_bin && sig[i]>sec_peak) sec_peak=sig[i];
        }
    }
    F.secondary_peak_ratio = sec_peak/(peak+1e-3);

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

    static long n=0; ++n;

    for (size_t i=0;i<raw.size();++i) {
        double delta = raw[i]-fFeatureMeans[i];
        fFeatureMeans[i] += delta/n;
        double delta2 = raw[i]-fFeatureMeans[i];
        fFeatureStdDevs[i] += delta*delta2;
    }
    if (n>1) {
        for (size_t i=0;i<raw.size();++i) {
            fFeatureStdDevs[i] = std::sqrt( fFeatureStdDevs[i]/(n-1) );
            if (fFeatureStdDevs[i]<1e-3) fFeatureStdDevs[i]=1.0;
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

    std::vector<double> mins = {0,0,0,0,-1,0,0,0,0,-5,-5,0,0,0,0,0,0};
    std::vector<double> maxs = {500,1000,1000,1000,1,10,100,1,100,20,5,1,1,1000,10,10,1};
    std::vector<double> z; z.reserve(raw.size());

    for (size_t i=0;i<raw.size();++i) {
        double v = (raw[i]-mins[i])/(maxs[i]-mins[i]+1e-3);
        if (v<0) v=0;
        if (v>1) v=1;
        z.push_back(v);
    }
    return z;
}

// TrainNetwork also performs validation-only threshold calibration when there is no training data
double PhotonTriggerML::TrainNetwork()
{
    // training (only if we have data)
    double train_loss = 0.0;
    if (!fTrainingFeatures.empty()) {
        std::vector<std::vector<double>> Bx;
        std::vector<int> By;
        const int M = std::min(100, (int)fTrainingFeatures.size());
        Bx.reserve(M); By.reserve(M);
        for (int i=0;i<M;++i) { Bx.push_back(fTrainingFeatures[i]); By.push_back(fTrainingLabels[i]); }

        int numP = std::count(By.begin(),By.end(),1);
        int numH = (int)By.size()-numP;
        std::cout<<"\n  Training with "<<Bx.size()<<" samples... (P:"<<numP<<" H:"<<numH<<")";

        const double lr=0.01;
        double total=0.0;
        for (int e=0;e<10;++e) total += fNeuralNetwork->Train(Bx,By,lr);
        train_loss = total/10.0;
        std::cout<<" Loss: "<<std::fixed<<std::setprecision(4)<<train_loss;

        int batch_num = fEventCount/5;
        hTrainingLoss->SetBinContent(batch_num, train_loss);

        // keep training cache bounded
        if (fTrainingFeatures.size()>1000) {
            fTrainingFeatures.erase(fTrainingFeatures.begin(),fTrainingFeatures.begin()+200);
            fTrainingLabels.erase(fTrainingLabels.begin(),fTrainingLabels.begin()+200);
        }
        fTrainingStep++;
    }

    // validation / threshold calibration (always if we have validation)
    double val_loss = 0.0;
    if (!fValidationFeatures.empty()) {
        const int N=(int)fValidationFeatures.size();
        std::vector<double> preds(N);
        for (int i=0;i<N;++i) preds[i] = fNeuralNetwork->Predict(fValidationFeatures[i],false);

        int correct=0;
        for (int i=0;i<N;++i) {
            const int y=fValidationLabels[i];
            const double p=preds[i];
            val_loss -= y*std::log(p+1e-7) + (1-y)*std::log(1-p+1e-7);
            const int yhat = (p>fPhotonThreshold)?1:0;
            if (yhat==y) ++correct;
        }
        val_loss/=N;
        const double val_acc = 100.0*correct/N;
        std::cout<<" Val: "<<val_loss<<" (Acc: "<<val_acc<<"%)";

        // sweep thresholds and select best-F1
        double bestF1=-1.0, bestThr=fPhotonThreshold;
        std::ofstream sweep("photon_trigger_threshold_sweep.txt", std::ios::out);
        for (double thr=0.45; thr<=0.75+1e-9; thr+=0.02) {
            int TP=0,FP=0,TN=0,FN=0;
            for (int i=0;i<N;++i) {
                int y=fValidationLabels[i];
                int z=(preds[i]>thr)?1:0;
                if (z==1 && y==1) ++TP;
                else if (z==1 && y==0) ++FP;
                else if (z==0 && y==0) ++TN;
                else ++FN;
            }
            double prec = (TP+FP>0)? (double)TP/(TP+FP) : 0.0;
            double rec  = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            double F1   = (prec+rec>0)? 2.0*prec*rec/(prec+rec) : 0.0;
            if (sweep.good()) sweep<<std::fixed<<std::setprecision(3)
                                  <<"thr="<<thr<<" F1="<<F1<<" P="<<prec<<" R="<<rec<<"\n";
            if (F1>bestF1) { bestF1=F1; bestThr=thr; }
        }
        if (sweep.good()) sweep.close();

        const double oldThr=fPhotonThreshold;
        // move 20% toward best
        fPhotonThreshold = 0.8*fPhotonThreshold + 0.2*bestThr;

        // collapse safeties: if no positive predictions at all recently, reset to median
        int recent = std::min(N, 400);
        int pos=0; for (int i=N-recent;i<N;++i) if (preds[i]>fPhotonThreshold) ++pos;
        if (pos==0 || pos==recent) {
            std::vector<double> tmp(preds.end()-recent, preds.end());
            std::nth_element(tmp.begin(), tmp.begin()+tmp.size()/2, tmp.end());
            double median = tmp[tmp.size()/2];
            // pull toward median
            fPhotonThreshold = 0.5*fPhotonThreshold + 0.5*std::min(0.80, std::max(0.40, median));
        }

        std::cout<<" | Recalibrated threshold -> "<<std::fixed<<std::setprecision(3)<<fPhotonThreshold
                 <<" (best="<<bestThr<<" F1="<<bestF1<<")\n";

        if (fLogFile.is_open()) {
            fLogFile<<"Recalibrated threshold: "<<oldThr<<" -> "<<fPhotonThreshold
                    <<" using bestF1="<<bestF1<<" at thr="<<bestThr<<"\n";
            fLogFile.flush();
        }

        // simple ROC from sweep file (optional; minimal)
        if (!gROCCurve) gROCCurve = new TGraph();
        gROCCurve->SetName("ROC");
        gROCCurve->SetTitle("ROC;False Positive Rate;True Positive Rate");
        gROCCurve->Set(0);
        for (double thr=0.45; thr<=0.75+1e-9; thr+=0.02) {
            int TP=0,FP=0,TN=0,FN=0;
            for (int i=0;i<N;++i) {
                int y=fValidationLabels[i];
                int z=(preds[i]>thr)?1:0;
                if (z==1 && y==1) ++TP;
                else if (z==1 && y==0) ++FP;
                else if (z==0 && y==0) ++TN;
                else ++FN;
            }
            double FPR = (FP+TN>0)? (double)FP/(FP+TN) : 0.0;
            double TPR = (TP+FN>0)? (double)TP/(TP+FN) : 0.0;
            gROCCurve->SetPoint(gROCCurve->GetN(), FPR, TPR);
        }
    }

    return val_loss;
}

void PhotonTriggerML::CalculateAndDisplayMetrics()
{
    int total = fTruePositives+fTrueNegatives+fFalsePositives+fFalseNegatives;
    if (total==0) return;

    double accuracy  = 100.0*(fTruePositives+fTrueNegatives)/total;
    double precision = (fTruePositives+fFalsePositives>0)?
                        100.0*fTruePositives/(fTruePositives+fFalsePositives) : 0.0;
    double recall    = (fTruePositives+fFalseNegatives>0)?
                        100.0*fTruePositives/(fTruePositives+fFalseNegatives) : 0.0;
    double f1        = (precision+recall>0)? 2*precision*recall/(precision+recall) : 0.0;

    double photon_frac = (fPhotonLikeCount+fHadronLikeCount>0)?
                          100.0*fPhotonLikeCount/(fPhotonLikeCount+fHadronLikeCount) : 0.0;

    std::cout << "│ " << std::setw(6) << fEventCount
              << " │ " << std::setw(8) << fStationCount
              << " │ " << std::fixed << std::setprecision(1) << std::setw(7) << photon_frac << "%"
              << " │ " << std::setw(7) << accuracy << "%"
              << " │ " << std::setw(8) << precision << "%"
              << " │ " << std::setw(7) << recall << "%"
              << " │ " << std::setw(7) << f1 << "%│\n";

    if (fLogFile.is_open()) {
        fLogFile << "Event " << fEventCount
                 << " - Acc: " << accuracy
                 << "% Prec: " << precision
                 << "% Rec: " << recall
                 << "% F1: "  << f1 << "%\n";
        fLogFile.flush();
    }
}

void PhotonTriggerML::CalculatePerformanceMetrics()
{
    std::cout<<"\n==========================================\nPERFORMANCE METRICS\n==========================================\n";

    int total = fTruePositives+fTrueNegatives+fFalsePositives+fFalseNegatives;
    if (total==0) { std::cout<<"No predictions made yet!\n"; return; }

    double accuracy  = 100.0*(fTruePositives+fTrueNegatives)/total;
    double precision = (fTruePositives+fFalsePositives>0)?
                        100.0*fTruePositives/(fTruePositives+fFalsePositives) : 0.0;
    double recall    = (fTruePositives+fFalseNegatives>0)?
                        100.0*fTruePositives/(fTruePositives+fFalseNegatives) : 0.0;
    double f1        = (precision+recall>0)? 2*precision*recall/(precision+recall) : 0.0;

    std::cout<<"Accuracy:  "<<std::fixed<<std::setprecision(1)<<accuracy<<"%\n";
    std::cout<<"Precision: "<<precision<<"%\n";
    std::cout<<"Recall:    "<<recall<<"%\n";
    std::cout<<"F1-Score:  "<<f1<<"%\n\n";

    std::cout<<"CONFUSION MATRIX:\n";
    std::cout<<"                Predicted\n";
    std::cout<<"             Hadron   Photon\n";
    std::cout<<"Actual Hadron  "<<std::setw(6)<<fTrueNegatives<<"   "<<std::setw(6)<<fFalsePositives<<"\n";
    std::cout<<"       Photon  "<<std::setw(6)<<fFalseNegatives<<"   "<<std::setw(6)<<fTruePositives<<"\n\n";

    std::cout<<"Total Stations: "<<fStationCount<<"\n";
    std::cout<<"Photon-like: "<<fPhotonLikeCount<<" ("
             << (100.0*fPhotonLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount)) <<"%)\n";
    std::cout<<"Hadron-like: "<<fHadronLikeCount<<" ("
             << (100.0*fHadronLikeCount/std::max(1, fPhotonLikeCount+fHadronLikeCount)) <<"%)\n";

    std::cout<<"\nPARTICLE TYPE BREAKDOWN:\n";
    for (const auto& pr : fParticleTypeCounts)
        std::cout<<"  "<<pr.first<<": "<<pr.second<<" events\n";
    std::cout<<"==========================================\n";

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
    std::cout<<"\n==========================================\n"
             <<"PHOTONTRIGGERML FINAL SUMMARY (PHYSICS)\n"
             <<"==========================================\n";
    std::cout<<"Events processed: "<<fEventCount<<"\n";
    std::cout<<"Stations analyzed: "<<fStationCount<<"\n";

    CalculatePerformanceMetrics();

    // save weights
    fNeuralNetwork->SaveWeights(fWeightsFileName);

    // ROOT outputs
    if (fOutputFile) {
        fOutputFile->cd();
        if (fMLTree) { fMLTree->Write(); std::cout<<"Wrote "<<fMLTree->GetEntries()<<" entries to tree\n"; }
        if (gROCCurve) gROCCurve->Write("ROC");
        hPhotonScore->Write(); hPhotonScorePhotons->Write(); hPhotonScoreHadrons->Write();
        hConfidence->Write(); hRisetime->Write(); hAsymmetry->Write(); hKurtosis->Write();
        hScoreVsEnergy->Write(); hScoreVsDistance->Write();
        hConfusionMatrix->Write(); hTrainingLoss->Write(); hValidationLoss->Write(); hAccuracyHistory->Write();
        fOutputFile->Close(); delete fOutputFile; fOutputFile=nullptr;
        std::cout<<"Histograms written to "<<fOutputFileName<<"\n";
    }

    // compact ROOT + text summary
    {
        TFile fsum("photon_trigger_summary.root","RECREATE");
        if (!fsum.IsZombie()) {
            if (hConfusionMatrix) hConfusionMatrix->Write("ConfusionMatrix");
            if (hPhotonScore) hPhotonScore->Write("PhotonScore_All");
            if (hPhotonScorePhotons) hPhotonScorePhotons->Write("PhotonScore_TruePhotons");
            if (hPhotonScoreHadrons) hPhotonScoreHadrons->Write("PhotonScore_Hadrons");
            if (hConfidence) hConfidence->Write("Confidence");
            if (gROCCurve) gROCCurve->Write("ROC");
            fsum.Close();
        }
    }
    {
        // Original text format (restored)
        std::ofstream txt("photon_trigger_summary.txt");
        if (txt.is_open()) {
            const int TP=fTruePositives, FP=fFalsePositives, TN=fTrueNegatives, FN=fFalseNegatives;
            const int total=TP+FP+TN+FN;
            const double acc  = (total>0)? 100.0*(TP+TN)/total : 0.0;
            const double prec = (TP+FP>0)? 100.0*TP/(TP+FP)   : 0.0;
            const double rec  = (TP+FN>0)? 100.0*TP/(TP+FN)   : 0.0;
            const double f1   = (prec+rec>0)? (2.0*prec*rec/(prec+rec)) : 0.0;
            txt<<"PhotonTriggerML Summary\n";
            txt<<"BaseThreshold="<<fPhotonThreshold<<"\n";
            txt<<"TP="<<TP<<" FP="<<FP<<" TN="<<TN<<" FN="<<FN<<"\n";
            txt<<std::fixed<<std::setprecision(2)
               <<"Accuracy="<<acc<<"%  Precision="<<prec<<"%  Recall="<<rec<<"%  F1="<<f1<<"%\n";
            txt.close();
        }
    }

    // optional annotation of existing trace file
    if (!gCalledFromSignal) {
        auto strip_ml = [](const std::string& t)->std::string {
            size_t p=t.find(" [ML:"); return (p==std::string::npos)? t : t.substr(0,p);
        };
        auto score_from_hist = [this](TH1* h, double& score, bool& isPhoton)->bool {
            if (!h) return false;
            const int n = h->GetNbinsX();
            if (n<=0) return false;
            std::vector<double> tr(n);
            for (int i=1;i<=n;++i) tr[i-1]=h->GetBinContent(i);
            EnhancedFeatures F = ExtractEnhancedFeatures(tr);
            std::vector<double> X = NormalizeFeatures(F);
            if (X.size()>=17) { X[1]*=2.0; X[3]*=2.0; X[7]*=1.5; X[16]*=1.5; }
            double ml = fNeuralNetwork->Predict(X,false);
            double pen=0.0;
            if (F.pulse_width<120.0 && F.peak_charge_ratio>0.35) pen-=0.20;
            if (F.num_peaks<=2 && F.secondary_peak_ratio>0.60)  pen-=0.10;
            score = std::max(0.0,std::min(1.0,ml+pen));
            isPhoton = (score>fPhotonThreshold);
            return true;
        };
        auto annotate_dir = [&](TDirectory* dir, auto&& self)->void {
            if (!dir) return;
            TIter next(dir->GetListOfKeys()); TKey* key=nullptr;
            while ((key=(TKey*)next())) {
                TObject* obj = key->ReadObj();
                if (!obj) continue;
                if (obj->InheritsFrom(TDirectory::Class())) {
                    self((TDirectory*)obj, self);
                } else if (obj->InheritsFrom(TH1::Class())) {
                    TH1* h = (TH1*)obj; double sc=0.0; bool ph=false;
                    if (score_from_hist(h,sc,ph)) {
                        const std::string base = strip_ml(h->GetTitle()?h->GetTitle():"");
                        std::ostringstream t; t<<base<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc
                                               <<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                        h->SetTitle(t.str().c_str());
                        dir->cd(); h->Write(h->GetName(), TObject::kOverwrite);
                    }
                } else if (obj->InheritsFrom(TCanvas::Class())) {
                    TCanvas* c=(TCanvas*)obj; TH1* h=nullptr;
                    if (TList* prim=c->GetListOfPrimitives()) {
                        TIter nx(prim); TObject* po=nullptr;
                        while ((po=nx())) if (po->InheritsFrom(TH1::Class())) { h=(TH1*)po; break; }
                    }
                    if (h) {
                        double sc=0.0; bool ph=false;
                        if (score_from_hist(h,sc,ph)) {
                            const std::string baseC = strip_ml(c->GetTitle()?c->GetTitle():"");
                            const std::string baseH = strip_ml(h->GetTitle()?h->GetTitle():"");
                            std::ostringstream tc,th;
                            tc<<baseC<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc
                                     <<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                            th<<baseH<<" [ML: "<<std::fixed<<std::setprecision(3)<<sc
                                     <<" "<<(ph?"ML-Photon":"ML-Hadron")<<"]";
                            c->SetTitle(tc.str().c_str());
                            h->SetTitle(th.str().c_str());
                            c->Modified(); c->Update();
                            dir->cd(); c->Write(c->GetName(), TObject::kOverwrite);
                        }
                    }
                }
            }
        };

        const char* candidates[] = {"pmt_traces_1EeV.root","pmt_Traces_1EeV.root",
                                    "pmt_traces_1eev.root","pmt_Traces_1eev.root"};
        TFile* f=nullptr;
        for (const char* nm : candidates) {
            f=TFile::Open(nm,"UPDATE");
            if (f && !f->IsZombie()) break;
            if (f) { delete f; f=nullptr; }
        }
        if (f && !f->IsZombie()) {
            if (!f->TestBit(TFile::kRecovered)) {
                annotate_dir(f, annotate_dir);
                f->Write("", TObject::kOverwrite);
                std::cout<<"Annotated ML tags in existing trace file: "<<f->GetName()<<"\n";
            } else {
                std::cout<<"Trace file appears recovered; skipping ML annotation for safety.\n";
            }
            f->Close(); delete f;
        } else {
            std::cout<<"Trace file not found for ML annotation (skipping).\n";
        }
    }

    if (fLogFile.is_open()) { fLogFile.close(); }
    if (gScoresTSVOpen) { gScoresTSV.close(); gScoresTSVOpen=false; }

    std::cout<<"==========================================\n";
}

VModule::ResultFlag PhotonTriggerML::Finish()
{
    INFO("PhotonTriggerML::Finish()");
    SaveAndDisplaySummary();
    return eSuccess;
}

bool PhotonTriggerML::GetMLResultForStation(int stationId, MLResult& result)
{
    auto it=fMLResultsMap.find(stationId);
    if (it!=fMLResultsMap.end()) { result=it->second; return true; }
    return false;
}

void PhotonTriggerML::ClearMLResults() { fMLResultsMap.clear(); }

