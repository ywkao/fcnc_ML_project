# FCNC ML Project

After the discovery of the Higgs boson in 2012, studies on its properties and interactions become more intriguing.
One of them is an interaction involved a top quark and a Higgs boson, $t \to Hq$ or $q \to tH$, which is known as Flavor-Changing Neutral Higgs (FCNH).
According to the Standard Model of particle physics (SM),
the branching fractions of FCNH are predicted to be very small O(10^{-15}) due to the prohibition in tree diagrams and the GIM mechnism.
However, any observation among the current LHC data could provide a substantial support for theories beyond the Standard Model (BSM).
This motivates the search for this kind of rare interactions.

In high energy physics community (HEP), Machine Learning techniques are widely used to identify out particular signatures or rare events we are interests in.
In the analysis, training a binomial algorithm for differentiating signal from background is one of the major tasks.
This page is meant for...
- Providing information of input variables
- Providing example code for Machine Learning study

The example code here is based on the XGBoost framework [1].
It is a simplified version derived from this working repository [2] (which is only used for developing analysis strategy not for final results). 
More details on the analysis can be found in this link [3].

[1] https://xgboost.readthedocs.io/en/latest/
[2] https://github.com/ywkao/ttH/tree/local_study
[3] https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=TOP-20-007

## procedure
There are three steps for running the example code:
1. prep: convert a root file to a hdf5 file for xgboost training 
2. train: execute training
3. make za plots: check performance of the training

Commands for execution are as follows:
'''
# download
git clone git@github.com:ywkao/fcnc_ML_project.git

# change path of the root files

# run the code
cd fcnc_ML_project/scripts
cmsenv # optional
vi exe.sh # check which step to run
time ./exe.sh
'''

## process id
process_id_ ==  0 : ttHJet / ttHToGG
process_id_ ==  1 : DY
process_id_ ==  2 : DiPhoton
process_id_ ==  3 : GJet_Pt
process_id_ ==  4 : QCD
process_id_ ==  5 : TTGG
process_id_ ==  6 : TTGJets
process_id_ ==  7 : WG / ZG
process_id_ ==  8 : WJets
process_id_ ==  9 : TT (Jets/2L2Nu/SemiLeptonic)
process_id_ == 10 : DoubleEG / EGamma
process_id_ == 11 : THQ
process_id_ == 12 : THW
process_id_ == 13 : TGJets
process_id_ == 14 : GluGluHToGG
process_id_ == 15 : VBF
process_id_ == 16 : VHToGG
process_id_ == 17 : GJets_HT
process_id_ == 18 : imputed_QCD_GJets
process_id_ == 19 : TTZ
process_id_ == 20 : WW / WZ / ZZ
process_id_ == 21 : ST_tW / tZq
process_id_ == 22 : TT_FCNC_hut
process_id_ == 23 : TT_FCNC_hct
process_id_ == 24 : ST_FCNC_hut
process_id_ == 25 : ST_FCNC_hct
process_id_ == 26 : TTW

source code: https://github.com/ywkao/ttH/blob/local_study/Loopers/ttHLooper.h#L369-L445

## variables
All the variables are stored in a tree, "t".
A full list of variables are provided as follows.

### variables for event info
mva_branches                : vector<string>
year_                       : year_/I
evt_                        : evt_/l
run_                        : run_/l
lumi_                       : lumi_/l
evt_weight_                 : evt_weight_/F
label_                      : label_/I
multi_label_                : multi_label_/I
signal_mass_label_          : signal_mass_label_/I
signal_mass_category_       : signal_mass_category_/I
data_sideband_label_        : data_sideband_label_/I
genPhotonId_                : genPhotonId_/I
process_id_                 : process_id_/I
rand_                       : rand_/F
super_rand_                 : super_rand_/F
mass_                       : mass_/F
tth_runII_mva_              : tth_runII_mva_/F
tth_2017_reference_mva_     : tth_2017_reference_mva_/F
tth_qcdX_mva_               : tth_qcdX_mva_/F
tth_ttX_mva_                : tth_ttX_mva_/F
tth_ttPP_mva_               : tth_ttPP_mva_/F
tth_dipho_mva_              : tth_dipho_mva_/F
tth_std_mva_                : tth_std_mva_/F
lead_sigmaEtoE_             : lead_sigmaEtoE_/F
sublead_sigmaEtoE_          : sublead_sigmaEtoE_/F
objects_                    : vector<vector<float> >
objects_boosted_            : vector<vector<float> >
top_candidates_             : vector<float>

### variables for basic kinematics & identification
maxIDMVA_                   : maxIDMVA_/F
minIDMVA_                   : minIDMVA_/F
max2_btag_                  : max2_btag_/F
max1_btag_                  : max1_btag_/F
dipho_delta_R               : dipho_delta_R/F
njets_                      : njets_/F
nbjets_                     : nbjets_/I
ht_                         : ht_/F
top_tag_score_              : top_tag_score_/F
top_tag_mass_               : top_tag_mass_/F
top_tag_pt_                 : top_tag_pt_/F
top_tag_eta_                : top_tag_eta_/F
top_tag_phi_                : top_tag_phi_/F
jet1_pt_                    : jet1_pt_/F
jet1_eta_                   : jet1_eta_/F
jet1_btag_                  : jet1_btag_/F
jet2_pt_                    : jet2_pt_/F
jet2_eta_                   : jet2_eta_/F
jet2_btag_                  : jet2_btag_/F
jet3_pt_                    : jet3_pt_/F
jet3_eta_                   : jet3_eta_/F
jet3_btag_                  : jet3_btag_/F
jet4_pt_                    : jet4_pt_/F
jet4_eta_                   : jet4_eta_/F
jet4_btag_                  : jet4_btag_/F
jet5_pt_                    : jet5_pt_/F
jet5_eta_                   : jet5_eta_/F
jet5_btag_                  : jet5_btag_/F
jet6_pt_                    : jet6_pt_/F
jet6_eta_                   : jet6_eta_/F
jet6_btag_                  : jet6_btag_/F
jet7_pt_                    : jet7_pt_/F
jet7_eta_                   : jet7_eta_/F
jet7_btag_                  : jet7_btag_/F
jet8_pt_                    : jet8_pt_/F
jet8_eta_                   : jet8_eta_/F
jet8_btag_                  : jet8_btag_/F
jet1_phi_                   : jet1_phi_/F
jet1_energy_                : jet1_energy_/F
jet2_phi_                   : jet2_phi_/F
jet2_energy_                : jet2_energy_/F
jet3_phi_                   : jet3_phi_/F
jet3_energy_                : jet3_energy_/F
jet4_phi_                   : jet4_phi_/F
jet4_energy_                : jet4_energy_/F
jet5_phi_                   : jet5_phi_/F
jet5_energy_                : jet5_energy_/F
jet6_phi_                   : jet6_phi_/F
jet6_energy_                : jet6_energy_/F
jet7_phi_                   : jet7_phi_/F
jet7_energy_                : jet7_energy_/F
jet8_phi_                   : jet8_phi_/F
jet8_energy_                : jet8_energy_/F
lead_pT_                    : lead_pT_/F
sublead_pT_                 : sublead_pT_/F
leadptoM_                   : leadptoM_/F
subleadptoM_                : subleadptoM_/F
leadIDMVA_                  : leadIDMVA_/F
subleadIDMVA_               : subleadIDMVA_/F
lead_eta_                   : lead_eta_/F
sublead_eta_                : sublead_eta_/F
leadPSV_                    : leadPSV_/F
subleadPSV_                 : subleadPSV_/F
lead_phi_                   : lead_phi_/F
sublead_phi_                : sublead_phi_/F
dipho_cosphi_               : dipho_cosphi_/F
dipho_rapidity_             : dipho_rapidity_/F
dipho_pt_                   : dipho_pt_/F
dipho_pt_over_mass_         : dipho_pt_over_mass_/F
met_                        : met_/F
log_met_                    : log_met_/F
met_phi_                    : met_phi_/F
helicity_angle_             : helicity_angle_/F
m_ggj_                      : m_ggj_/F
m_jjj_                      : m_jjj_/F
top_candidates_1_           : top_candidates_1_/F
top_candidates_2_           : top_candidates_2_/F
top_candidates_3_           : top_candidates_3_/F
top_candidates_4_           : top_candidates_4_/F
top_candidates_5_           : top_candidates_5_/F
top_candidates_6_           : top_candidates_6_/F
top_candidates_7_           : top_candidates_7_/F
top_candidates_8_           : top_candidates_8_/F
top_candidates_9_           : top_candidates_9_/F
top_candidates_10_          : top_candidates_10_/F
top_candidates_11_          : top_candidates_11_/F
top_candidates_12_          : top_candidates_12_/F

### fcnc NN scores
mc_mva_score_tt_v4_         : mc_mva_score_tt_v4_/F
mc_mva_score_st_v4_         : mc_mva_score_st_v4_/F

### variables derived based on chi-2 method
chi2_tbw_mass_              : chi2_tbw_mass_/F
chi2_tbw_pt_                : chi2_tbw_pt_/F
chi2_tbw_eta_               : chi2_tbw_eta_/F
chi2_tbw_deltaR_dipho_      : chi2_tbw_deltaR_dipho_/F
chi2_qjet_pt_               : chi2_qjet_pt_/F
chi2_qjet_eta_              : chi2_qjet_eta_/F
chi2_qjet_deltaR_dipho_     : chi2_qjet_deltaR_dipho_/F
chi2_tqh_ptOverM_           : chi2_tqh_ptOverM_/F
chi2_tqh_eta_               : chi2_tqh_eta_/F
chi2_tqh_deltaR_tbw_        : chi2_tqh_deltaR_tbw_/F
chi2_tqh_deltaR_dipho_      : chi2_tqh_deltaR_dipho_/F
chi2_3x3_tbw_mass_          : chi2_3x3_tbw_mass_/F
chi2_3x3_tbw_pt_            : chi2_3x3_tbw_pt_/F
chi2_3x3_tbw_eta_           : chi2_3x3_tbw_eta_/F
chi2_3x3_tbw_deltaR_dipho_  : chi2_3x3_tbw_deltaR_dipho_/F
chi2_3x3_qjet_pt_           : chi2_3x3_qjet_pt_/F
chi2_3x3_qjet_eta_          : chi2_3x3_qjet_eta_/F
chi2_3x3_qjet_btag_         : chi2_3x3_qjet_btag_/F
chi2_3x3_qjet_deltaR_dipho_ : chi2_3x3_qjet_deltaR_dipho_/F
chi2_3x3_tqh_ptOverM_       : chi2_3x3_tqh_ptOverM_/F
chi2_3x3_tqh_eta_           : chi2_3x3_tqh_eta_/F
chi2_3x3_tqh_deltaR_tbw_    : chi2_3x3_tqh_deltaR_tbw_/F
chi2_3x3_tqh_deltaR_dipho_  : chi2_3x3_tqh_deltaR_dipho_/F
chi2_bjet_btagScores_       : chi2_bjet_btagScores_/F
chi2_wjet1_btagScores_      : chi2_wjet1_btagScores_/F
chi2_wjet2_btagScores_      : chi2_wjet2_btagScores_/F
chi2_qjet_btagScores_       : chi2_qjet_btagScores_/F
chi2_bjet_ctagScores_       : chi2_bjet_ctagScores_/F
chi2_wjet1_ctagScores_      : chi2_wjet1_ctagScores_/F
chi2_wjet2_ctagScores_      : chi2_wjet2_ctagScores_/F
chi2_qjet_ctagScores_       : chi2_qjet_ctagScores_/F
chi2_bjet_udsgtagScores_    : chi2_bjet_udsgtagScores_/F
chi2_wjet1_udsgtagScores_   : chi2_wjet1_udsgtagScores_/F
chi2_wjet2_udsgtagScores_   : chi2_wjet2_udsgtagScores_/F
chi2_qjet_udsgtagScores_    : chi2_qjet_udsgtagScores_/F
chi2_bjet_CvsL_             : chi2_bjet_CvsL_/F
chi2_wjet1_CvsL_            : chi2_wjet1_CvsL_/F
chi2_wjet2_CvsL_            : chi2_wjet2_CvsL_/F
chi2_qjet_CvsL_             : chi2_qjet_CvsL_/F
chi2_bjet_CvsB_             : chi2_bjet_CvsB_/F
chi2_wjet1_CvsB_            : chi2_wjet1_CvsB_/F
chi2_wjet2_CvsB_            : chi2_wjet2_CvsB_/F
chi2_qjet_CvsB_             : chi2_qjet_CvsB_/F
