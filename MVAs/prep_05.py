# imports and parser{{{
import os, sys
import h5py
import ROOT
import numpy
import root_numpy
import json

#import dnn_helper # will lead to a fatal error of not finding cuda_runtime.h 
import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input root file", type=str)
parser.add_argument("--channel", help = "Hadronic or Leptonic", type=str, default = "Hadronic")
parser.add_argument("--tag", help = "name to append to filename", type=str, default = "")

parser.add_argument("--dnn_models", help = "csv list of dnn models to add as inputs", type=str, default = "")
parser.add_argument("--dont_train_with_dnn", help = "don't add DNN scores as input features (one might include a dnn_model but not include it in training if one wanted to use it as a reference MVA in train.py", action = "store_true")
parser.add_argument("--do_top_tag", help = "add top tag score", action = "store_true")

parser.add_argument("-r", "--randomize", help = "use a random test/train split", action="store_true")
parser.add_argument("-i", "--invert", help = "invert the test/train split", action="store_true")
parser.add_argument("-s", "--sideband", help = "use data sideband for training", action = "store_true")

parser.add_argument("--train_frac", help = "what fraction of data to use for training", type = float, default = 0.5)
parser.add_argument("--handicap", help = "use only 1/3 of training data (to get a feel for how much stats matter)", action = "store_true")
parser.add_argument("--sideband_name", help = "which data sideband to use", type=str, default="none")
parser.add_argument("--old_vars", help = "remove top tagger, delta r of photons, and helicity", action="store_true")
parser.add_argument("--no_psv", help = "remove pixel seed veto", action = "store_true")
parser.add_argument("--dipho_only", help = "use only diphoton as bkg", action = "store_true")
parser.add_argument("--ttGG_only", help = "use only ttGG as bkg", action = "store_true")
parser.add_argument("--no_lepton_id", help = "don't use lepton ID vars", action = "store_true")
parser.add_argument("--one_mass_point", help = "only use one signal mass point in training", action = "store_true")
parser.add_argument("--cut_ptoM", help = "apply cuts on photon pT/mgg", action = "store_true")
parser.add_argument("--fcnc_hut", help = "use FCNC Hut as signal, other SM Higgs processes as bkg", action = "store_true")
parser.add_argument("--fcnc_hct", help = "use FCNC Hct as signal, other SM Higgs processes as bkg", action = "store_true")
parser.add_argument("--ttH_vs_tH", help = "use ttH as signal, tH as background", action = "store_true")
parser.add_argument("--FCNC_vs_SMHiggs", help = "use FCNC as signal, SM Higgs as background", action="store_true")
parser.add_argument("--add_year", help = "add the year as a feature", action = "store_true")
parser.add_argument("--no_mass_constraint", help = "don't use m_ggj, m_jjj", action = "store_true")
parser.add_argument("--non_resonant_bkg", help = "for FCNC, only train with non-resonant bkgs", action = "store_true")

args = parser.parse_args()
#}}}
# Hello World{{{
print "------------------------------------------------------------"
print "Hello World! (from prep.py)"
print "------------------------------------------------------------"
print "args.input               : %s" % args.input
print "args.channel             : %s" % args.channel
print "args.tag                 : %s" % args.tag
print "args.dnn_models          : %s" % args.dnn_models
print "args.dont_train_with_dnn : %s" % args.dont_train_with_dnn
print "args.do_top_tag          : %s" % args.do_top_tag
print "args.randomize           : %s" % args.randomize
print "args.invert              : %s" % args.invert
print "args.sideband            : %s" % args.sideband
print "args.train_frac          : %s" % args.train_frac
print "args.handicap            : %s" % args.handicap
print "args.sideband_name       : %s" % args.sideband_name
print "args.old_vars            : %s" % args.old_vars
print "args.no_psv              : %s" % args.no_psv
print "args.dipho_only          : %s" % args.dipho_only
print "args.ttGG_only           : %s" % args.ttGG_only
print "args.no_lepton_id        : %s" % args.no_lepton_id
print "args.one_mass_point      : %s" % args.one_mass_point
print "args.cut_ptoM            : %s" % args.cut_ptoM
print "args.fcnc_hut            : %s" % args.fcnc_hut
print "args.fcnc_hct            : %s" % args.fcnc_hct
print "args.ttH_vs_tH           : %s" % args.ttH_vs_tH
print "args.FCNC_vs_SMHiggs     : %s" % args.FCNC_vs_SMHiggs
print "args.add_year            : %s" % args.add_year
print "args.no_mass_constraint  : %s" % args.no_mass_constraint
print "args.non_resonant_bkg    : %s" % args.non_resonant_bkg
print "------------------------------------------------------------"
#}}}

# I/O{{{
baby_file = args.input.replace(".root", "") + ".root"
output_file = args.input.replace(".root", "").replace("../Loopers/MVABaby_","") + "_features" + args.tag + "%s%s%s%s.hdf5" % ("_invert" if args.invert else "", "_remove_new_vars" if args.old_vars else "", "_noPSV" if args.no_psv else "", "noLepID" if args.no_lepton_id else "")

print(output_file)

f = ROOT.TFile(baby_file)
tree = f.Get("t")
#}}}

#feature_names{{{
# load tree to array
#feature_names = (root_numpy.tree2array(tree, branches = ["mva_branches"], start=0, stop=1))[0][0]
#feature_names = list(feature_names) 

#original
#feature_names = ["maxIDMVA_", "minIDMVA_", "max2_btag_", "max1_btag_", "dipho_delta_R", "njets_", "ht_", "leadptoM_", "subleadptoM_", "lead_eta_", "sublead_eta_", "jet1_pt_", "jet1_eta_", "jet1_btag_", "jet2_pt_", "jet2_eta_", "jet2_btag_", "jet3_pt_", "jet3_eta_", "jet3_btag_", "jet4_pt_", "jet4_eta_", "jet4_btag_", "leadPSV_", "subleadPSV_", "dipho_cosphi_", "dipho_rapidity_", "met_", "dipho_pt_over_mass_", "helicity_angle_"] 

# all = topTagger + chi-2
if args.channel == "Hadronic":
    feature_names = [
        "maxIDMVA_", 
        "minIDMVA_", 
        "max2_btag_", 
        "max1_btag_", 
        "dipho_delta_R", 
        "njets_", 
        "ht_", 
        "leadptoM_", 
        "subleadptoM_", 
        "lead_eta_", 
        "sublead_eta_", 
        "jet1_pt_", 
        "jet1_eta_", 
        "jet1_btag_", 
        "jet2_pt_", 
        "jet2_eta_", 
        "jet2_btag_", 
        "jet3_pt_", 
        "jet3_eta_", 
        "jet3_btag_", 
        "jet4_pt_", 
        "jet4_eta_", 
        "jet4_btag_", 
        "leadPSV_", 
        "subleadPSV_", 
        "dipho_cosphi_", 
        "dipho_rapidity_", 
        "met_", 
        "dipho_pt_over_mass_", 
        "helicity_angle_", 
        "mc_mva_score_tt_v2_",
        "mc_mva_score_st_v2_",
        "chi2_tbw_mass_", 
        "chi2_tbw_pt_", 
        "chi2_tbw_eta_", 
        "chi2_tbw_deltaR_dipho_", 
        "chi2_qjet_pt_", 
        "chi2_qjet_eta_", 
        "chi2_qjet_deltaR_dipho_", 
        "chi2_tqh_ptOverM_", 
        "chi2_tqh_eta_", 
        "chi2_tqh_deltaR_tbw_", 
        "chi2_tqh_deltaR_dipho_", 
        "chi2_3x3_tbw_mass_", 
        "chi2_3x3_tbw_pt_", 
        "chi2_3x3_tbw_eta_", 
        "chi2_3x3_tbw_deltaR_dipho_", 
        "chi2_3x3_qjet_pt_", 
        "chi2_3x3_qjet_eta_", 
        "chi2_3x3_qjet_btag_", 
        "chi2_3x3_qjet_deltaR_dipho_", 
        "chi2_3x3_tqh_ptOverM_", 
        "chi2_3x3_tqh_eta_", 
        "chi2_3x3_tqh_deltaR_tbw_", 
        "chi2_3x3_tqh_deltaR_dipho_",
        "chi2_bjet_btagScores_",
        "chi2_wjet1_btagScores_",
        "chi2_wjet2_btagScores_",
        "chi2_qjet_btagScores_"
        ]
elif args.channel == "Leptonic":
    feature_names = [
        "maxIDMVA_", 
        "minIDMVA_", 
        "max2_btag_", 
        "max1_btag_", 
        "dipho_delta_R", 
        "njets_", 
        "ht_", 
        "leadptoM_", 
        "subleadptoM_", 
        "lead_eta_", 
        "sublead_eta_", 
        "jet1_pt_", 
        "jet1_eta_", 
        "jet1_btag_", 
        "jet2_pt_", 
        "jet2_eta_", 
        "jet2_btag_", 
        "jet3_pt_", 
        "jet3_eta_", 
        "jet3_btag_", 
        "jet4_pt_", 
        "jet4_eta_", 
        "jet4_btag_", 
        "leadPSV_", 
        "subleadPSV_", 
        "dipho_cosphi_", 
        "dipho_rapidity_", 
        "met_", 
        "dipho_pt_over_mass_", 
        "helicity_angle_", 
        "mc_mva_score_tt_v2_",
        "mc_mva_score_st_v2_",
        "chi2_neutrino_pz_", 
        "chi2_tbw_mass_", 
        "chi2_tbw_pt_", 
        "chi2_tbw_eta_", 
        "chi2_tbw_deltaR_dipho_", 
        "chi2_qjet_pt_", 
        "chi2_qjet_eta_", 
        "chi2_qjet_deltaR_dipho_", 
        "chi2_tqh_ptOverM_", 
        "chi2_tqh_eta_", 
        "chi2_tqh_deltaR_tbw_", 
        "chi2_tqh_deltaR_dipho_",
        "chi2_bjet_btagScores_",
        "chi2_qjet_btagScores_"
    ]

if args.do_top_tag:
  feature_names += ["top_tag_score_"]

if args.add_year:
  feature_names += ["year_"]

if (args.fcnc_hut or args.fcnc_hct) and args.channel == "Hadronic" and not args.no_mass_constraint:
  feature_names += ["m_ggj_", "m_jjj_"]

if args.ttH_vs_tH:
  feature_names += ["lep1_charge_", "lep2_charge_", "forward_jet_eta_", "forward_jet_pt_"]

to_remove = []
if args.channel == "Leptonic":
  feature_names += ["lep_pt_", "lep_eta_", "n_lep_tight_"]
  #feature_names += ["lep_pt_", "lep_eta_", "n_lep_loose_", "n_lep_medium_", "n_lep_tight_", "muon1_mini_iso_", "muon2_mini_iso_"]
  if args.no_lepton_id:
    to_remove += ["n_lep_tight_"]
    #to_remove += ["n_lep_loose_", "n_lep_medium_", "n_lep_tight_", "muon1_mini_iso_", "muon2_mini_iso_"] 
  to_remove += ["jet4_pt_", "jet4_eta_", "jet4_btag_"]

if args.sideband: # remove b-tagging features
  if args.sideband_name == "0b":
    to_remove += ["max2_btag_", "max1_btag_", "jet1_btag_", "jet2_btag_", "jet3_btag_", "jet4_btag_", "top_tag_score_"]
  elif args.sideband_name == "phoID":
    to_remove += ["leadIDMVA_", "subleadIDMVA_", "maxIDMVA_", "minIDMVA_"]

if args.old_vars:
  to_remove += ["top_tag_score_", "dipho_delta_R", "helicity_angle_"]

if args.no_psv:
  to_remove += ["leadPSV_", "subleadPSV_"]

if args.channel == "Hadronic":
  branches = numpy.concatenate((feature_names, ["evt_weight_", "label_", "multi_label_", "process_id_", "mass_", "lead_sigmaEtoE_", "sublead_sigmaEtoE_", "tth_ttX_mva_", "tth_qcdX_mva_", "tth_ttPP_mva_", "lead_phi_", "sublead_phi_", "log_met_", "met_phi_", "signal_mass_label_", "tth_2017_reference_mva_", "evt_", "run_", "lumi_", "year_", "tth_runII_mva_", "signal_mass_category_"]))
  #branches = numpy.concatenate((feature_names, ["evt_weight_", "label_", "multi_label_", "process_id_", "mass_", "lead_sigmaEtoE_", "sublead_sigmaEtoE_", "tth_ttX_mva_", "tth_qcdX_mva_", "tth_ttPP_mva_", "objects_", "lead_phi_", "sublead_phi_", "log_met_", "met_phi_", "signal_mass_label_", "tth_2017_reference_mva_", "evt_", "run_", "lumi_", "year_", "tth_runII_mva_", "signal_mass_category_"]))
elif args.channel == "Leptonic":
  branches = numpy.concatenate((feature_names, ["evt_weight_", "label_", "multi_label_", "process_id_", "mass_", "lead_sigmaEtoE_", "sublead_sigmaEtoE_", "lead_phi_", "sublead_phi_", "log_met_", "met_phi_", "signal_mass_label_", "tth_2017_reference_mva_", "evt_", "run_", "lumi_", "year_", "tth_runII_mva_", "signal_mass_category_"]))
  #branches = numpy.concatenate((feature_names, ["evt_weight_", "label_", "multi_label_", "process_id_", "mass_", "lead_sigmaEtoE_", "sublead_sigmaEtoE_", "objects_", "lead_phi_", "sublead_phi_", "log_met_", "met_phi_", "signal_mass_label_", "tth_2017_reference_mva_", "evt_", "run_", "lumi_", "year_", "tth_runII_mva_", "signal_mass_category_"]))
#}}}
# grab features{{{
train_frac = args.train_frac
rand_branch = "super_rand_" if args.randomize else "rand_"

data_label = 2

branches = list(set(branches))

if args.FCNC_vs_SMHiggs:
    if args.channel == "Leptonic":
        fcnc_sm_higgs_bkg_selection = " && (process_id_ == 0 || process_id_ == 11 || process_id_ == 12)"
    elif args.channel == "Hadronic":
        fcnc_sm_higgs_bkg_selection = " && (process_id_ == 0 || process_id_ == 11 || process_id_ == 14)"
else:
    fcnc_sm_higgs_bkg_selection =  ""

if args.non_resonant_bkg:
    non_resonant_train_selection = " && !(process_id_ == 0 || (process_id_ >= 11 && process_id_ <= 16))"
else:
    non_resonant_train_selection = ""

if args.fcnc_hut:
    selection_train      = '((label_ == 0%s%s%s%s) || (label_ == 1 && (process_id_ == 22 || process_id_ == 24))) && %s %s %.6f %s' % (fcnc_sm_higgs_bkg_selection, non_resonant_train_selection, " && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, ">" if args.invert else "<", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")
    selection_validation = '((label_ == 0%s%s%s && !(process_id_ == 0 && signal_mass_label_ != 0)) || (label_ == 1 && (process_id_ == 22 || process_id_ == 24))) && %s %s %.6f %s' % (fcnc_sm_higgs_bkg_selection, " && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, "<" if args.invert else ">", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")

    selection_final_fit      = '((process_id_ == 2 && rand_ < 0.01))' # dummy selection

elif args.fcnc_hct:
    selection_train      = '((label_ == 0%s%s%s%s) || (label_ == 1 && (process_id_ == 23 || process_id_ == 25))) && %s %s %.6f %s' % (fcnc_sm_higgs_bkg_selection, non_resonant_train_selection, "&& process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, ">" if args.invert else "<", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")
    selection_validation = '((label_ == 0%s%s%s && !(process_id_ == 0 && signal_mass_label_ != 0)) || (label_ == 1 && (process_id_ == 23 || process_id_ == 25))) && %s %s %.6f %s' % (fcnc_sm_higgs_bkg_selection, " && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, "<" if args.invert else ">", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")

    selection_final_fit      = '((process_id_ == 2 && rand_ < 0.01))' # dummy selection

elif args.ttH_vs_tH:
  selection_train      = '((label_ == 0%s%s && (process_id_ == 11 || process_id_ == 12)) || (label_ == 1 && signal_mass_label_ != 0)) && %s %s %.6f %s' % (" && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, ">" if args.invert else "<", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")

  selection_validation = '((label_ == 0%s%s && (process_id_ == 11 || process_id_ == 12)) || (label_ == 1 && signal_mass_label_ == 1)) && %s %s %.6f %s' % (" && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, "<" if args.invert else ">", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")

  selection_final_fit      = '((label_ == 1 && signal_mass_label_ == 0))'

else:
  selection_train      = '((label_ == 0%s%s && signal_mass_label_ != 0) || (label_ == 1 && signal_mass_label_ != 0)) && %s %s %.6f %s' % (" && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, ">" if args.invert else "<", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")

  if args.one_mass_point:
    selection_train      = '((label_ == 0%s%s) || (label_ == 1 && signal_mass_label_ == 1)) && %s %s %.6f %s' % (" && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, ">" if args.invert else "<", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")

  selection_validation = '((label_ == 0%s%s && (signal_mass_label_ == 1 || signal_mass_label_ == -1)) || (label_ == 1 && signal_mass_label_ == 1)) && %s %s %.6f %s' % (" && process_id_ == 2" if args.dipho_only else "", " && (process_id_ == 5 && abs(evt_weight_) < 0.01)" if args.ttGG_only else "", rand_branch, "<" if args.invert else ">", train_frac, "&& data_sideband_label_ == 0" if args.sideband else "")

  selection_final_fit      = '((label_ == 1 && signal_mass_label_ == 0))' 
#}}}
# others{{{
ptoM_cut = ""
if args.cut_ptoM:
  if args.channel == "Hadronic":
    ptoM_cut = "&& (leadptoM_ > 0.333 && subleadptoM_ > 0.25)"
  elif args.channel == "Leptonic":
    ptoM_cut = "&& (leadptoM_ > 0.33 && subleadptoM_ > 0.25)"
  selection_train += ptoM_cut
  selection_validation += ptoM_cut
  selection_final_fit += ptoM_cut

print(("Selection for training events: %s" % selection_train))
print(("Selection for validation events: %s" % selection_validation))
print(("Selection for final fit events: %s" % selection_final_fit))

features = root_numpy.tree2array(tree, branches = branches, selection = selection_train) 
features_validation = root_numpy.tree2array(tree, branches = branches, selection = selection_validation) 
features_final_fit = root_numpy.tree2array(tree, branches = branches, selection = selection_final_fit) 


# Scale ttH training weights
#if args.fcnc_hut or args.fcnc_hct:
#    for i in range(len(features["process_id_"])):
#        if features["process_id_"][i] == 0: # scale ttH by 1/7 bc we use 7 mass points in training
#            features["evt_weight_"][i] *= 1.0/7.0

if args.sideband:
  handicap = ""
  if args.handicap:
    handicap = " && %s < 0.5" % (rand_branch)
  features_data_sideband = root_numpy.tree2array(tree, branches = branches, selection = '(data_sideband_label_ == 1 && label_ == 2 %s) || (label_ == 1 && %s < %.6f && data_sideband_label_ == 0)' % (handicap, rand_branch, train_frac)) 
  features_data_sideband_mc = root_numpy.tree2array(tree, branches = branches, selection = '(data_sideband_label_ == 1 && label_ == 0 %s) || (label_ == 1 && %s < %.6f && data_sideband_label_ == 0)' % (handicap, rand_branch, train_frac))

features_data = root_numpy.tree2array(tree, branches = branches, selection = 'label_ == %d %s %s' % (data_label, "&& data_sideband_label_ == 0" if args.sideband else "", ptoM_cut))
#}}}

print "before dnn"
# Calculate DNN scores{{{
dnn_models = args.dnn_models.split(",")
do_dnn = len(args.dnn_models) > 0 and (not args.dont_train_with_dnn)

dnn_predictions = []
dnn_features = ["lead_eta_", "sublead_eta_", "lead_phi_", "sublead_phi_", "leadptoM_", "subleadptoM_", "maxIDMVA_", "minIDMVA_", "log_met_", "met_phi_", "leadPSV_", "subleadPSV_", "dipho_rapidity_", "dipho_pt_over_mass_", "dipho_delta_R", "max1_btag_", "max2_btag_", "njets_"]
if args.channel == "Leptonic":
    dnn_features += ["n_lep_tight_"]

if args.ttH_vs_tH:
    dnn_features += ["lep1_charge_", "lep2_charge_", "forward_jet_eta_", "forward_jet_pt_"]

#if args.do_top_tag:
#    dnn_features += ["top_tag_score_"]
if (args.fcnc_hut or args.fcnc_hct) and args.channel == "Hadronic" and not args.no_mass_constraint:
    dnn_features += ["m_ggj_", "m_jjj_"] 

if do_dnn:
  print("Calculating dnn scores")
  print((len(dnn_features)))
  print([feat for feat in dnn_features])
  i = 0
  print(dnn_features)
  for dnn_model in dnn_models:
    with open(dnn_model, "r") as f_in:
      model = json.load(f_in)

    dnn_features_data = dnn_helper.DNN_Features(name = 'data', global_features = utils.create_array(features_data, dnn_features, model["preprocess_scheme"], True), objects = utils.preprocess_array(utils.pad_array(features_data["objects_"]), model["preprocess_scheme"]))
    #print dnn_features_data.global_features
    #print dnn_features_data.objects
    #print dnn_features_data.features
    dnn_features_validation = dnn_helper.DNN_Features(name = 'validation', global_features = utils.create_array(features_validation, dnn_features, model["preprocess_scheme"], True), objects = utils.preprocess_array(utils.pad_array(features_validation["objects_"]), model["preprocess_scheme"]))
    dnn_features_final_fit = dnn_helper.DNN_Features(name = 'final_fit', global_features = utils.create_array(features_final_fit, dnn_features, model["preprocess_scheme"], True), objects = utils.preprocess_array(utils.pad_array(features_final_fit["objects_"]), model["preprocess_scheme"]))
    dnn_features_train = dnn_helper.DNN_Features(name = 'train', global_features = utils.create_array(features, dnn_features, model["preprocess_scheme"], True), objects = utils.preprocess_array(utils.pad_array(features["objects_"]), model["preprocess_scheme"]))

    dnn = dnn_helper.DNN_Helper(features_validation = dnn_features_validation, features_train = dnn_features_train, features_data = dnn_features_data, features_final_fit = dnn_features_final_fit, metadata = model, weights_file = "dnn_weights/" + model["weights"], train_mode = False)
    #dnn.predict()
    #dnn_predictions.append([dnn.predictions["train"], dnn.predictions["validation"], dnn.predictions["data"]])
    dnn_predictions.append(dnn.predict(debug=True))
    feature_names.append("dnn_score_%d" % i)
    i += 1 

print(dnn_predictions)

#if do_dnn:
#  i = 0
#  dnn_features = ["lead_eta_", "sublead_eta_", "lead_phi_", "sublead_phi_", "leadptoM_", "subleadptoM_", "maxIDMVA_", "minIDMVA_", "log_met_", "met_phi_", "leadPSV_", "subleadPSV_", "dipho_rapidity_", "dipho_pt_over_mass_", "dipho_delta_R", "max1_btag_", "max2_btag_", "njets_", "top_tag_score_"]
#  for model in dnn_models:
#    dnn_predictions.append([])
#    for features_set in [features, features_validation, features_data]:
#      dnn = dnn_helper.DNN_Helper(objects = features_set["objects_"], global_features = [features_set[feat] for feat in dnn_features], weights_file = model)
#      prediction = dnn.predict()
#      dnn_predictions[i].append(prediction)
#    feature_names.append("dnn_score_%d" % i)
#    i += 1
#}}}

training_feature_names = [feature for feature in feature_names if feature not in to_remove]

print(training_feature_names)

# organize features{{{
global_features = []
global_features_validation = []
global_features_data = []
global_features_final_fit = []
global_features_data_sideband = []
global_features_data_sideband_mc = []
for feature in training_feature_names:
  if "dnn_score" in feature:
    continue
  global_features.append(features[feature])
  global_features_validation.append(features_validation[feature])
  global_features_data.append(features_data[feature])
  global_features_final_fit.append(features_final_fit[feature])
  if args.sideband:
    global_features_data_sideband.append(features_data_sideband[feature])
    global_features_data_sideband_mc.append(features_data_sideband_mc[feature])

if do_dnn:
  for i in range(len(dnn_predictions)):
    global_features.append(dnn_predictions[i][0])
    global_features_validation.append(dnn_predictions[i][1])
    global_features_data.append(dnn_predictions[i][2])
    global_features_final_fit.append(dnn_predictions[i][3])


global_features = numpy.asarray(global_features)
global_features_validation = numpy.asarray(global_features_validation)
global_features_data = numpy.asarray(global_features_data)
global_features_final_fit = numpy.asarray(global_features_final_fit)

if len(args.dnn_models) > 0:
    with open(dnn_models[0], "r") as f_in:
        preprocess_scheme = json.load(f_in)["preprocess_scheme"]
else:
    preprocess_scheme = "none"

# Check if we got a json file or the actual scheme
if ".json" in preprocess_scheme:
    with open(preprocess_scheme) as f_in:
        preprocess_scheme = json.load(f_in)

print("Preprocessing scheme: ", preprocess_scheme)

global_features_dnn = utils.create_array(features, dnn_features, preprocess_scheme) 
global_features_dnn_validation = utils.create_array(features_validation, dnn_features, preprocess_scheme) 
global_features_dnn_data = utils.create_array(features_data, dnn_features, preprocess_scheme) 
global_features_dnn_final_fit = utils.create_array(features_final_fit, dnn_features, preprocess_scheme) 

if args.sideband:
  global_features_data_sideband = numpy.asarray(global_features_data_sideband)
  global_features_data_sideband_mc = numpy.asarray(global_features_data_sideband_mc)

if args.channel == "Hadronic":
  mva_names = ["max1_btag_", "max2_btag_", "maxIDMVA_", "minIDMVA_", "tth_ttX_mva_", "tth_qcdX_mva_", "tth_ttPP_mva_"]
elif args.channel == "Leptonic":
  mva_names = ["max1_btag_", "max2_btag_", "maxIDMVA_", "minIDMVA_"]
  #}}}

# features{{{
label = features["label_"]
multi_label = features["multi_label_"]
weights = features["evt_weight_"]
mass = features["mass_"]
njets = features["njets_"]
lead_sigmaEtoE = features["lead_sigmaEtoE_"]
sublead_sigmaEtoE = features["sublead_sigmaEtoE_"]
signal_mass_label = features["signal_mass_label_"]
signal_mass_category = features["signal_mass_category_"]
tth_2017_reference_mva = features["tth_2017_reference_mva_"]
evt = features["evt_"]
run = features["run_"]
lumi = features["lumi_"]
process_id = features["process_id_"]
year = features["year_"]
#objects = utils.preprocess_array(utils.pad_array(features["objects_"]), preprocess_scheme)
tth_runII_mva = features["tth_runII_mva_"]

mvas = {}
for name in mva_names:
  mvas[name] = numpy.asarray(features[name])
#}}}
# features_validation{{{
label_validation = features_validation["label_"]
multi_label_validation = features_validation["multi_label_"]
weights_validation = features_validation["evt_weight_"]
mass_validation = features_validation["mass_"]
njets_validation = features_validation["njets_"]
signal_mass_label_validation = features_validation["signal_mass_label_"]
signal_mass_category_validation = features_validation["signal_mass_category_"]
tth_2017_reference_mva_validation = features_validation["tth_2017_reference_mva_"]
evt_validation = features_validation["evt_"]
run_validation = features_validation["run_"]
lumi_validation = features_validation["lumi_"]
process_id_validation = features_validation["process_id_"]
year_validation = features_validation["year_"]
#objects_validation = utils.preprocess_array(utils.pad_array(features_validation["objects_"]), preprocess_scheme)
tth_runII_mva_validation = features_validation["tth_runII_mva_"]


mvas_validation = {}
for name in mva_names:
  mvas_validation[name] = numpy.asarray(features_validation[name])
#}}}
# features_data{{{
label_data = features_data["label_"]
multi_label_data = features_data["multi_label_"]
weights_data = features_data["evt_weight_"]
mass_data = features_data["mass_"]
njets_data = features_data["njets_"]
signal_mass_label_data = features_data["signal_mass_label_"]
signal_mass_category_data = features_data["signal_mass_category_"]
tth_2017_reference_mva_data = features_data["tth_2017_reference_mva_"]
evt_data = features_data["evt_"]
run_data = features_data["run_"]
lumi_data = features_data["lumi_"]
process_id_data = features_data["process_id_"]
year_data = features_data["year_"]
#objects_data = utils.preprocess_array(utils.pad_array(features_data["objects_"]), preprocess_scheme)
tth_runII_mva_data = features_data["tth_runII_mva_"]

mvas_data = {}
for name in mva_names:
  mvas_data[name] = numpy.asarray(features_data[name])
#}}}
# features_final_fit{{{
label_final_fit = features_final_fit["label_"]
multi_label_final_fit = features_final_fit["multi_label_"]
weights_final_fit = features_final_fit["evt_weight_"]
mass_final_fit = features_final_fit["mass_"]
njets_final_fit = features_final_fit["njets_"]
signal_mass_label_final_fit = features_final_fit["signal_mass_label_"]
signal_mass_category_final_fit = features_final_fit["signal_mass_category_"]
tth_2017_reference_mva_final_fit = features_final_fit["tth_2017_reference_mva_"]
evt_final_fit = features_final_fit["evt_"]
run_final_fit = features_final_fit["run_"]
lumi_final_fit = features_final_fit["lumi_"]
process_id_final_fit = features_final_fit["process_id_"]
year_final_fit = features_final_fit["year_"]
#objects_final_fit = utils.preprocess_array(utils.pad_array(features_final_fit["objects_"]), preprocess_scheme)
#if len(objects_final_fit) == 0:
#    objects_final_fit = objects_data[0:1]
tth_runII_mva_final_fit = features_final_fit["tth_runII_mva_"]


mvas_final_fit = {}
for name in mva_names:
  mvas_final_fit[name] = numpy.asarray(features_final_fit[name])
#}}}
# features_data_sideband{{{
if args.sideband:
  label_data_sideband = features_data_sideband["label_"]
  multi_label_data_sideband = features_data_sideband["multi_label_"]
  weights_data_sideband = features_data_sideband["evt_weight_"]
  mass_data_sideband = features_data_sideband["mass_"]
  mvas_data_sideband = {}
  for name in mva_names:
    mvas_data_sideband[name] = numpy.asarray(features_data_sideband[name])

  label_data_sideband_mc = features_data_sideband_mc["label_"]
  multi_label_data_sideband_mc = features_data_sideband_mc["multi_label_"]
  weights_data_sideband_mc = features_data_sideband_mc["evt_weight_"]
  mass_data_sideband_mc = features_data_sideband_mc["mass_"]
  mvas_data_sideband_mc = {}
  for name in mva_names:
    mvas_data_sideband_mc[name] = numpy.asarray(features_data_sideband_mc[name])

  label_data_sideband[label_data_sideband == 2] = 0 # reassign label so that it plays nice with xgboost
#}}}
# reorganize features{{{
global_features = numpy.transpose(global_features)
global_features_validation = numpy.transpose(global_features_validation)
global_features_data = numpy.transpose(global_features_data)
global_features_final_fit = numpy.transpose(global_features_final_fit)

if args.sideband:
  global_features_data_sideband = numpy.transpose(global_features_data_sideband)
  global_features_data_sideband_mc = numpy.transpose(global_features_data_sideband_mc)
#}}}

f_out = h5py.File(output_file, "w")

dset_feature_names = f_out.create_dataset("feature_names", data=feature_names)
dset_training_feature_names = f_out.create_dataset("training_feature_names", data=training_feature_names)

#dset_mva = f_out.create_dataset(name, data=mvas[name]){{{
dset_global = f_out.create_dataset("global", data=global_features)
dset_global_dnn = f_out.create_dataset("global_dnn", data=global_features_dnn)
dset_label = f_out.create_dataset("label", data=label)
dset_multi_label = f_out.create_dataset("multi_label", data=multi_label)
dset_weights = f_out.create_dataset("weights", data=weights)
dset_mass = f_out.create_dataset("mass", data=mass)
dset_njets = f_out.create_dataset("njets", data=njets)
dset_lead_sigmaEtoE = f_out.create_dataset("lead_sigmaEtoE", data=lead_sigmaEtoE)
dset_sublead_sigmaEtoE = f_out.create_dataset("sublead_sigmaEtoE", data=sublead_sigmaEtoE)
dset_signal_mass_label  = f_out.create_dataset("signal_mass_label", data=signal_mass_label)
dset_signal_mass_category  = f_out.create_dataset("signal_mass_category", data=signal_mass_category)
dset_tth_2017_reference_mva  = f_out.create_dataset("tth_2017_reference_mva", data=tth_2017_reference_mva)
dset_evt = f_out.create_dataset("evt", data=evt)
dset_run = f_out.create_dataset("run", data=run)
dset_lumi = f_out.create_dataset("lumi", data=lumi)
dset_process_id = f_out.create_dataset("process_id", data=process_id)
dset_year = f_out.create_dataset("year", data=year)
#dset_objects = f_out.create_dataset("objects", data=objects)
dset_tth_runII_mva = f_out.create_dataset("tth_runII_mva", data=tth_runII_mva)

for name in mva_names:
  dset_mva = f_out.create_dataset(name, data=mvas[name])
#}}}
#dset_mva = f_out.create_dataset(name+"_validation", data=mvas_validation[name]){{{
dset_global_validation = f_out.create_dataset("global_validation", data=global_features_validation)
dset_global_dnn_validation = f_out.create_dataset("global_dnn_validation", data=global_features_dnn_validation)
dset_label_validation = f_out.create_dataset("label_validation", data=label_validation)
dset_multi_label_validation = f_out.create_dataset("multi_label_validation", data=multi_label_validation)
dset_weights_validation = f_out.create_dataset("weights_validation", data=weights_validation)
dset_mass_validation = f_out.create_dataset("mass_validation", data=mass_validation)
dset_njets_validation = f_out.create_dataset("njets_validation", data=njets_validation)
dset_signal_mass_label_validation  = f_out.create_dataset("signal_mass_label_validation", data=signal_mass_label_validation)
dset_signal_mass_category_validation  = f_out.create_dataset("signal_mass_category_validation", data=signal_mass_category_validation)
dset_tth_2017_reference_mva_validation  = f_out.create_dataset("tth_2017_reference_mva_validation", data=tth_2017_reference_mva_validation)
dset_evt_validation = f_out.create_dataset("evt_validation", data=evt_validation)
dset_run_validation = f_out.create_dataset("run_validation", data=run_validation)
dset_lumi_validation = f_out.create_dataset("lumi_validation", data=lumi_validation)
dset_process_id_validation = f_out.create_dataset("process_id_validation", data=process_id_validation)
dset_year_validation = f_out.create_dataset("year_validation", data=year_validation)
#dset_objects_validation = f_out.create_dataset("objects_validation", data=objects_validation)
dset_tth_runII_mva_validation = f_out.create_dataset("tth_runII_mva_validation", data=tth_runII_mva_validation)

for name in mva_names:
  dset_mva = f_out.create_dataset(name+"_validation", data=mvas_validation[name])
#}}}
#dset_mva = f_out.create_dataset(name+"_data", data=mvas_data[name]){{{
dset_global_data = f_out.create_dataset("global_data", data=global_features_data)
dset_global_dnn_data = f_out.create_dataset("global_dnn_data", data=global_features_dnn_data)
dset_label_data = f_out.create_dataset("label_data", data=label_data)
dset_multi_label_data = f_out.create_dataset("multi_label_data", data=multi_label_data)
dset_weights_data = f_out.create_dataset("weights_data", data=weights_data)
dset_mass_data = f_out.create_dataset("mass_data", data=mass_data)
dset_njets_data = f_out.create_dataset("njets_data", data=njets_data)
dset_signal_mass_label_data  = f_out.create_dataset("signal_mass_label_data", data=signal_mass_label_data)
dset_signal_mass_category_data  = f_out.create_dataset("signal_mass_category_data", data=signal_mass_category_data)
dset_tth_2017_reference_mva_data  = f_out.create_dataset("tth_2017_reference_mva_data", data=tth_2017_reference_mva_data)
dset_evt_data = f_out.create_dataset("evt_data", data=evt_data)
dset_run_data = f_out.create_dataset("run_data", data=run_data)
dset_lumi_data = f_out.create_dataset("lumi_data", data=lumi_data)
dset_process_id_data = f_out.create_dataset("process_id_data", data=process_id_data)
dset_year_data = f_out.create_dataset("year_data", data=year_data)
#dset_objects_data = f_out.create_dataset("objects_data", data=objects_data)
dset_tth_runII_mva_data = f_out.create_dataset("tth_runII_mva_data", data=tth_runII_mva_data)

for name in mva_names:
  dset_mva = f_out.create_dataset(name+"_data", data=mvas_data[name])
#}}}
#dset_mva = f_out.create_dataset(name+"_final_fit", data=mvas_final_fit[name]){{{
dset_global_final_fit = f_out.create_dataset("global_final_fit", data=global_features_final_fit)
dset_global_dnn_final_fit = f_out.create_dataset("global_dnn_final_fit", data=global_features_dnn_final_fit)
dset_label_final_fit = f_out.create_dataset("label_final_fit", data=label_final_fit)
dset_multi_label_final_fit = f_out.create_dataset("multi_label_final_fit", data=multi_label_final_fit)
dset_weights_final_fit = f_out.create_dataset("weights_final_fit", data=weights_final_fit)
dset_mass_final_fit = f_out.create_dataset("mass_final_fit", data=mass_final_fit)
dset_njets_final_fit = f_out.create_dataset("njets_final_fit", data=njets_final_fit)
dset_signal_mass_label_final_fit  = f_out.create_dataset("signal_mass_label_final_fit", data=signal_mass_label_final_fit)
dset_signal_mass_category_final_fit  = f_out.create_dataset("signal_mass_category_final_fit", data=signal_mass_category_final_fit)
dset_tth_2017_reference_mva_final_fit  = f_out.create_dataset("tth_2017_reference_mva_final_fit", data=tth_2017_reference_mva_final_fit)
dset_evt_final_fit = f_out.create_dataset("evt_final_fit", data=evt_final_fit)
dset_run_final_fit = f_out.create_dataset("run_final_fit", data=run_final_fit)
dset_lumi_final_fit = f_out.create_dataset("lumi_final_fit", data=lumi_final_fit)
dset_process_id_final_fit = f_out.create_dataset("process_id_final_fit", data=process_id_final_fit)
dset_year_final_fit = f_out.create_dataset("year_final_fit", data=year_final_fit)
#dset_objects_final_fit = f_out.create_dataset("objects_final_fit", data=objects_final_fit)
dset_tth_runII_mva_final_fit = f_out.create_dataset("tth_runII_mva_final_fit", data=tth_runII_mva_final_fit)

for name in mva_names:
  dset_mva = f_out.create_dataset(name+"_final_fit", data=mvas_final_fit[name])
#}}}
#  dset_mva = f_out.create_dataset(name+"_data_sideband_mc", data=mvas_data_sideband_mc[name]){{{
if args.sideband:
  dset_global_data_sideband = f_out.create_dataset("global_data_sideband", data=global_features_data_sideband)
  dset_label_data_sideband = f_out.create_dataset("label_data_sideband", data=label_data_sideband)
  dset_multi_label_data_sideband = f_out.create_dataset("multi_label_data_sideband", data=multi_label_data_sideband)
  dset_weights_data_sideband = f_out.create_dataset("weights_data_sideband", data=weights_data_sideband)
  dset_mass_data_sideband = f_out.create_dataset("mass_data_sideband", data=mass_data_sideband)
  for name in mva_names:
    dset_mva = f_out.create_dataset(name+"_data_sideband", data=mvas_data_sideband[name])

  dset_global_data_sideband_mc = f_out.create_dataset("global_data_sideband_mc", data=global_features_data_sideband_mc)
  dset_label_data_sideband_mc = f_out.create_dataset("label_data_sideband_mc", data=label_data_sideband_mc)
  dset_multi_label_data_sideband_mc = f_out.create_dataset("multi_label_data_sideband_mc", data=multi_label_data_sideband_mc)
  dset_weights_data_sideband_mc = f_out.create_dataset("weights_data_sideband_mc", data=weights_data_sideband_mc)
  dset_mass_data_sideband_mc = f_out.create_dataset("mass_data_sideband_mc", data=mass_data_sideband_mc)
  for name in mva_names:
    dset_mva = f_out.create_dataset(name+"_data_sideband_mc", data=mvas_data_sideband_mc[name])
  #}}}

f_out.close()
print "the end"
