import os, sys
import ROOT
import numpy
import root_numpy

import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input root file", type=str)
parser.add_argument("--reference", help = "reference npz file", type=str, default="dnn_scores_Hadronic_v1.7_9Jul2019_scaleByXs_impute_hct_FCNC_vs_SMHiggs_addMassConstraint_.npz")
args = parser.parse_args()

f = ROOT.TFile(args.input)
tree = f.Get("t")

feature_names = ["mass", "weight", "sample_id", "fcnc_vs_smhiggs_dnn", "mva_score", "tth_runII_mva"]

validation_events = root_numpy.tree2array(tree, branches = feature_names, selection = "train_id == 1 && ((process_id == 0 && signal_mass_label == 0) || (process_id == 11 || process_id == 12) || (process_id >= 22))")
data_events = root_numpy.tree2array(tree, branches = feature_names, selection = "process_id == 10")

y_test = validation_events["sample_id"]
pred_bdt = validation_events["mva_score"]
pred_dnn = validation_events["fcnc_vs_smhiggs_dnn"]
weights_test = validation_events["weight"]
tth_mva = validation_events["tth_runII_mva"]

print numpy.sum(weights_test[numpy.where(y_test == 0)])
print numpy.sum(weights_test[numpy.where(y_test == 1)])

auc_bdt, unc_bdt, fpr_bdt, tpr_bdt, thresh_bdt = utils.auc_and_unc(y_test, pred_bdt, weights_test, 25)
auc_dnn, unc_dnn, fpr_dnn, tpr_dnn, thresh_dnn = utils.auc_and_unc(y_test, pred_dnn, weights_test, 25)

print(auc_bdt)
print(auc_dnn)

#reference = numpy.load(args.reference)

#for i in range(len(data_events["mass"])):
#    for j in range(len(reference["mass_data"])):
#        if data_events["mass"][i] == reference["mass_data"][j]:
#            print "DNN score from tree: %.4f, from ref: %.4f" % (data_events["fcnc_vs_smhiggs_dnn"][i], reference["scores_data"][j])
