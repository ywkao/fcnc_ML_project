import os, sys
import ROOT
import numpy
import root_numpy

import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input root file", type=str)
args = parser.parse_args()

f = ROOT.TFile(args.input)
tree = f.Get("t")

feature_names = ["weight", "sample_id", "ttH_vs_tH_DNN", "mva_score", "tth_runII_mva"]

validation_events = root_numpy.tree2array(tree, branches = feature_names, selection = "train_id == 1 && ((process_id == 0 && signal_mass_label == 0) || process_id == 11 || process_id == 12)")

y_test = validation_events["sample_id"]
pred_bdt = validation_events["mva_score"]
pred_dnn = validation_events["ttH_vs_tH_DNN"]
weights_test = validation_events["weight"]
tth_mva = validation_events["tth_runII_mva"]

auc_bdt, unc_bdt, fpr_bdt, tpr_bdt, thresh_bdt = utils.auc_and_unc(y_test, pred_bdt, weights_test, 25)
auc_dnn, unc_dnn, fpr_dnn, tpr_dnn, thresh_dnn = utils.auc_and_unc(y_test, pred_dnn, weights_test, 25)

print(auc_bdt)
print(auc_dnn)

tth_sr_bounds = [0.8435, 0.9346, 0.9625, 0.9890]
sr_0 = weights_test[numpy.where(tth_mva > tth_sr_bounds[3])]
sr_1 = weights_test[numpy.where((tth_mva > tth_sr_bounds[2]) & (tth_mva < tth_sr_bounds[3]))]
sr_2 = weights_test[numpy.where((tth_mva > tth_sr_bounds[1]) & (tth_mva < tth_sr_bounds[2]))]
sr_3 = weights_test[numpy.where((tth_mva > tth_sr_bounds[0]) & (tth_mva < tth_sr_bounds[1]))]

sr = { "0" : sr_0, "1" : sr_1, "2" : sr_2, "3" : sr_3 }

returned_th = {}
sacrificed_tth = {}

signal_effs = [1.0, 0.99, 0.95, 0.8]
for eff in signal_effs:
    value, idx = utils.find_nearest(tpr_dnn, eff)
    #value, idx = utils.find_nearest(fpr_dnn, 0.05)
    cut = thresh_dnn[idx]
    returned_th[str(eff)] = { "weight" : [], "tth_mva" : [] }
    sacrificed_tth[str(eff)] = { "weight" : [], "tth_mva" : [] , "SR_0" : [], "SR_1" : [], "SR_2" : [], "SR_3" : []}
    for i in range(len(validation_events)):
        if y_test[i] == 0 and pred_dnn[i] < cut:
            returned_th[str(eff)]["weight"].append(weights_test[i])
            returned_th[str(eff)]["tth_mva"].append(tth_mva[i])
        elif y_test[i] == 1 and pred_dnn[i] < cut:
            sacrificed_tth[str(eff)]["weight"].append(weights_test[i])
            sacrificed_tth[str(eff)]["tth_mva"].append(tth_mva[i])
            if tth_mva[i] > tth_sr_bounds[3]:
                sacrificed_tth[str(eff)]["SR_0"].append(weights_test[i])
            elif tth_mva[i] > tth_sr_bounds[2]:
                sacrificed_tth[str(eff)]["SR_1"].append(weights_test[i])
            elif tth_mva[i] > tth_sr_bounds[1]:
                sacrificed_tth[str(eff)]["SR_2"].append(weights_test[i])
            elif tth_mva[i] > tth_sr_bounds[0]:
                sacrificed_tth[str(eff)]["SR_3"].append(weights_test[i])


#print(numpy.sum(sacrificed_tth["weight"]))
#print(numpy.mean(sacrificed_tth["tth_mva"]))

for eff in signal_effs:
    for i in range(len(tth_sr_bounds)):
        print("SR %d loses %.2f%% of its original events (%.2f)" % (i, 100.*(numpy.sum(sacrificed_tth[str(eff)]["SR_%s" % str(i)]) / numpy.sum(sr[str(i)])), numpy.sum(sr[str(i)]))) 
        #print "%.2f%% of lost events come from SR %d" % (100.*(numpy.sum(sacrificed_tth[str(eff)]["SR_%s" % str(i)]) / (numpy.sum(sacrificed_tth[str(eff)]["weight"]))), i)

    print("And %.2f%% of the originally stolen tH events are returned (%.2f)" % (100.*(numpy.sum(returned_th[str(eff)]["weight"]) / numpy.sum(weights_test[numpy.where(y_test == 0)])), numpy.sum(weights_test[numpy.where(y_test == 0)])))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

groups = ['SR0 (Tightest)', 'SR1', 'SR2', 'SR3 (Loosest)']

x = numpy.arange(len(groups))
width = 0.2

labels = []
for eff in signal_effs:
    labels += [r'%.0f%% $\epsilon_{ttH}$' % (eff*100.)]
labels[0] = "No cut"

print labels
rects_set = [] 

for i in range(len(signal_effs)):
    yields = []
    for j in range(len(groups)):
        yields.append(numpy.sum(sr[str(j)]) - numpy.sum(sacrificed_tth[str(signal_effs[i])]["SR_%s" % str(j)])) 
    x_vals = x + (((2*float(i)) - 3) / 2) * width
    rects_set.append(ax1.bar(numpy.array(x_vals), numpy.array(yields), width, label = labels[i])) 

ax1.set_ylabel('ttH Events')
ax1.legend()
ax1.set_xticks(x)
ax1.set_xticklabels(groups)

def autolabel(rects, ref_height):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i in range(len(rects)):
        rect = rects[i]
        height = rect.get_height()
        label = 100.* (rect.get_height() / ref_height[i])
        ax1.annotate(('%.1f%%' % label) if label < 100 else "100%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6)


ref_height = [rect.get_height() for rect in rects_set[0]]
for rect in rects_set:
    autolabel(rect, ref_height)


fig.tight_layout()
plt.savefig('ttH_vs_tH_SR_inspection.pdf')
