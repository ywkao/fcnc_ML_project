# v#im: set fdm=marker:
import os, sys, glob, subprocess
import parallel_utils, func_make_za_plot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tag", help = "tag to denote with", type=str)
parser.add_argument("--baby_version", help = "which version of babies to use", type=str)
parser.add_argument("--prep", help = "enable preparation for training", action="store_true")
parser.add_argument("--train", help = "enable training commands", action="store_true")
parser.add_argument("--make_za_plots", help = "enable making za plots", action="store_true")
args = parser.parse_args()

# global variables
my_own_tag = "_02"
path = "/wk_cms2/ykao/CMSSW_9_4_10/src/ttH/Loopers" # modify the path to where the input files are placed


os.chdir("../MVAs/")
#--------------------------------------------------#
# Convert root files to hdf5 files
#--------------------------------------------------#
if args.prep:
    command_list = []
    command_list.append('python prep%s.py --dont_train_with_dnn --input "%s/MVABaby_ttHLeptonic_%s_FCNC.root" --channel "Leptonic" --fcnc_hut --tag "%s"' % (my_own_tag, path, args.tag + "_hut_BDT", my_own_tag))
    command_list.append('python prep%s.py --dont_train_with_dnn --input "%s/MVABaby_ttHLeptonic_%s_FCNC.root" --channel "Leptonic" --fcnc_hct --tag "%s"' % (my_own_tag, path, args.tag + "_hct_BDT", my_own_tag))
    command_list.append('python prep%s.py --dont_train_with_dnn --input "%s/MVABaby_ttHHadronic_%s_FCNC.root" --channel "Hadronic" --fcnc_hut --tag "%s"' % (my_own_tag, path, args.tag + "_impute_hut_BDT", my_own_tag)) 
    command_list.append('python prep%s.py --dont_train_with_dnn --input "%s/MVABaby_ttHHadronic_%s_FCNC.root" --channel "Hadronic" --fcnc_hct --tag "%s"' % (my_own_tag, path, args.tag + "_impute_hct_BDT", my_own_tag))
    parallel_utils.submit_jobs(command_list, 4)
    print ">>> after prep..."

#--------------------------------------------------#
# Taining with XGBoost
#--------------------------------------------------#
if args.train:
    parallel_utils.run('python train.py --input "ttHLeptonic_%s_FCNC_features%s.hdf5" --channel "Leptonic" --tag "%s" --ext ""' % (args.tag + "_hut_BDT", my_own_tag, my_own_tag + "_hut"))
    parallel_utils.run('python train.py --input "ttHLeptonic_%s_FCNC_features%s.hdf5" --channel "Leptonic" --tag "%s" --ext ""' % (args.tag + "_hct_BDT", my_own_tag, my_own_tag + "_hct"))
    parallel_utils.run('python train.py --input "ttHHadronic_%s_FCNC_features%s.hdf5" --channel "Hadronic" --tag "%s" --ext ""' % (args.tag + "_impute_hut_BDT", my_own_tag, my_own_tag + "_impute_hut"))
    parallel_utils.run('python train.py --input "ttHHadronic_%s_FCNC_features%s.hdf5" --channel "Hadronic" --tag "%s" --ext ""' % (args.tag + "_impute_hct_BDT", my_own_tag, my_own_tag + "_impute_hct")) 
    print ">>> training is finished!"

#--------------------------------------------------#
# Make ZA plots
#--------------------------------------------------#
if args.make_za_plots:
    dir = "depository_symbolicLink"
    func_make_za_plot.make_za_plot(dir, "Leptonic", "hut")
    func_make_za_plot.make_za_plot(dir, "Leptonic", "hct")
    func_make_za_plot.make_za_plot(dir, "Hadronic", "hut")
    func_make_za_plot.make_za_plot(dir, "Hadronic", "hct")
