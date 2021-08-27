import sys, os
from za_label_code import code_to_label
sys.path.append("../")
import parallel_utils

def make_labels(methods, channel):
    labels = ''
    for code in methods:
        if len(labels) != 0:
            labels += ', '
        labels += code_to_label(code, channel)
    return labels

def make_files(methods, pre, suf):
    files = ''
    for code in methods:
        if len(files) != 0:
            files += ', '
        single_file = '%s_17_%s' % (pre, suf) if code == 8 else '%s_0%s_%s' % (pre, str(code), suf) 
        files += single_file
    return files

def make_za_plot(dir, channel, coupling):
  pre = "za_" + channel + "__"
  suf = coupling + ".npz" if channel == "Leptonic" else "impute_" + coupling + ".npz"

  methods = [0, 2, 3, 5] # chi2 vs ann
  #methods = [0, 1, 3, 6] # top tagger vs ann
  #methods = [0, 1, 2, 4] # top tagger vs chi2
  #methods = [0, 1, 2, 3, 7] # three methods
  #methods = [0, 7, 8] # three methods
  labels = make_labels(methods, channel)
  files = make_files(methods, pre, suf)

  #parallel_utils.run('python make_za_comparisons.py --inputs "%s" --labels "%s" --tag "%s" --ylim "0.0, 3.0" --colors black blue red' % (files, labels, channel + "_" + coupling))
  parallel_utils.run('python make_za_comparisons.py --inputs "%s" --labels "%s" --tag "%s" --ylim "0.0, 3.0" --colors black blue green red orange purple' % (files, labels, channel + "_" + coupling))

def make_data_mc_plots(input_file, plot_type, plot_labels, signals, backgrounds):
    parallel_utils.run('python plot_wrapper.py --input_file "../%s" --plot_type "%s" --plot_labels "%s" --signals "%s" --backgrounds "%s"' % (input_file, plot_type, plot_labels, signals, backgrounds))

# Reminder of za labels:
# 00: default
# 01: top tagger
# 02: Minimum chi-2 / quad. eq.
# 03: FCNC reco ANN
# 04: 01 + 02
# 05: 02 + 03
# 06: 03 + 01
# 07: 01 + 02 + 03
# 17: 07 + cvsl and cvsb
