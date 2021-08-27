import os, sys
import json

from bayes_opt import BayesianOptimization

sys.path.append("../Loopers")
import parallel_utils
import train_core

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--channel", help = "e.g. Hadronic or Leptonic", type=str)
parser.add_argument("--input", help = "input hdf5 file", type=str)
parser.add_argument("--input_inverted", help = "input hdf5 file, with test/train split inverted", type=str)
parser.add_argument("--ext", help = "extension, e.g. '1'", type=str)
parser.add_argument("--tag", help = "tag to identify this training", type=str)
parser.add_argument("-m", "--multi", help = "run a multiclassifier based BDT", action="store_true")
parser.add_argument("-r", "--res", help = "weight signal events by relative mass resolution", action = "store_true")
parser.add_argument("-s", "--sideband", help = "use data sideband for training", action = "store_true")
parser.add_argument("--njets_cut", help = "apply manual cut on njets >= 4", action = "store_true")
parser.add_argument("--optimization_vars", help = "csv list of additional variables to perform N-d Z_A optimization scan in (along with BDT score)", type=str)

args = parser.parse_args()

def auc(n_round, max_depth, eta, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda):
  global idx, log, full_results

  n_round = int(n_round)
  max_depth = int(max_depth)

  kparam = {}
  kparam["max_depth"] = max_depth
  kparam["eta"] = eta
  kparam["subsample"] = subsample
  kparam["colsample_bytree"] = colsample_bytree
  kparam["min_child_weight"] = min_child_weight
  kparam["gamma"] = gamma
  kparam["reg_alpha"] = reg_alpha
  kparam["reg_lambda"] = reg_lambda

  config = {}
  config["n_round"] = n_round
  config["kparam"] = kparam
  config["args"] = args 
  config["skip_tree"] = True

  results_1 = train_core.train_bdt(config, False)
  #results_2 = train_core.train_bdt(config, True)

  auc_train = results_1["auc_train"]
  auc_test  = results_1["auc_test"]
  auc_test_unc = results_1["auc_test_unc"]

  #auc_train = (results_1["auc_train"] + results_2["auc_train"])/2.
  #auc_test = (results_1["auc_test"] + results_2["auc_test"])/2.
  #auc_test_unc = ((results_1["auc_test_unc"]**2 + results_2["auc_test_unc"]**2)**(0.5))/2.

  full_results[idx] = {"config" : config, "results" : {"auc_train" : auc_train, "auc_test" : auc_test, "auc_test_unc" : auc_test_unc}}

  config.pop("args")
  with open(log, "w") as f_out:
    json.dump(full_results, f_out, indent=4, sort_keys=True)

  idx += 1
  return auc_test

idx = 0
log = "bayes_hyperparam_scan_%s_%s.json" % (args.channel, args.tag)
full_results = {}

pbounds = {
        "n_round"               : (75, 75),
        "max_depth"             : (2, 15),
        "eta"                   : (0.001, 0.99),

        "subsample"            : (1.0, 1.0),
        "colsample_bytree"     : (1.0, 1.0),
        "min_child_weight"     : (1, 1),
        "gamma"                : (0, 0),
        "reg_alpha"            : (0, 0),
        "reg_lambda"           : (1, 1),
        #"subsample"             : (0.5, 1.0),
        #"colsample_bytree"      : (0.1, 1.0),
        #"min_child_weight"      : (0, 20),
        #"gamma"                 : (0, 10),
        #"reg_alpha"             : (0, 10),
        #"reg_lambda"            : (0, 10),
}

starting_point = {
        "n_round"               : 75,
        "max_depth"             : 4,
        "eta"                   : 0.2,
        "subsample"             : 1.0,
        "colsample_bytree"      : 1.0,
        "min_child_weight"      : 1,
        "gamma"                 : 0,
        "reg_lambda"            : 1,
        "reg_alpha"             : 0,
}

optimizer = BayesianOptimization(
    f=auc,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.probe(params = starting_point, lazy = True)

optimizer.maximize(
        init_points = 1,
        n_iter = 1000,
        acq = "ei",
        xi = 0.001,
)

for i, res in enumerate(optimizer.res):
    print(("Iteration {}: \n\t{}".format(i, res)))
