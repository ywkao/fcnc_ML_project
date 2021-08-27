import os, sys
import json
import numpy

import itertools
import random

sys.path.append("../Loopers")
import parallel_utils
import train_dnn_core

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input hdf5 file", type=str)
parser.add_argument("--tag", help = "save name for weights file", type=str)
parser.add_argument("--channel", help = "Hadronic or Leptonic", type=str, default="Hadronic")
parser.add_argument("--preprocess_scheme", help = "json used for preprocessing features", type=str)
parser.add_argument("--no_global", help = "don't use global features", action="store_true")
parser.add_argument("--no_lstm", help = "don't use object features (lstm)", action="store_true")
parser.add_argument("--load", help = "give weights file to use as starting point", type=str)
parser.add_argument("--no_bootstrap", help = "don't use bootstrapping to estimate unc. in AUC (to save time during hyperparameter opt)", action="store_true")
parser.add_argument("--absolute_weights", help = "set negative weights to positive in *training* only (to improve stats)", action="store_true")
parser.add_argument("--n_points", help = "how many points to probe", type=str, default="200")
args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Optimization_Helper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.input  = kwargs.get('input')
        self.channel = kwargs.get('channel')
        self.tag    = kwargs.get('tag')
        self.verbose = kwargs.get('verbose')
        self.n_points = int(kwargs.get('n_points', 3))

        self.pbounds = kwargs.get('pbounds')
        self.function = kwargs.get('function') # function to evaluate

        self.args = kwargs.get('args')

        self.parameter_ranges = {}
        self.parameter_combinations_unshuffled = []
        self.parameter_combinations = []
        self.results = {}
        self.idx = 0

        self.log_file = "optimizer_log_%s_%s.json" % (self.channel, self.tag)
        self.default_config = {"n_nodes_dense_1" : 300, "n_nodes_dense_2" : 200, "n_dense_1" : 1, "n_dense_2" : 4, "n_nodes_lstm" : 100, "n_lstm" : 3, "maxnorm" : 3, "dropout_rate" : 0.25, "learning_rate" : 0.001, "start_batch" : 512, "batch_norm" : True, "batch_momentum" : 0.99, "layer_norm" : False, "epsilon" : 1e-08}


    def dnn_auc(self, config, i):
        if self.verbose:
            print("Now probing this point: ", config)
        trained_dnn = train_dnn_core.train(self.args, config)
        self.results[i] = {"config" : config, "target" : trained_dnn.auc["validation"][-1], "results" : {"auc_train" : trained_dnn.auc["train"], "auc_train_unc" : trained_dnn.auc_unc["train"], "auc_test" : trained_dnn.auc["validation"], "auc_test_unc" : trained_dnn.auc_unc["validation"]}}
        with open(log, "w") as f_out:
            json.dump(self.results, f_out, indent=4)


    def define_arrays(self):
        for hyperparameter, settings in pbounds.items():
            points = [] 
            if settings["distribution"] == "log":
                points = numpy.logspace(settings["lower"], settings["upper"], settings["n_sample"])
            elif "uniform" in settings["distribution"]:
                points = numpy.linspace(settings["lower"], settings["upper"], settings["n_sample"])
                if "discrete" in settings["distribution"]:
                    points = [int(point) for point in points]
            else:
                points = settings["distribution"]

            self.parameter_ranges[hyperparameter] = points
        raw_parameters = list(itertools.product(*[tuple(self.parameter_ranges[key]) for key in self.parameter_ranges.keys()]))
        for params in raw_parameters:
            point = dict(self.default_config)
            i = 0
            for key in self.parameter_ranges.keys():
                point[key] = params[i]
                i += 1
            self.parameter_combinations_unshuffled.append(point)
        self.parameter_combinations = random.sample(self.parameter_combinations_unshuffled, len(self.parameter_combinations_unshuffled)) 
        self.n_points = min(self.n_points, len(self.parameter_combinations))
        if self.verbose:
            print("Here are all %d hyperparameter combinations that will be tested:" % self.n_points) 
            for i in range(self.n_points):
                print(self.parameter_combinations[i])

    def optimize(self):
        self.define_arrays() 
        for i in range(self.n_points):
            self.dnn_auc(self.parameter_combinations[i], i)



pbounds = {
    "learning_rate"     : { "upper" : -1.5,     "lower" : -3.5,     "distribution" : "log",                 "n_sample" : 10 },
    "n_lstm"            : { "upper" : 3,        "lower" : 1,        "distribution" : "discrete_uniform",    "n_sample" : 3},
    "n_nodes_lstm"      : { "upper" : 250,      "lower" : 25,       "distribution" : "discrete_uniform",    "n_sample" : 5},
    "dropout_rate"      : { "upper" : 0.5,      "lower" : 0.0,      "distribution" : "uniform",             "n_sample" : 5},
    "n_nodes_dense_1"   : { "upper" : 500,      "lower" : 100,      "distribution" : "discrete_uniform",    "n_sample" : 3},
    "n_nodes_dense_2"   : { "upper" : 300,      "lower" : 50,       "distribution" : "discrete_uniform",    "n_sample" : 3},
    #"batch_momentum"    : { "upper" : 0.999,    "lower" : 0.5,      "distribution" : [0.5, 0.9, 0.99, 0.999]},
}

optimizer = Optimization_Helper(input = args.input, channel = args.channel, tag = args.tag, pbounds = pbounds, args = args, verbose = True, n_points = int(args.n_points))
optimizer.optimize()
