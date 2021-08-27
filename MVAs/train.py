import os, sys
import numpy
import h5py
import xgboost
import pandas
import json
import math

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics

import ROOT
import root_numpy

import utils
import tmva_utils
import ks_test
import significance_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--channel", help = "e.g. Hadronic or Leptonic", type=str)
parser.add_argument("--input", help = "input hdf5 file", type=str)
parser.add_argument("--ext", help = "extension, e.g. '1'", type=str)
parser.add_argument("--tag", help = "tag to identify this training", type=str)
parser.add_argument("--reference_mva", help = "reference mva to save in final fit tree", type=str, default="none")
parser.add_argument("--reference_mva_name", help = "name of the reference mva", type=str, default="reference_mva")
parser.add_argument("-m", "--multi", help = "run a multiclassifier based BDT", action="store_true")
parser.add_argument("-r", "--res", help = "weight signal events by relative mass resolution", action = "store_true")
parser.add_argument("-s", "--sideband", help = "use data sideband for training", action = "store_true")
parser.add_argument("--optimization_vars", help = "csv list of additional variables to perform N-d Z_A optimization scan in (along with BDT score)", type=str)
parser.add_argument("--equal_weights", help = "set all class weights equal for multiclassification", action="store_true")
args = parser.parse_args()

print "args.input               : %s" % args.input

config = {}
config["args"] = args

import train_core

train_core.train_bdt(config)
