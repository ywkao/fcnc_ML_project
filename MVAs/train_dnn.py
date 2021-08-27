import train_dnn_core
import os
#import tensorflow as tf
#print((tf.__version__))
#if tf.__version__ == "1.14.0":
#    print("Disabling new tf behaviors, because Sam wrote this DNN in tf 1.3.0")
#    import tensorflow.compat.v1 as tf
#    tf.disable_v2_behavior()
#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

#print(session)
#import keras
import numpy
import sys
from sklearn import metrics
import h5py
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dnn_helper
#import significance_utils
import utils
import ks_test

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input hdf5 file", type=str)
parser.add_argument("--tag", help = "save name for weights file", type=str)
parser.add_argument("--channel", help = "Hadronic or Leptonic", type=str, default="Hadronic")
parser.add_argument("--no_global", help = "don't use global features", action="store_true")
parser.add_argument("--no_lstm", help = "don't use object features (lstm)", action="store_true")
parser.add_argument("--load", help = "give weights file to use as starting point", type=str)
parser.add_argument("--no_bootstrap", help = "don't use bootstrapping to estimate unc. in AUC (to save time during hyperparameter opt)", action="store_true")
parser.add_argument("--absolute_weights", help = "set negative weights to positive in *training* only (to improve stats)", action="store_true")
parser.add_argument("--preprocess_scheme", help = "json file used for preprocessing dnn. for bookkeeping purposes", type=str)
args = parser.parse_args()


train_dnn_core.train(args, {})
