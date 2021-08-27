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

import utils
import tmva_utils
import ks_test
import significance_utils

def train_bdt(config):
  # Trains BDT with given hyperparams and returns max Z_A (as calculated on bkg MC), requiring at least 4 signal events
  if config["invert_test_and_train"]:
    config["input_file"] = config["input_file_2"]
  else:
    config["input_file"] = config["input_file_1"]
  f = h5py.File(config["input_file"], "r")

  feature_names = utils.load_array(f, 'feature_names')
  training_feature_names = utils.load_array(f, 'training_feature_names')

  print(("Training with the following features: ", training_feature_names))

  #if config["invert_test_and_train"]:
    #print "Inverting test and train splits"
    #if config["sideband"]:
    #  print "Not yet implemented how to handle inverting the test/train set when training on data sidebands, exiting"
    #  return -1

    #global_features = utils.load_array(f, 'global_validation')
    #label = utils.load_array(f, 'label_validation')
    #multi_label = utils.load_array(f, 'multi_label_validation')
    #weights = utils.load_array(f, 'weights_validation')
    #mass = utils.load_array(f, 'mass_validation')
 
    #global_features_validation = utils.load_array(f, 'global')
    #label_validation = utils.load_array(f, 'label')
    #multi_label_validation = utils.load_array(f, 'multi_label')
    #weights_validation = utils.load_array(f, 'weights')
    #mass_validation = utils.load_array(f, 'mass')

  #else:
  global_features = utils.load_array(f, 'global')
  label = utils.load_array(f, 'label')
  multi_label = utils.load_array(f, 'multi_label')
  weights = utils.load_array(f, 'weights')
  mass = utils.load_array(f, 'mass')

  global_features_validation = utils.load_array(f, 'global_validation')
  label_validation = utils.load_array(f, 'label_validation')
  multi_label_validation = utils.load_array(f, 'multi_label_validation')
  weights_validation = utils.load_array(f, 'weights_validation')
  mass_validation = utils.load_array(f, 'mass_validation')

  if config["sideband"]:
    global_features = utils.load_array(f, 'global_data_sideband')
    label = utils.load_array(f, 'label_data_sideband')
    multi_label = utils.load_array(f, 'multi_label_data_sideband')
    weights = utils.load_array(f, 'weights_data_sideband')
    mass = utils.load_array(f, 'mass_data_sideband')

  global_features_data = utils.load_array(f, 'global_data')
  label_data = utils.load_array(f, 'label_data')
  multi_label_data = utils.load_array(f, 'multi_label_data')
  weights_data = utils.load_array(f, 'weights_data')
  mass_data = utils.load_array(f, 'mass_data')

  print((global_features.shape))
  print((label.shape))
  print((weights.shape))

  print((global_features_validation.shape))
  print((label_validation.shape))
  print((weights_validation.shape))

  print((global_features_data.shape))
  print((label_data.shape))
  print((weights_data.shape))

  x_train, y_train, y_train_multi, weights_train = global_features, label, multi_label, weights
  x_test, y_test, y_test_multi, weights_test  = global_features_validation, label_validation, multi_label_validation, weights_validation

  X_train = pandas.DataFrame(data=x_train, columns = training_feature_names)
  X_test = pandas.DataFrame(data=x_test, columns = training_feature_names)
  X_data = pandas.DataFrame(data=global_features_data, columns = training_feature_names)

  if config["multiclassifier"]:
    Y_train = y_train_multi
    Y_test = y_test_multi
  else:
    Y_train = y_train
    Y_test = y_test

  sum_neg_weights = utils.sum_of_weights_v2(weights_train, label, 0)
  sum_pos_weights = utils.sum_of_weights_v2(weights_train, label, 1)

  print((sum_pos_weights, sum_neg_weights))

  d_train = xgboost.DMatrix(X_train, label = Y_train, weight = weights_train)
  d_test = xgboost.DMatrix(X_test, label = Y_test)
  d_data = xgboost.DMatrix(X_data)

  param = {
	'max_depth': config["max_depth"],
        'eta': config["eta"],
        'subsample': config["subsample"],
        'colsample_bytree': config["colsample_bytree"],
        'min_child_weight' : config["min_child_weight"],
	'gamma' : config["gamma"],
	'reg_alpha' : config["reg_alpha"],
	'reg_lambda' : config["reg_lambda"],

	'scale_pos_weight': sum_neg_weights / sum_pos_weights,    
	'objective': 'binary:logistic',
	'nthread' : 16,
  } 

  if config["multiclassifier"]:
    param["num_class"] = config["n_class"]
    param["objective"] = "multi:softprob"
    param["scale_pos_weight"] = 1

  evallist = [(d_train, 'train'), (d_test, 'test')]
  progress = {}

  n_round = config["n_round"]
  print((param, n_round))

  # train
  bdt = xgboost.train(param, d_train, n_round, evallist, evals_result = progress)

  bdt.save_model(config["tag"] + "_bdt.xgb")
  model = bdt.get_dump()

  input_variables = []
  for name in feature_names:
    input_variables.append((name, 'F'))
  #tmva_utils.convert_model(model, input_variables = input_variables, output_xml = config["tag"] + '_bdt.xml')
  
  # predict
  pred_train = bdt.predict(d_train, output_margin=config["multiclassifier"])
  pred_test = bdt.predict(d_test, output_margin=config["multiclassifier"])
  pred_data = bdt.predict(d_data, output_margin=config["multiclassifier"])

  fpr_train, tpr_train, thresh_train = metrics.roc_curve(y_train, pred_train, pos_label = 1, sample_weight = weights_train)
  fpr_test, tpr_test, thresh_test = metrics.roc_curve(y_test, pred_test, pos_label = 1, sample_weight = weights_test) 
  
  auc_train, auc_train_unc = utils.auc_and_unc(y_train, pred_train, weights_train, 100)
  auc_test, auc_test_unc = utils.auc_and_unc(y_test, pred_test, weights_test, 100)
  
  #auc_train = metrics.auc(fpr_train, tpr_train, reorder = True)
  #auc_test  = metrics.auc(fpr_test , tpr_test , reorder = True)

  print(("Training AUC: %.3f" % auc_train))
  print(("Testing  AUC: %.3f" % auc_test)) 

  # estimate z_a w/at least 4 signal events
  n_quantiles = 25
  signal_mva_scores = {"bdt_score" : ks_test.logical_vector(pred_test, y_test, 1)}
  bkg_mva_scores = {"bdt_score" : ks_test.logical_vector(pred_test, y_test, 0)}
  data_mva_scores = {"bdt_score" : pred_data}

  signal_mass = ks_test.logical_vector(mass_validation, y_test, 1)
  bkg_mass = ks_test.logical_vector(mass_validation, y_test, 0)

  signal_weights = ks_test.logical_vector(weights_validation, y_test, 1)
  bkg_weights = ks_test.logical_vector(weights_validation, y_test, 0)

  optimization_vars = config["optimization_vars"].split(",") if config["optimization_vars"] else []
  for var in optimization_vars:
    signal_mva_scores[var] = ks_test.logical_vector(utils.load_array(f, var + '_validation'), y_test, 1)
    bkg_mva_scores[var]    = ks_test.logical_vector(utils.load_array(f, var + '_validation'), y_test, 0)
    data_mva_scores[var]   = utils.load_array(f, var + '_data')

  signal_events = { "mass" : signal_mass, "weights" : signal_weights, "mva_score" : signal_mva_scores }
  bkg_events = { "mass" : bkg_mass, "weights" : bkg_weights, "mva_score" : bkg_mva_scores }
  data_events = { "mass" : mass_data, "weights" : weights_data, "mva_score" : data_mva_scores }

  za, za_unc, s, b, sigma_eff = significance_utils.za_scores(n_quantiles, signal_events, bkg_events, False)
  za_data, za_unc_data, s_data, b_data, sigma_eff_data = significance_utils.za_scores(n_quantiles, signal_events, data_events, True)
  za = numpy.asarray(za)
  za_data = numpy.asarray(za_data)

  if numpy.all(za == 0) or numpy.all(za_data == 0):
    return 0.0, 0.0, 0.0, 0.0

  max_za_mc = numpy.max(za[numpy.where(numpy.asarray(s) >= 4.)])
  max_za_data = numpy.max(za_data[numpy.where(numpy.asarray(s_data) >= 4.)])

  max_za_mc, max_za_mc_idx = utils.find_nearest(za, max_za_mc)
  max_za_data, max_za_data_idx = utils.find_nearest(za_data, max_za_data)

  max_za_mc_unc   = za_unc[max_za_mc_idx]
  max_za_data_unc = za_unc_data[max_za_data_idx]

  print(("Max Z_A on MC:   %.4f +/- %.4f" % (max_za_mc,   max_za_mc_unc)))
  print(("Max Z_A on data: %.4f +/- %.4f" % (max_za_data, max_za_data_unc)))

  return max_za_mc, max_za_mc_unc, max_za_data, max_za_data_unc, auc_train, auc_train_unc, auc_test, auc_test_unc
