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

import tree_utils
import utils
import tmva_utils
import ks_test
import significance_utils

def train_bdt(config, invert=False):
  results = {}

  args = config["args"]
  ### Read features ###
  if not invert:
    f = h5py.File(args.input.replace(".hdf5", "") + ".hdf5", "r")
  else:
    f = h5py.File(args.input_invert.replace(".hdf5", "") + ".hdf5", "r")


  feature_names = utils.load_array(f, 'feature_names')
  training_feature_names = utils.load_array(f, 'training_feature_names')

  print(training_feature_names)

  global_features = utils.load_array(f, 'global')
  global_dnn_features = utils.load_array(f, 'global_dnn')
  label = utils.load_array(f, 'label')
  multi_label = utils.load_array(f, 'multi_label')
  weights = utils.load_array(f, 'weights')
  mass = utils.load_array(f, 'mass')
  njets = utils.load_array(f, 'njets')
  lead_sigmaEtoE = utils.load_array(f, 'lead_sigmaEtoE')
  sublead_sigmaEtoE = utils.load_array(f, 'sublead_sigmaEtoE')
  signal_mass_label = utils.load_array(f, 'signal_mass_label')
  signal_mass_category = utils.load_array(f, 'signal_mass_category')
  tth_2017_reference_mva = utils.load_array(f, 'tth_2017_reference_mva')
  evt = utils.load_array(f, 'evt')
  run = utils.load_array(f, 'run')
  lumi = utils.load_array(f, 'lumi')
  process_id = utils.load_array(f, 'process_id')
  year = utils.load_array(f, 'year')
  #objects = utils.load_array(f, 'objects')
  tth_runII_mva = utils.load_array(f, 'tth_runII_mva')

  if args.sideband:
    global_features = utils.load_array(f, 'global_data_sideband')
    label = utils.load_array(f, 'label_data_sideband')
    multi_label = utils.load_array(f, 'multi_label_data_sideband')
    weights = utils.load_array(f, 'weights_data_sideband')
    mass = utils.load_array(f, 'mass_data_sideband')
    #lead_sigmaEtoE = utils.load_array(f, 'lead_sigmaEtoE_data_sideband')
    #sublead_sigmaEtoE = utils.load_array(f, 'sublead_sigmaEtoE_data_sideband')

  global_features_validation = utils.load_array(f, 'global_validation')
  global_dnn_features_validation = utils.load_array(f, 'global_dnn_validation')
  label_validation = utils.load_array(f, 'label_validation')
  multi_label_validation = utils.load_array(f, 'multi_label_validation')
  weights_validation = utils.load_array(f, 'weights_validation')
  mass_validation = utils.load_array(f, 'mass_validation')
  njets_validation = utils.load_array(f, 'njets_validation')
  signal_mass_label_validation = utils.load_array(f, 'signal_mass_label_validation')
  signal_mass_category_validation = utils.load_array(f, 'signal_mass_category_validation')
  tth_2017_reference_mva_validation = utils.load_array(f, 'tth_2017_reference_mva_validation')
  evt_validation = utils.load_array(f, 'evt_validation')
  run_validation = utils.load_array(f, 'run_validation')
  lumi_validation = utils.load_array(f, 'lumi_validation')
  process_id_validation = utils.load_array(f, 'process_id_validation')
  year_validation = utils.load_array(f, 'year_validation')
  #objects_validation = utils.load_array(f, 'objects_validation')
  tth_runII_mva_validation = utils.load_array(f, 'tth_runII_mva_validation')

  global_features_data = utils.load_array(f, 'global_data')
  global_dnn_features_data = utils.load_array(f, 'global_dnn_data')
  label_data = utils.load_array(f, 'label_data')
  multi_label_data = utils.load_array(f, 'multi_label_data')
  weights_data = utils.load_array(f, 'weights_data')
  mass_data = utils.load_array(f, 'mass_data')
  njets_data = utils.load_array(f, 'njets_data')
  signal_mass_label_data = utils.load_array(f, 'signal_mass_label_data')
  signal_mass_category_data = utils.load_array(f, 'signal_mass_category_data')
  tth_2017_reference_mva_data = utils.load_array(f, 'tth_2017_reference_mva_data')
  evt_data = utils.load_array(f, 'evt_data')
  run_data = utils.load_array(f, 'run_data')
  lumi_data = utils.load_array(f, 'lumi_data')
  process_id_data = utils.load_array(f, 'process_id_data')
  year_data = utils.load_array(f, 'year_data')
  #objects_data = utils.load_array(f, 'objects_data')
  tth_runII_mva_data = utils.load_array(f, 'tth_runII_mva_data')

  global_features_final_fit = utils.load_array(f, 'global_final_fit')
  global_dnn_features_final_fit = utils.load_array(f, 'global_dnn_final_fit')
  label_final_fit = utils.load_array(f, 'label_final_fit')
  multi_label_final_fit = utils.load_array(f, 'multi_label_final_fit')
  weights_final_fit = utils.load_array(f, 'weights_final_fit')
  mass_final_fit = utils.load_array(f, 'mass_final_fit')
  njets_final_fit = utils.load_array(f, 'njets_final_fit')
  signal_mass_label_final_fit = utils.load_array(f, 'signal_mass_label_final_fit')
  signal_mass_category_final_fit = utils.load_array(f, 'signal_mass_category_final_fit')
  tth_2017_reference_mva_final_fit = utils.load_array(f, 'tth_2017_reference_mva_final_fit')
  evt_final_fit = utils.load_array(f, 'evt_final_fit')
  run_final_fit = utils.load_array(f, 'run_final_fit')
  lumi_final_fit = utils.load_array(f, 'lumi_final_fit')
  process_id_final_fit = utils.load_array(f, 'process_id_final_fit')
  year_final_fit = utils.load_array(f, 'year_final_fit')
  #objects_final_fit = utils.load_array(f, 'objects_final_fit')
  tth_runII_mva_final_fit = utils.load_array(f, 'tth_runII_mva_final_fit')

  print global_dnn_features.shape, global_dnn_features_validation.shape, global_dnn_features_data.shape, global_dnn_features_final_fit.shape

  num_multi_class = 3#len(numpy.unique(multi_label, return_index = True))

  train_frac = 1.0 # use this fraction of data for training, use 1-train_frac for testing
  nTrain = int(len(label)*train_frac)

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
  X_final_fit = pandas.DataFrame(data=global_features_final_fit, columns = training_feature_names)

  if args.multi:
    Y_train = y_train_multi 
    Y_test = y_test_multi
  else:
    Y_train = y_train
    Y_test = y_test

  #unique, count =  numpy.unique(multi_label,return_counts=True)
  #weights_train = numpy.multiply(weights_train, numpy.where(multi_label == 0, 1/(count[0]/float(sum(count))), 1) )
  #weights_train = numpy.multiply(weights_train, numpy.where(multi_label == 1, 1/(count[1]/float(sum(count))), 1) )
  #weights_train = numpy.multiply(weights_train, numpy.where(multi_label == 2, 1/(count[2]/float(sum(count))), 1) )
  #weights_train = numpy.multiply(weights_train, numpy.where(multi_label == 3, 1/(count[3]/float(sum(count))), 1) )
  #weights_train = numpy.multiply(weights_train, numpy.where(multi_label == 4, 1/(count[4]/float(sum(count))), 1) )

  sum_neg_weights = utils.sum_of_weights(weights_train, label, 0)
  sum_pos_weights = utils.sum_of_weights(weights_train, label, 1)

  print((sum_pos_weights, sum_neg_weights))

  scale_tth = False
  if scale_tth:
    for i in range(len(weights_train)):
      if multi_label[i] == 1:
        weights_train[i] *= 6.
  weights_train_modified = weights_train

  equal_weights = args.equal_weights 
  if args.multi:
    if not equal_weights:
      for j in range(len(weights_train_modified)):
        if multi_label[j] == 0:
          weights_train_modified[j] *= sum_neg_weights / sum_pos_weights
    else:
      for i in range(num_multi_class):
        sum_class_weights = utils.sum_of_weights(weights_train_modified, multi_label, i)
        print(("Normalizing class %d by %.6f" % (i, sum_class_weights)))
        for j in range(len(weights_train_modified)):
          if multi_label[j] == i:
            weights_train_modified[j] *= 1. / sum_class_weights

  if args.res:
    for i in range(len(weights_train_modified)):
      if label[i] == 1:
	    print((weights_train_modified[i], 1/math.sqrt(lead_sigmaEtoE[i] ** 2 + sublead_sigmaEtoE[i] ** 2)))
	    weights_train_modified[i] *= 1/math.sqrt(lead_sigmaEtoE[i] ** 2 + sublead_sigmaEtoE[i] ** 2) 
	    print((weights_train_modified[i]))

  sum_neg_weights = utils.sum_of_weights(weights_train_modified, label, 0)
  sum_pos_weights = utils.sum_of_weights(weights_train_modified, label, 1)

  print((sum_pos_weights, sum_neg_weights))

  d_train = xgboost.DMatrix(X_train, label = Y_train, weight = weights_train_modified)
  d_test = xgboost.DMatrix(X_test, label = Y_test)
  d_data = xgboost.DMatrix(X_data)
  d_final_fit = xgboost.DMatrix(X_final_fit)

  # Define BDT parameters
  if "kparam" not in list(config.keys()): 
   param = { 
	    'max_depth': 4,
	    'eta': 0.2,
	    'objective': 'binary:logistic',
	    'scale_pos_weight': sum_neg_weights / sum_pos_weights,
	    'subsample': 1.0,
	    'colsample_bytree': 1.0,
	    'nthread' : 12,
	    'min_child_weight' : min(1, (sum_neg_weights) / 100000.), # min_child_weight depends on the absolute value of the weights
	    }
  else:
    param = config["kparam"]

  if args.multi:
    param["num_class"] = num_multi_class
    param["objective"] = "multi:softprob"
    param["scale_pos_weight"] = 1
    param["min_child_weight"] = 0.000001

  print(param)

  if "n_round" not in list(config.keys()): 
    n_round = 300 if args.channel == "Hadronic" else 150
    #n_round = 10
    if "FCNC" in args.input:
      n_round = 150
      if args.multi:
        n_round = 150
      if "SMHiggs" in args.input and args.channel == "Hadronic":
        n_round = 500
  else:
    n_round = config["n_round"]
  evallist = [(d_train, 'train'), (d_test, 'test')]
  progress = {}

  print((param, n_round))

  # train
  bdt = xgboost.train(param, d_train, n_round, evallist, evals_result = progress)	

  bdt.save_model(args.channel + "_" + args.tag + "_" + args.ext + "_bdt.xgb")
  model = bdt.get_dump()

  input_variables = []
  for name in feature_names:
    input_variables.append((name, 'F'))
  tmva_utils.convert_model(model, input_variables = input_variables, output_xml = args.channel + "_" + args.tag + "_" + args.ext + '_bdt.xml')

  # predict
  pred_train = bdt.predict(d_train) 
  pred_test = bdt.predict(d_test) 
  pred_data = bdt.predict(d_data) 
  pred_final_fit = bdt.predict(d_final_fit) 


  if args.reference_mva != "none":
    if ".xgb" in args.reference_mva:
      ref_mva = xgboost.Booster()
      ref_mva.load_model(args.reference_mva)
      pred_ref_train = ref_mva.predict(d_train, output_margin=args.multi)
      pred_ref_test = ref_mva.predict(d_test, output_margin=args.multi)
      pred_ref_data = ref_mva.predict(d_data, output_margin=args.multi)
      pred_ref_final_fit = ref_mva.predict(d_final_fit, output_margin=args.multi)
    elif ".json" in args.reference_mva:
      import dnn_helper
      dnn_features_train = dnn_helper.DNN_Features(name = 'train', global_features = global_dnn_features, objects = objects)
      dnn_features_validation = dnn_helper.DNN_Features(name = 'validation', global_features = global_dnn_features_validation, objects = objects_validation)
      dnn_features_data = dnn_helper.DNN_Features(name = 'data', global_features = global_dnn_features_data, objects = objects_data)
      dnn_features_final_fit = dnn_helper.DNN_Features(name = 'final_fit', global_features = global_dnn_features_final_fit, objects = objects_final_fit)
      with open(args.reference_mva, "r") as f_in:
        metadata = json.load(f_in)
      dnn = dnn_helper.DNN_Helper(features_validation = dnn_features_validation, features_train = dnn_features_train, features_data = dnn_features_data, features_final_fit = dnn_features_final_fit, metadata = metadata, weights_file = "dnn_weights/" + metadata["weights"], train_mode = False)
      dnn.predict(debug=True)
      pred_ref_train = dnn.predictions["train"]
      pred_ref_test = dnn.predictions["validation"]
      pred_ref_data = dnn.predictions["data"]
      pred_ref_final_fit = dnn.predictions["final_fit"]

  print((pred_test.shape))

  #if args.multi:
  #  pred_train = pred_train[:,0] 
  #  pred_test = pred_test[:,0] 
  #  pred_data = pred_data[:,0]
  #  pred_final_fit = pred_final_fit[:,0]

  print((pred_test.shape))

  # analysis
  # ks test

  if args.multi:
    prediction_train = pred_train[:,0]
    prediction_test = pred_test[:,0]
  else:
    prediction_train = pred_train
    prediction_test = pred_test

  d_sig, p_value_sig, d_bkg, p_value_bkg = ks_test.ks_test(prediction_train, prediction_test, y_train, y_test)
  print(("Results of ks-test (d-score) for signal: %.10f and background: %.10f" % (d_sig, d_bkg)))
  print(("Results of ks-test (p-value) for signal: %.10f and background: %.10f" % (p_value_sig, p_value_bkg)))

  # roc curves
  fpr_train, tpr_train, thresh_train = metrics.roc_curve(y_train, prediction_train, pos_label = 1, sample_weight = weights_train)
  fpr_test, tpr_test, thresh_test = metrics.roc_curve(y_test, prediction_test, pos_label = 1, sample_weight = weights_test)
  
  y_train_2016 = ks_test.logical_vector(y_train, year, 2016)
  y_test_2016 = ks_test.logical_vector(y_test, year_validation, 2016)
  prediction_train_2016 = ks_test.logical_vector(prediction_train, year, 2016)
  prediction_test_2016 = ks_test.logical_vector(prediction_test, year_validation, 2016)
  weights_train_2016 = ks_test.logical_vector(weights_train, year, 2016)
  weights_test_2016 = ks_test.logical_vector(weights_test, year_validation, 2016)

  y_train_2017 = ks_test.logical_vector(y_train, year, 2017)
  y_test_2017 = ks_test.logical_vector(y_test, year_validation, 2017)
  prediction_train_2017 = ks_test.logical_vector(prediction_train, year, 2017)
  prediction_test_2017 = ks_test.logical_vector(prediction_test, year_validation, 2017)
  weights_train_2017 = ks_test.logical_vector(weights_train, year, 2017)
  weights_test_2017 = ks_test.logical_vector(weights_test, year_validation, 2017)

  y_train_2018 = ks_test.logical_vector(y_train, year, 2018)
  y_test_2018 = ks_test.logical_vector(y_test, year_validation, 2018)
  prediction_train_2018 = ks_test.logical_vector(prediction_train, year, 2018)
  prediction_test_2018 = ks_test.logical_vector(prediction_test, year_validation, 2018)
  weights_train_2018 = ks_test.logical_vector(weights_train, year, 2018)
  weights_test_2018 = ks_test.logical_vector(weights_test, year_validation, 2018)

  if len(y_train_2016) > 0:
    fpr_train_2016, tpr_train_2016, thresh_train_2016 = metrics.roc_curve(y_train_2016, prediction_train_2016, pos_label = 1, sample_weight = weights_train_2016)
    fpr_test_2016, tpr_test_2016, thresh_test_2016 = metrics.roc_curve(y_test_2016, prediction_test_2016, pos_label = 1, sample_weight = weights_test_2016)
    auc_2016, unc_2016, blah, blah, blah = utils.auc_and_unc(y_test_2016, prediction_test_2016, weights_test_2016, 25)
    print(("Testing  AUC (2016): %.3f +/- %.4f" % (auc_2016, unc_2016)))
    numpy.savez("bdt_roc_2016_%s.npz" % (args.channel + "_" + args.tag), y_train = y_train_2016, y_test = y_test_2016, prediction_train = prediction_train_2016, prediction_test = prediction_test_2016, fpr_train = fpr_train_2016, fpr_test = fpr_test_2016, tpr_train = tpr_train_2016, tpr_test = tpr_test_2016)

  if len(y_train_2017) > 0:
    fpr_train_2017, tpr_train_2017, thresh_train_2017 = metrics.roc_curve(y_train_2017, prediction_train_2017, pos_label = 1, sample_weight = weights_train_2017)
    fpr_test_2017, tpr_test_2017, thresh_test_2017 = metrics.roc_curve(y_test_2017, prediction_test_2017, pos_label = 1, sample_weight = weights_test_2017)
    auc_2017, unc_2017, blah, blah, blah = utils.auc_and_unc(y_test_2017, prediction_test_2017, weights_test_2017, 25)
    print(("Testing  AUC (2017): %.3f +/- %.4f" % (auc_2017, unc_2017)))
    numpy.savez("bdt_roc_2017_%s.npz" % (args.channel + "_" + args.tag), y_train = y_train_2017, y_test = y_test_2017, prediction_train = prediction_train_2017, prediction_test = prediction_test_2017, fpr_train = fpr_train_2017, fpr_test = fpr_test_2017, tpr_train = tpr_train_2017, tpr_test = tpr_test_2017)

  if len(y_train_2018) > 0:
    fpr_train_2018, tpr_train_2018, thresh_train_2018 = metrics.roc_curve(y_train_2018, prediction_train_2018, pos_label = 1, sample_weight = weights_train_2018)
    fpr_test_2018, tpr_test_2018, thresh_test_2018 = metrics.roc_curve(y_test_2018, prediction_test_2018, pos_label = 1, sample_weight = weights_test_2018)
    auc_2018, unc_2018, blah, blah, blah = utils.auc_and_unc(y_test_2018, prediction_test_2018, weights_test_2018, 25)
    print(("Testing  AUC (2018): %.3f +/- %.4f" % (auc_2018, unc_2018)))
    numpy.savez("bdt_roc_2018_%s.npz" % (args.channel + "_" + args.tag), y_train = y_train_2018, y_test = y_test_2018, prediction_train = prediction_train_2018, prediction_test = prediction_test_2018, fpr_train = fpr_train_2018, fpr_test = fpr_test_2018, tpr_train = tpr_train_2018, tpr_test = tpr_test_2018)

  auc_train = metrics.auc(fpr_train, tpr_train, reorder = True)
  auc_test  = metrics.auc(fpr_test , tpr_test , reorder = True)

  auc, unc, blah, blah, blah = utils.auc_and_unc(y_test, prediction_test, weights_test, 25)

  results["auc_train"] = auc_train
  results["auc_test"]  = auc_test
  results["auc_test_unc"] = unc
  if "skip_tree" in list(config.keys()): 
    return results

  print(("Training AUC: %.3f" % auc_train))
  print(("Testing  AUC: %.3f" % auc_test))

  print(("Testing  AUC: %.3f +/- %.4f" % (auc, unc)))
  
  numpy.savez("bdt_roc_%s.npz" % (args.channel + "_" + args.tag), y_train = y_train, y_test = y_test, prediction_train = prediction_train, prediction_test = prediction_test, fpr_train = fpr_train, fpr_test = fpr_test, tpr_train = tpr_train, tpr_test = tpr_test)

  # Write output to TTree
  tree_train_id = numpy.concatenate((numpy.zeros(len(pred_train)), numpy.ones(len(pred_test)), numpy.ones(len(pred_data)), numpy.ones(len(pred_final_fit))))
  tree_sample_id = numpy.concatenate((label, label_validation, label_data, numpy.ones(len(pred_final_fit))))
  tree_mass = numpy.concatenate((mass, mass_validation, mass_data, mass_final_fit))
  tree_weight = numpy.concatenate((weights, weights_validation, weights_data, weights_final_fit))
  tree_signal_mass_label = numpy.concatenate((signal_mass_label, signal_mass_label_validation, signal_mass_label_data, numpy.zeros(len(pred_final_fit))))
  tree_signal_mass_category = numpy.concatenate((signal_mass_category, signal_mass_category_validation, signal_mass_category_data, numpy.zeros(len(pred_final_fit))))
  tree_tth_2017_reference_mva = numpy.concatenate((tth_2017_reference_mva, tth_2017_reference_mva_validation, tth_2017_reference_mva_data, tth_2017_reference_mva_final_fit))
  tree_evt = numpy.concatenate((evt, evt_validation, evt_data, evt_final_fit))
  tree_tth_runII_mva = numpy.concatenate((tth_runII_mva, tth_runII_mva_validation, tth_runII_mva_data, tth_runII_mva_final_fit))
  tree_run = numpy.concatenate((run, run_validation, run_data, run_final_fit))
  tree_lumi = numpy.concatenate((lumi, lumi_validation, lumi_data, lumi_final_fit))
  tree_process_id = numpy.concatenate((process_id, process_id_validation, process_id_data, process_id_final_fit))
  tree_year = numpy.concatenate((year, year_validation, year_data, year_final_fit))
  tree_global_features = numpy.concatenate((global_features, global_features_validation, global_features_data, global_features_final_fit))

  if ".json" in args.reference_mva:
      tree_dnn_features = numpy.concatenate((global_dnn_features, global_dnn_features_validation, global_dnn_features_data, global_dnn_features_final_fit))

  training_feature_names = [training_feature_names for i in range(len(label))]
  training_feature_names_validation = [training_feature_names for i in range(len(label_validation))]
  training_feature_names_data = [training_feature_names for i in range(len(label_data))]
  training_feature_names_final_fit = [training_feature_names for i in range(len(label_final_fit))]
  #tree_training_feature_names = numpy.concatenate((training_feature_names, training_feature_names_validation, training_feature_names_data, training_feature_names_final_fit))


  tree_train_id = tree_train_id.astype(numpy.int64)
  tree_sample_id = tree_sample_id.astype(numpy.int64)
  tree_mass = tree_mass.astype(numpy.float64)
  tree_weight = tree_weight.astype(numpy.float64)
  tree_signal_mass_label = tree_signal_mass_label.astype(numpy.int64)
  tree_signal_mass_category = tree_signal_mass_category.astype(numpy.int64)
  tree_tth_2017_reference_mva = tree_tth_2017_reference_mva.astype(numpy.float64)
  tree_evt = tree_evt.astype(numpy.uint64)
  tree_tth_runII_mva = tree_tth_runII_mva.astype(numpy.float64)
  tree_run = tree_run.astype(numpy.uint64)
  tree_lumi = tree_lumi.astype(numpy.uint64)
  tree_process_id = tree_process_id.astype(numpy.int64)
  tree_year = tree_year.astype(numpy.int64)
  tree_global_features = tree_global_features.astype(numpy.float64)
  if ".json" in args.reference_mva:
      tree_dnn_features = tree_dnn_features.astype(numpy.float64)
#tree_training_feature_names = tree_training_feature_names.astype(numpy.string_)

  dict = {"train_id" : tree_train_id, "sample_id" : tree_sample_id, "mass" : tree_mass, "weight" : tree_weight, "signal_mass_label" : tree_signal_mass_label, "signal_mass_category" : tree_signal_mass_category, "tth_2017_reference_mva" : tree_tth_2017_reference_mva, "process_id" : tree_process_id, "year" : tree_year, "event" : tree_evt, "lumi" : tree_lumi, "run" : tree_run, "global_features" : tree_global_features, "tth_runII_mva" : tree_tth_runII_mva}#, "training_feature_names" : tree_training_feature_names}

  if ".json" in args.reference_mva:
      dict["dnn_global_features"] = tree_dnn_features

  if args.multi:
    tree_bdt_score = []
    for i in range(num_multi_class):
      tree_bdt_score.append(numpy.concatenate((pred_train[:,i], pred_test[:,i], pred_data[:,i], numpy.ones(len(pred_final_fit))))) 
      tree_bdt_score[i] = tree_bdt_score[i].astype(numpy.float64)
      dict["mva_score_%d" % i] = tree_bdt_score[i]

  else:
    tree_bdt_score = numpy.concatenate((pred_train, pred_test, pred_data, pred_final_fit))
    tree_bdt_score = tree_bdt_score.astype(numpy.float64)
    dict["mva_score"] = tree_bdt_score

  if args.reference_mva != "none":
    tree_ref_mva_score = numpy.concatenate((pred_ref_train, pred_ref_test, pred_ref_data, pred_ref_final_fit))
    tree_ref_mva_score = tree_ref_mva_score.astype(numpy.float64)
    dict[args.reference_mva_name] = tree_ref_mva_score

  tree_utils.numpy_to_tree(dict, "ttH%s_%s_FinalFitTree.root" % (args.channel, args.tag))

  ### Make diagnostic plots ###
  import matplotlib.pyplot as plt

  # variable importance #
  fig = plt.figure()
  xgboost.plot_importance(bdt)
  plt.tight_layout()
  plt.savefig('feature_importance_' + args.channel + '.pdf')

  # make ROC curve #
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.yaxis.set_ticks_position('both')
  ax.grid(True)
  plt.grid(color='black', linestyle='--', linewidth = 0.1, which = 'both')

  plt.plot(fpr_train, tpr_train, color='red', label='Training Set', lw = 3)
  plt.plot(fpr_test, tpr_test, color='green', label='Testing Set', lw = 3)

  plt.xscale('log')

  plt.xlim([0.005, 1.0])
  plt.ylim([0.3, 1.05])
  plt.xlabel('False Positive Rate (background efficiency)')
  plt.ylabel('True Positive Rate (signal efficiency)')
  plt.legend(loc='lower right')
  plt.savefig('roc' + args.channel + '.pdf', bbox_inches='tight')

  estimate_za = True 
  use_tth_runII_mva = False
  use_tth_2017_mva = False
  if estimate_za:
    n_quantiles = 30

    if args.multi:
      signal_mva_scores = {}
      bkg_mva_scores = {}
      data_mva_scores = {}
      for i in range(0,num_multi_class-1): # optimize with each of the bkg probabilities (the signal probability is redundant, i.e. sum of probabilities = 1) 
        reverse = 1 if i == 0 else -1
        signal_mva_scores["bdt_score_%d" % i] = reverse*ks_test.logical_vector(pred_test[:,i], y_test, 1) # factor of -1 so that we cut *below* certain values, as these are background probabilities, not signal
        bkg_mva_scores["bdt_score_%d" % i] = reverse*ks_test.logical_vector(pred_test[:,i], y_test, 0)
        data_mva_scores["bdt_score_%d" % i] = reverse*pred_data[:,i]
    elif use_tth_runII_mva:
      print "Using RunII MVA from flashgg"
      signal_mva_scores = {"bdt_score" : ks_test.logical_vector(tth_runII_mva_validation, y_test, 1)}
      bkg_mva_scores = {"bdt_score" : ks_test.logical_vector(tth_runII_mva_validation, y_test, 0)}
      data_mva_scores = {"bdt_score" : tth_runII_mva_data}
    elif use_tth_2017_mva:
      print "Using 2017 ttH PAS MVA"
      signal_mva_scores = {"bdt_score" : ks_test.logical_vector(tth_2017_reference_mva_validation, y_test, 1)}
      bkg_mva_scores = {"bdt_score" : ks_test.logical_vector(tth_2017_reference_mva_validation, y_test, 0)}
      data_mva_scores = {"bdt_score" : tth_2017_reference_mva_data}
    else:
      print "Using the MVA we just trained"
      signal_mva_scores = {"bdt_score" : ks_test.logical_vector(pred_test, y_test, 1)}
      bkg_mva_scores = {"bdt_score" : ks_test.logical_vector(pred_test, y_test, 0)}
      data_mva_scores = {"bdt_score" : pred_data}


    signal_mass = ks_test.logical_vector(mass_validation, y_test, 1)
    bkg_mass = ks_test.logical_vector(mass_validation, y_test, 0)

    signal_njets = ks_test.logical_vector(njets_validation, y_test, 1)
    bkg_njets = ks_test.logical_vector(njets_validation, y_test, 0)

    signal_weights = ks_test.logical_vector(weights_validation, y_test, 1)
    #if args.channel == "Leptonic" and "FCNC" in args.input:
    #  signal_weights *= 1./1.53 # to account for bug in MC sample where W->lv decays don't include taus
    bkg_weights = ks_test.logical_vector(weights_validation, y_test, 0)
   
    bkg_process_id = ks_test.logical_vector(process_id_validation, y_test, 0)

    optimization_vars = args.optimization_vars.split(",") if args.optimization_vars else []
    for var in optimization_vars: 
      signal_mva_scores[var] = ks_test.logical_vector(utils.load_array(f, var + '_validation'), y_test, 1)
      bkg_mva_scores[var]    = ks_test.logical_vector(utils.load_array(f, var + '_validation'), y_test, 0)
      data_mva_scores[var]   = utils.load_array(f, var + '_data')

    signal_events = { "mass" : signal_mass, "weights" : signal_weights, "mva_score" : signal_mva_scores} 
    bkg_events = { "mass" : bkg_mass, "weights" : bkg_weights, "mva_score" : bkg_mva_scores, "process_id" : bkg_process_id} 
    data_events = { "mass" : mass_data, "weights" : weights_data, "mva_score" : data_mva_scores, "process_id" : numpy.ones_like(mass_data)} 

    # Trim these dictionaries down
    #for evts_dict in [signal_events, bkg_events, data_events]:
    #  good_indices = [index for index, value in enumerate(evts_dict["mass"]) if value < 180.]
    #  print float(len(good_indices))/float(len(evts_dict["mass"]))
    #  for key in evts_dict.iterkeys():
    #    full_array = evts_dict[key]
    #    trimmed_array = [full_array[i] for i in good_indices] 
    #    evts_dict[key] = trimmed_array

    mass_shift = not("FCNC" in args.input) # if we're using FCNC as signal, all Higgs mass points should be 125
                                           # but, if we're using ttH as signal, we use M127 sample for testing, so need to shift for proper comparison with other M125 samples
    za, za_unc, s, b, sigma_eff = significance_utils.za_scores(n_quantiles, signal_events, bkg_events, False, {}, mass_shift)
    za_data, za_unc_data, s_data, b_data, sigma_eff_data = significance_utils.za_scores(n_quantiles, signal_events, data_events, True, bkg_events, mass_shift)
    za = numpy.asarray(za)

    max_za = numpy.max(za)
    max_za_unc = za_unc[numpy.argmax(za)]
    print((max_za, max_za_unc))

    numpy.savez("za_%s.npz" % (args.channel + "_" + args.ext + "_" + args.tag), za = za, za_unc = za_unc, signal = s, bkg = b, sigma_eff = sigma_eff, za_data = za_data, za_unc_data = za_unc_data, signal_data = s_data, bkg_data = b_data, sigma_eff_data = sigma_eff_data)
    numpy.savez("sigma_eff.npz", sigma_eff = sigma_eff, n_sig = s)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(s, za, label='MC', color = 'red')
    ax1.fill_between(s, numpy.asarray(za) - numpy.asarray(za_unc), numpy.asarray(za) + numpy.asarray(za_unc), color = 'red', alpha = 0.25)
    ax1.plot(s_data, za_data, label='Data', color = 'black')
    ax1.fill_between(s_data, numpy.asarray(za_data) - numpy.asarray(za_unc_data), numpy.asarray(za_data) + numpy.asarray(za_unc_data), color = 'black', alpha = 0.25)

    plt.xlabel('# Signal Events')
    ax1.set_ylabel('Significance (Z_A)')

    plt.ylim([0.0, 3.0])
    l, r = plt.xlim()
    plt.xlim([1.0, r])

    ax1.legend(loc='upper right')
    plt.savefig('za_curve.pdf')

  return results
