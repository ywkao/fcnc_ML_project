import dnn_model
import numpy
from sklearn import metrics
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import keras

class DNN_Features:
  def __init__(self, **kwargs):
    self.name = kwargs.get('name', 'validation')
    self.no_prep = kwargs.get('no_prep', True)
    if not self.no_prep:
      self.global_features = numpy.transpose(numpy.array(kwargs.get('global_features')))
      self.objects = utils.pad_array(kwargs.get('objects'))
    else:
      self.global_features = numpy.array(kwargs.get('global_features'))
      self.objects = numpy.array(kwargs.get('objects'))

    if 'leptons' in kwargs:
      if not kwargs.get('no_prep'):
        self.leptons = utils.pad_array(kwargs.get('leptons', []))
      else:
        self.leptons = numpy.array(kwargs.get('leptons'))
      self.features = [self.global_features, self.objects, self.leptons]
      self.channel = "Leptonic"
    else:
      self.features = [self.global_features, self.objects]
      self.channel = "Hadronic"
 
    self.label = numpy.array(kwargs.get('label', []))
    self.weights = numpy.array(kwargs.get('weights', []))

    self.references = kwargs.get('references', {}) # dictionary of reference BDT/DNN values for comparisons

class DNN_Helper:
  def __init__(self, **kwargs):
    self.kwargs = kwargs


    self.metadata = kwargs.get('metadata')
    self.config = self.metadata["config"]

    self.features_validation = kwargs.get('features_validation', [])
    self.features_train = kwargs.get('features_train', [])
    self.features_data = kwargs.get('features_data', [])
    self.features_final_fit = kwargs.get('features_final_fit', "none")   
    self.mass_data = kwargs.get('mass_data', [])
    self.evt_data = kwargs.get('evt_data', [])
    self.run_data = kwargs.get('run_data', [])
    self.lumi_data = kwargs.get('lumi_data', [])
 
    self.tag = kwargs.get('tag', '')

    self.batch_size = kwargs.get('batch_size', 10000)
    self.batch_size_train = self.config["start_batch"] 

    self.channel = self.features_validation.channel 

    self.weights_file = kwargs.get('weights_file', '')
    self.tag = kwargs.get('tag', '')

    self.n_bootstrap = kwargs.get('n_bootstrap', 0)
    self.max_epochs = kwargs.get('max_epochs', 25)

    self.curriculum_learn = kwargs.get('curriculum_learn', False)

    if 'tag' in kwargs:
      self.save_name = "dnn_weights/" + self.tag + "_weights_{epoch:02d}.hdf5"
      self.callbacks = [keras.callbacks.ModelCheckpoint(self.save_name)]
    else:
      self.callbacks = []

    self.train_mode = kwargs.get('train_mode', True)

    if self.train_mode:
        self.metadata["weights"] = self.tag + "_weights.hdf5"
        if self.metadata["preprocess_scheme"] != "none":
          with open(self.metadata["preprocess_scheme"], "r") as f_in:
            self.metadata["preprocess_scheme"] = json.load(f_in)
        with open("dnn_weights/metadata_" + self.tag + ".json", "w") as f_out:
          json.dump(self.metadata, f_out, indent=4, sort_keys=True)

    self.n_objects = len(self.features_validation.objects[0])
    self.n_object_features = len(self.features_validation.objects[0][0])
    self.n_global_features = len(self.features_validation.global_features[0])

    self.model = dnn_model.tth_learner(self.n_objects, self.n_object_features, self.n_global_features, self.config)

    #if self.channel == "Hadronic":
    #  self.model = dnn_model.baseline_v1(self.n_objects, self.n_object_features, self.n_global_features, False, False, 100)

    #elif self.channel == "Leptonic":
    #  self.n_leptons = len(self.features_validation.leptons[0])
    #  self.n_lepton_features = len(self.features_validation.leptons[0][0])
    #  self.model = dnn_model.baseline_leptonic_v1(self.n_objects, self.n_object_features, self.n_leptons, self.n_lepton_features, self.n_global_features, False, False, 100)

    if self.weights_file:
      self.model.load_weights(self.weights_file)

    self.n_epochs = 0
    self.predictions = { "train" : [], "validation" : [], "data" : [], "final_fit" : []} # store predictions for most recent epoch
    self.auc = { "train" : [], "validation" : [] } # store auc and unc for all epochs
    self.auc_unc = { "train" : [], "validation" : [] }
    self.tpr = { "train" : [], "validation" : [] } # store tpr, fpr for all epochs
    self.fpr = { "train" : [], "validation" : [] } 

    self.prepped = False

  #def oversample(self):
  #  n_sig = len(self.features_train.label[numpy.where(self.features_train.label == 1]))
  #  n_bkg = len(self.features_train.label[numpy.where(self.features_train.label == 1]))

  #  if n_sig > n_bkg:
  #    oversample_indices = numpy.random.randint(0, n_bkg, n_sig)
  #    self.features_train.label = utils.oversample(self.features_train.label, oversample_indices)
  #    self.features_train.weights = utils.oversample(self.features_train.weights, oversample_indices)
  #    self.features_train.global_features = utils.oversample(self.features_train.global_features, oversample_indices)
  #    self.features_train.objects = utils.oversample(self.features_train.objects, oversample_indices)

  def predict(self, debug=False):
    #for i in range(len(self.features_final_fit.global_features)):
    #    print i
    #    for j in range(len(self.features_final_fit.global_features[i])):
    #        print "Global features %d: %.6f" % (j, self.features_final_fit.global_features[i][j])
    #    for j in range(len(self.features_final_fit.objects[i])):
    #        for k in range(len(self.features_final_fit.objects[i][j])):
    #            print "Object feature %d, %d: %.6f" % (j, k, self.features_final_fit.objects[i][j][k]) 
    self.predictions["train"] = self.model.predict(self.features_train.features, self.batch_size).flatten()
    self.predictions["validation"] = self.model.predict(self.features_validation.features, self.batch_size).flatten()
    self.predictions["data"] = self.model.predict(self.features_data.features, self.batch_size).flatten()
    if self.features_final_fit == "none":
      self.predictions["final_fit"] = []
    else:
      self.predictions["final_fit"] = self.model.predict(self.features_final_fit.features, self.batch_size).flatten()
      for i in range(len(self.features_final_fit.global_features)):
          print "DNN Score %d: %.6f" % (i, self.predictions["final_fit"][i])
    return [self.predictions["train"], self.predictions["validation"], self.predictions["data"], self.predictions["final_fit"]]

  def train(self, n_epochs, n_batch):
    if not self.prepped:
      sum_neg_weights = utils.sum_of_weights_v2(self.features_train.weights, self.features_train.label, 0)
      sum_pos_weights = utils.sum_of_weights_v2(self.features_train.weights, self.features_train.label, 1)
      print(("Sum of weights before scaling: ", sum_pos_weights, sum_neg_weights))

      self.features_train.weights[numpy.where(self.features_train.label == 1)] *= sum_neg_weights / sum_pos_weights 
      #self.features_train.weights[numpy.where(self.features_train.label == 1)] *= 10
      self.prepped = True

      sum_neg_weights = utils.sum_of_weights_v2(self.features_train.weights, self.features_train.label, 0)
      sum_pos_weights = utils.sum_of_weights_v2(self.features_train.weights, self.features_train.label, 1)
      print(("Sum of weights after scaling: ", sum_pos_weights, sum_neg_weights))
      print(("Sum of weights in validation set ", utils.sum_of_weights_v2(self.features_validation.weights, self.features_validation.label, 1), utils.sum_of_weights_v2(self.features_validation.weights, self.features_validation.label, 0)))

    for i in range(n_epochs):
      self.model.fit(self.features_train.features, self.features_train.label, epochs = 1, batch_size = self.batch_size_train, sample_weight = self.features_train.weights, callbacks = self.callbacks)
      self.n_epochs += 1
      self.predict()

      fpr_train, tpr_train, thresh_train = metrics.roc_curve(self.features_train.label, self.predictions["train"], pos_label = 1, sample_weight = self.features_train.weights)
      fpr_validation, tpr_validation, thresh_validation = metrics.roc_curve(self.features_validation.label, self.predictions["validation"], pos_label = 1, sample_weight = self.features_validation.weights)

      auc, auc_unc, blah, blah, blah = utils.auc_and_unc(self.features_validation.label, self.predictions["validation"], self.features_validation.weights, self.n_bootstrap)
      auc_train, auc_unc_train, blah, blah, blah = utils.auc_and_unc(self.features_train.label, self.predictions["train"], self.features_train.weights, self.n_bootstrap)

      print(("Test   AUC: %.4f +/- %.4f" % (auc, auc_unc)))
      print(("Train  AUC: %.4f +/- %.4f" % (auc_train, auc_unc_train)))

      self.tpr["validation"].append(tpr_validation)
      self.tpr["train"].append(tpr_train)
      self.fpr["validation"].append(fpr_validation)
      self.fpr["train"].append(fpr_train)

      self.auc["validation"].append(auc)
      self.auc_unc["validation"].append(auc_unc)
      self.auc["train"].append(auc_train)
      self.auc_unc["train"].append(auc_unc_train)

      self.model.save_weights("dnn_weights/" + self.tag + "_weights_%d.hdf5" % i)
      with open("dnn_weights/" + self.tag + "_model_architecture_%d.json" % i, "w") as f_out:
          f_out.write(self.model.to_json())

    rocs = { "fpr_train" : fpr_train, "tpr_train" : tpr_train, "thresh_train" : thresh_train, "fpr_validation" : fpr_validation, "tpr_validation" : tpr_validation, "thresh_validation" : thresh_validation }
    return auc_train, auc, rocs
    
  def train_with_early_stopping(self):
    best_auc = 0.5
    keep_training = True

    max_batch_size = 10000
    epochs = 1
    bad_epochs = 0
    while keep_training:
      auc_train, auc, rocs = self.train(epochs, self.batch_size_train)
      improvement = ((1-best_auc)-(1-auc))/(1-best_auc)
      overfit = (auc_train - auc) / auc_train
      if improvement > 0.01:
          print(("Improvement in (1-AUC) of %.3f percent! Keeping batch size the same" % (improvement*100.)))
          best_auc = auc
          bad_epochs = 0
      elif self.batch_size_train * 4 < max_batch_size:
          print(("Improvement in (1-AUC) of %.3f percent. Increasing batch size" % (improvement*100.)))
          self.batch_size_train *= 4
          bad_epochs = 0
          if auc > best_auc:
              best_auc = auc
      elif self.batch_size_train < max_batch_size:
          print(("Improvement in (1-AUC) of %.3f percent. Increasing batch size" % (improvement*100.)))
          self.batch_size_train = max_batch_size
          bad_epochs = 0
          if auc > best_auc:
              best_auc = auc 
      elif improvement > 0:
          print(("Improvement in (1-AUC) of %.3f percent. Can't increase batch size anymore" % (improvement*100.))) 
          bad_epochs = 0
          best_auc = auc
      #elif improvement < 0 and overfit < 0.01 and bad_epochs < 3:
      #    print (("Overfitting by less than 1%, continue training"))
      #    bad_epochs += 1
      else:
          print("AUC did not improve and we can't increase batch size anymore. Stopping training.")
          keep_training = False
      if self.n_epochs >= self.max_epochs:
          print("Have already trained for 25 epochs. Stopping training.")
          keep_training = False
      if self.curriculum_learn:
          value, idx = utils.find_nearest(rocs["tpr_train"], 0.90)
          cut = rocs["thresh_train"][idx]
          good_indices = numpy.where(self.predictions["train"] > cut)
          self.features_train.features[0] = self.features_train.features[0][good_indices]
          self.features_train.features[1] = self.features_train.features[1][good_indices]
          self.features_train.global_features = self.features_train.global_features[good_indices]
          self.features_train.objects = self.features_train.objects[good_indices]
          self.features_train.label = self.features_train.label[good_indices]
          self.features_train.weights = self.features_train.weights[good_indices]
          self.prepped = False


    auc, auc_unc, fpr, tpr, thresh = utils.auc_and_unc(self.features_validation.label, self.predictions["validation"], self.features_validation.weights, 50)
    auc_train, auc_unc_train, fpr_train, tpr_train, threshd_train = utils.auc_and_unc(self.features_train.label, self.predictions["train"], self.features_train.weights, 50)
    self.auc_unc["validation"] = auc_unc
    self.auc_unc["train"] = auc_unc_train

    self.model.save_weights("dnn_weights/" + self.tag + "_weights.hdf5")
    with open("dnn_weights/" + self.tag + "_model_architecture.json", "w") as f_out:
      f_out.write(self.model.to_json())

    return

  def debug(self):
    self.model.load_weights(self.weights_file)    
    self.predictions["data"] = self.model.predict(self.features_data.features, self.batch_size).flatten()
    for i in range(len(self.predictions["data"])):
      print(("Event", self.run_data[i], self.lumi_data[i], self.evt_data[i]))
      print(("Mass", self.mass_data[i]))
      print("Global features")
      print((self.features_data.global_features[i]))
      print("Object features")
      print((self.features_data.objects[i]))
      print("DNN Score")
      print((self.predictions["data"][i]))

  def initialize_plot(self):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    ax.grid(True)

  def make_learning_curve(self):
    x = numpy.linspace(1, self.n_epochs, num = self.n_epochs)
    numpy.savez("dnn_learning_curve_%s.npz" % self.tag, x = x, auc = self.auc["validation"], auc_unc = self.auc_unc["validation"], auc_train = self.auc["train"], auc_train_unc = self.auc_unc["train"])

    self.initialize_plot()
    plt.errorbar(x, self.auc["validation"], yerr = self.auc_unc["validation"], label = 'Validation Set', color = 'green')
    plt.errorbar(x, self.auc["train"], yerr = self.auc_unc["train"], label = "Training Set", color = 'red')
    plt.xlabel("Epoch #")
    plt.ylabel("AUC")
    plt.legend(loc = 'lower right')
    plt.savefig('learning_curve_%s.pdf' % self.tag)
    plt.close('all')

  def make_comparison(self, reference):
    fpr_ref, tpr_ref, thresh_ref = metrics.roc_curve(self.features_validation.label, self.features_validation.references[reference], pos_label = 1, sample_weight = self.features_validation.weights)

    numpy.savez("dnn_rocs_%s_%s.npz" % (reference.replace(" ", "_"), self.tag), fpr_validation = self.fpr["validation"][-1], tpr_validation = self.tpr["validation"][-1], fpr_train = self.fpr["train"][-1], tpr_train = self.tpr["train"][-1], fpr_ref = fpr_ref, tpr_ref = tpr_ref) 

    self.initialize_plot()
    plt.plot(self.fpr["validation"][-1], self.tpr["validation"][-1], color = 'darkred', lw = 2, label = 'DNN [AUC = %.3f]' % self.auc["validation"][-1])
    plt.plot(fpr_ref, tpr_ref, color = 'blue', lw = 2, label = '%s [AUC = %.3f]' % (reference, metrics.auc(fpr_ref, tpr_ref, reorder = True)))
    plt.xlabel("False Positive Rate (bkg. eff.)")
    plt.ylabel("True Positive Rate (sig. eff.)")
    plt.legend(loc = 'lower right')
    plt.savefig('dnn_roc_%s_%s.pdf' % (reference.replace(" ", "_"), self.tag))
    plt.close('all')

  def do_diagnostics(self):
    numpy.savez("dnn_scores_%s_.npz" % self.tag, scores_train = self.predictions["train"], scores_validation = self.predictions["validation"], scores_data = self.predictions["data"], scores_final_fit = self.predictions["final_fit"], evt_data = self.evt_data, run_data = self.run_data, lumi_data = self.lumi_data, mass_data = self.mass_data)
    self.make_learning_curve()
    for ref in list(self.features_validation.references.keys()):
      self.make_comparison(ref)

