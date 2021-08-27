from scipy.stats import ks_2samp
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def logical_vector(values, labels, label_target):
  array = []
  if not(len(values) == len(labels)):
    raise RuntimeError("length of target array must be the same as label array")

  values = numpy.asarray(values)
  labels = numpy.asarray(labels)

  array = values[numpy.where(labels == label_target)]
  return array

def plot_hist(distribution, bins, color, label):
  plt.hist(distribution, bins = bins, color = color, label = label, histtype = 'step', fill = False)

def ks_test(pred_train, pred_test, y_train, y_test):
  pred_train_sig = logical_vector(pred_train, y_train, 1)
  pred_train_bkg = logical_vector(pred_train, y_train, 0)
  pred_test_sig = logical_vector(pred_test, y_test, 1)
  pred_test_bkg = logical_vector(pred_test, y_test, 0)

  d_sig, p_value_sig = ks_2samp(pred_train_sig, pred_test_sig)
  d_bkg, p_value_bkg = ks_2samp(pred_train_bkg, pred_test_bkg)

  bins = numpy.linspace(0, 1, 20)
  # signal hist
  fig = plt.figure()
  plot_hist(pred_train_sig, bins, 'red', 'training set (signal)')
  plot_hist(pred_test_sig, bins, 'green', 'testing set (signal)')
  plt.text(0.05, 6000, 'KS Test D-statistic: %.6f' % (d_sig)) 
  plt.text(0.05, 5000, 'KS Test p-value: %.6f' % (p_value_sig))
  plt.xlabel('BDT Score')
  plt.legend(loc = 'upper center')
  plt.savefig('discriminant_hist_signal.pdf')

  # background hist
  fig = plt.figure()
  plot_hist(pred_train_bkg, bins, 'red', 'training set (background)')
  plot_hist(pred_test_bkg, bins, 'green', 'testing set (background)')
  plt.text(0.1, 6000, 'KS Test D-statistic: %.6f' % (d_bkg))
  plt.text(0.1, 5000, 'KS Test p-value: %.6f' % (p_value_bkg))
  plt.xlabel('BDT Score')
  plt.legend(loc = 'upper center')
  plt.savefig('discriminant_hist_background.pdf')

  return d_sig, p_value_sig, d_bkg, p_value_bkg

