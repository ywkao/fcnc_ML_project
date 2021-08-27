import numpy
import random
from sklearn import metrics
#from sklearn.preprocessing import quantile_transform

n_max_objects = 8
pad_value = -9
def preprocess(array, mean, std, z_score=True):
  if z_score:
    array[array != pad_value] += -mean
    array[array != pad_value] *= 1./std
  #elif z_score:
  else:
    min = numpy.min(array[array != pad_value])
    max = numpy.max(array[array != pad_value])
    array[array != pad_value] += -min
    array[array != pad_value] *= 1./(max - min)
    array[array != pad_value] += -0.5
    array[array != pad_value] *= 2
    array[array == pad_value] = -1.1
    print (min, max, numpy.min(array[array != pad_value]), numpy.max(array[array != pad_value]), numpy.mean(array[array != -1.1]))
  #else:
  #  array[array != pad_value] = quantile_transform(array[array != pad_value].reshape(-1,1))
  #  array[array != pad_value] += -0.5
  #  array[array != pad_value] *= 2
  #  array[array == pad_value] = -1.1
  #  print (min, max, numpy.min(array[array != pad_value]), numpy.max(array[array != pad_value]), numpy.mean(array[array != pad_value]))
#array[array == pad_value] = 0
  return array

def get_mean_and_std(array):
  array_trimmed = array[array != pad_value]
  if (len(array_trimmed) <= 1):
    return 0, 1
  mean = numpy.mean(array_trimmed)
  std = numpy.std(array_trimmed)
  if std == 0:
    std = 1
  return mean, std

def create_array(features_list, names, dict_, z_score = True):
    arr = []
    for name in names:
        if "leptons_" == name or "jets_" == name or "objects_" in name:
            continue
        else:
            feat = numpy.array(features_list[name])
            if z_score and dict_ != "none":
                preprocessed_feat = preprocess(feat, dict_[name]["mean"], dict_[name]["std_dev"])
            else:
                preprocessed_feat = feat
            arr.append(preprocessed_feat)
    return numpy.transpose(numpy.array(arr))

def pad_array(array):
  if len(array) == 0:
    return array
  max_objects = n_max_objects
  nData = len(array)
  nFeatures = len(array[0][0])
  y = numpy.ones((nData, max_objects, nFeatures))
  y *= pad_value
  for i in range(nData):
    for j in range(min(max_objects, len(array[i]))):
      for k in range(nFeatures):
        y[i][j][k] = array[i][j][k]
  return y

def preprocess_array(y, dict_):
  if dict_ == "none":
    return y
  for i in range(len(y[0][0])):
    y[:,:,i] = preprocess(y[:,:,i], dict_["objects_" + str(i)]["mean"], dict_["objects_" + str(i)]["std_dev"])
  return y

def oversample(array, indices):
  return numpy.array([array[i] for i in indices])

def auc_and_unc(label, pred, sample_weight, n_bootstraps):
  fpr, tpr, thresh = metrics.roc_curve(label, pred, pos_label = 1, sample_weight = sample_weight)
  auc = metrics.auc(fpr, tpr, reorder=True)

  label = numpy.asarray(label)
  pred  = numpy.asarray(pred)
  sample_weight = numpy.asarray(sample_weight)

  n_points = len(label)
  bootstrap_aucs = numpy.empty(n_bootstraps)
  for i in range(n_bootstraps):
    bootstrap_indices = numpy.random.randint(0, n_points, n_points)
    bootstrap_label = label[bootstrap_indices]
    bootstrap_pred = pred[bootstrap_indices]
    bootstrap_weights = sample_weight[bootstrap_indices]
    fpr_b, tpr_b, thresh_b = metrics.roc_curve(bootstrap_label, bootstrap_pred, pos_label = 1, sample_weight = bootstrap_weights)
    bootstrap_auc = metrics.auc(fpr_b, tpr_b, reorder=True)
    bootstrap_aucs[i] = bootstrap_auc

  unc = numpy.std(bootstrap_aucs)
  #print bootstrap_aucs
  return auc, unc, fpr, tpr, thresh

def sum_of_weights(weights, label, label_index):
  sum = 0
  for i in range(len(weights)):
    if label[i] == label_index:
      #if weights[i] > 999:
      #  print weights[i]
      sum += weights[i]
  return sum

def sum_of_weights_v2(weights, label, label_index):
  weights = numpy.asarray(weights)
  return numpy.sum(weights[numpy.asarray(label) == label_index])

def shuffle(x, y, weights):
  rand = numpy.random.permutation(len(y))
  x = numpy.asarray(x)
  x = x[rand]
  y = numpy.asarray(y)
  y = y[rand]
  weights = numpy.asarray(weights)
  weights = weights[rand]
  return x, y, weights

def load_array(file, name):
  array = file[name]
  array = numpy.asarray(array)
  return array

def find_nearest(array,value):
    val = numpy.ones_like(array)*value
    idx = (numpy.abs(array-val)).argmin()
    return array[idx], idx

import subprocess, re

# Nvidia-smi GPU memory parsing.

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu
