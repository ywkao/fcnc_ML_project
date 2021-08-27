import numpy
import random
from sklearn import metrics
import ROOT

def numpy_to_tree(dict, file_name, tree_name = "t"): # dict should contain {"branch_name" : array}
  f = ROOT.TFile(file_name, "recreate")
  t = ROOT.TTree(tree_name, "tree")

  for branch in list(dict.keys()):
    branch_name = branch

  n_events = len(dict[branch_name])

  branches = {}
  for branch in list(dict.keys()):
    if branch == "global_features" or branch == "dnn_global_features":
      b = numpy.empty((1, len(dict[branch][0])), dtype = "float32")
      branches[branch] = b
      t.Branch(branch, branches[branch], "%s[%d]/F" % (branch, len(dict[branch][0])))
    else:
      b = numpy.empty((1), dtype = "float32")
      branches[branch] = b
      t.Branch(branch, branches[branch], branch + "/F")

  for i in range(n_events):
    for branch in list(dict.keys()):
      if branch == "global_features" or branch == "dnn_global_features":
        for j in range(len(dict[branch][i])):
          branches[branch][0][j] = dict[branch][i][j]
      else:
        branches[branch][0] = dict[branch][i]
    t.Fill()

  f.Write()
  f.Close()

  return

