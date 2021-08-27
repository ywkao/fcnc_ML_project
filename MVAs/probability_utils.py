import ROOT
import significance_utils
from array import array

def logical_vector(values, labels, label_target):
  array = []
  if not(len(values) == len(labels)):
    raise RuntimeError("length of target array must be the same as label array")
  for i in range(len(values)):
    if labels[i] == label_target:
      array.append(values[i])
  return array

def calculate_prob_ratios(scores, label):
  n_quantiles = 10

  scores_signal = logical_vector(scores, label, 1)
  scores_bkg = logical_vector(scores, label, 0)

  quantiles, bins = significance_utils.quantiles_to_mva_score(n_quantiles, scores_signal)
  bins = bins[::-1]
  bins.insert(0,0) 

  print(bins)

  hSig = ROOT.TH1D("hSig", "", len(bins) - 1, array('d',bins))
  for evt in scores_signal:
    hSig.Fill(evt, 1.)

  hBkg = ROOT.TH1D("hBkg", "", len(bins) - 1, array('d', bins))
  for evt in scores_bkg:
    hBkg.Fill(evt, 1.)

  # Scale histos then take ratio
  hSig.Scale(1./hSig.GetSumOfWeights())
  hBkg.Scale(1./hBkg.GetSumOfWeights())

  for i in range(len(bins) - 1):
    print((hSig.GetBinContent(i+1), hBkg.GetBinContent(i+1)))

  hSig.Divide(hBkg) # divide Madgraph probabilities by Pythia probabilities

  for i in range(len(bins) - 1):
    print((hSig.GetBinContent(i+1)))

  prob_ratios = []
  for i in range(len(bins) - 1):
    prob_ratios.append(hSig.GetBinContent(i+1))

  return prob_ratios, bins[1:]

