import numpy
import ROOT, root_numpy
import math
from scipy.optimize import curve_fit
import itertools

import multiprocessing
from functools import partial
from contextlib import contextmanager

def Z_A(s, b):
  if s <= 0:
    return 0
  if b <= 0:
    return 0
  q1 = 2 * ( ((s+b) * math.log(1 + (s/b) )) - s)
  if q1 <= 0:
    return 0
  else:
    return math.sqrt(q1)

def unc_ZA(s, b, ds, db):
  Zs = math.log(1 + (s/b)) / Z_A(s,b)
  Zb = (math.log(1 + (s/b)) - (s/b)) / Z_A(s,b)
  return ((Zs*ds)**2 + (Zb*db)**2)**(0.5)

def quantiles_to_mva_score(n_quantiles, mva_scores): # return mva_scores corresponding to each quantile in n_quantiles
  sorted_mva = numpy.flip(numpy.sort(mva_scores), 0)
  quantiles = []
  mva = []
  for i in range(n_quantiles):
    idx = int((float(i+1) / float(n_quantiles)) * len(sorted_mva)) - 1
    quantiles.append(float(i+1) / float(n_quantiles))
    mva.append(sorted_mva[idx])
  return quantiles, mva

def all_ones(array):
  for item in array:
    if item != 1:
      return False
  return True

def gaus(x, a, b, c):
  return a * numpy.exp(-0.5 * (( (x - b) / c ) ** 2))

def constant_estimate(weights, sigma_eff, is_data):
  sideband_width = (180. - 130.) + (120. - 100.) if is_data else (180. - 100.)
  est = numpy.sum(weights) * ((2 * 1.645 * sigma_eff) / sideband_width)
  unc = math.sqrt(numpy.sum(weights**2)) * ((2 * 1.645 * sigma_eff) / (180. - 100.))
  if est > 0:
    return est, unc
  else:
    return 999999, 999999

def calc_sigma_eff(signal_data, weights): 
  h = ROOT.TH1D("h_sig", "", 80, 100, 180)
  for i in range(len(signal_data)):
    h.Fill(signal_data[i], weights[i])

  x_fit = numpy.linspace(120.5, 129.5, 10)
  y_fit = []
  y_err = []
  for i in range(h.GetNbinsX()):
    if not (h.GetBinCenter(i+1) > 120 and h.GetBinCenter(i+1) < 130):
      continue
    y_fit.append(h.GetBinContent(i+1))
    y_err.append(h.GetBinError(i+1))

  # fit
  popt, pcov = curve_fit(gaus, x_fit, y_fit, sigma = y_err, p0 = [1, 125, 2])

  del h
  return popt[1], abs(popt[2])


resonant_ids = [0, 11, 12, 14, 15, 16] # ttH, THQ, THW, ggH, VBF, VH
def events_passing_cut(events, cut, choose_resonant = False):
  events_pass = {"mass" : [], "weights" : []}#, "mva_score" : []}
  is_bkg = "process_id" in list(events.keys())
  for i in range(len(events["mass"])):
    pass_cut = True
    for mva_name in list(events["mva_score"].keys()):
      if not events["mva_score"][mva_name][i] > cut[mva_name]:
        pass_cut = False
    if is_bkg and not choose_resonant: # only do non-resonant bkgs here
        if events["process_id"][i] in resonant_ids:
            pass_cut = False # ignore these here, we will deal with resonant bkgs separately
    elif is_bkg and choose_resonant: # only do resonant bkgs here
        if events["process_id"][i] not in resonant_ids:
            pass_cut = False
    if pass_cut:
      events_pass["mass"].append(events["mass"][i])
      events_pass["weights"].append(events["weights"][i])
    
      #events_pass["mva_score"].append(events["mva_score"]["bdt_score"][i])
  return events_pass

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def calc_za(cut_tuple, signal_events, background_events, is_data, mc_bkg_events, mass_shift):
  testing_frac = 0.5
  cut_list = dict((x,y) for x,y in cut_tuple)
  print(cut_list)
  # Get events passing cut
  signal_events_passing_cut = events_passing_cut(signal_events, cut_list)
  bkg_events_passing_cut = events_passing_cut(background_events, cut_list)
  if is_data:
    resonant_bkg_events_passing_cut = events_passing_cut(mc_bkg_events, cut_list, True)
  else:
    resonant_bkg_events_passing_cut = events_passing_cut(background_events, cut_list, True)

  # Calculate sigma eff
  mean_eff, sigma_eff = calc_sigma_eff(signal_events_passing_cut["mass"], signal_events_passing_cut["weights"])

  # Calculate s
  signal_events_mass_window = []
  for i in range(len(signal_events_passing_cut["mass"])):
    if signal_events_passing_cut["mass"][i] > mean_eff - (1.645 * sigma_eff) and signal_events_passing_cut["mass"][i] < mean_eff + (1.645 * sigma_eff):
      signal_events_mass_window.append(signal_events_passing_cut["weights"][i])
  signal_events_mass_window = numpy.asarray(signal_events_mass_window)
  s = ( 1. / testing_frac) * numpy.sum(signal_events_mass_window)
  s_unc = ( 1. / testing_frac) * math.sqrt(numpy.sum(signal_events_mass_window**2))

  # Calculate b
  bkg_events_mass_window = []
  for i in range(len(bkg_events_passing_cut["mass"])):
    if bkg_events_passing_cut["mass"][i] > 100 and bkg_events_passing_cut["mass"][i] < 180:
      bkg_events_mass_window.append(bkg_events_passing_cut["weights"][i])
  bkg_events_mass_window = numpy.asarray(bkg_events_mass_window)
  #b = ( 1. / testing_frac) * constant_estimate(bkg_events_mass_window, sigma_eff)
  b_non_res, b_non_res_unc = constant_estimate(bkg_events_mass_window, sigma_eff, is_data)
  if not is_data:
    b_non_res *= ( 1. / testing_frac)
    b_non_res_unc *= ( 1. / testing_frac)
  # Now add resonant bkg contribution
  resonant_bkg_events_mass_window = []
  shift = 2 if mass_shift else 0
  for i in range(len(resonant_bkg_events_passing_cut["mass"])):
    if resonant_bkg_events_passing_cut["mass"][i] > (mean_eff-shift) - (1.645 * sigma_eff) and resonant_bkg_events_passing_cut["mass"][i] < (mean_eff-shift) + (1.645 * sigma_eff): # (mean_eff-shift) because we use the M127 ttH sample but M125 for other processes
      resonant_bkg_events_mass_window.append(resonant_bkg_events_passing_cut["weights"][i])
  resonant_bkg_events_mass_window = numpy.asarray(resonant_bkg_events_mass_window)
  b_res = ( 1. / testing_frac) * numpy.sum(resonant_bkg_events_mass_window)
  b_res_unc = ( 1. / testing_frac) * math.sqrt(numpy.sum(resonant_bkg_events_mass_window**2))

  b = b_non_res + b_res
  b_unc = ((b_non_res_unc**2) + (b_res_unc**2))**(0.5)

  # Calculate Z_A
  z_mc = Z_A(s, b)
  z_mc_unc = unc_ZA(s, b, s_unc, b_unc) if z_mc > 0 else 0
  print(("s: %.3f +/- %.3f, b: %.3f +/- %.3f, mean_eff: %.3f, sigma_eff: %.3f" % (s, s_unc, b, b_unc, mean_eff, sigma_eff)))
  print(("non-resonant background: %.3f +/- %.3f, resonant background: %.3f +/- %.3f" % (b_non_res, b_non_res_unc, b_res, b_res_unc)))
  print(("za: %.3f +/- %.3f" % (z_mc, z_mc_unc)))
  return cut_list, z_mc, z_mc_unc, s, b, sigma_eff 

  #za_dict.append({"z_mc": z_mc, "cut_list" : cut_list})

def scan_za_parallel(cut_combos, signal_events, background_events, is_data, mc_bkg_events, mass_shift):
  testing_frac = 0.5 # this assumes that we always use half the mc for testing and half for training

  za_dict = []
  n_points = len(cut_combos)
  cuts = [] 
  za = []
  za_unc = []
  n_sig = []
  n_bkg = []
  sigma_eff_ = []

  with poolcontext(processes=8) as pool:
    results = pool.map(partial(calc_za, signal_events = signal_events, background_events = background_events, is_data = is_data, mc_bkg_events = mc_bkg_events, mass_shift = mass_shift), cut_combos) 
    for result in results:
      print(result)
      cuts.append(result[0])
      za.append(result[1])
      za_unc.append(result[2])
      n_sig.append(result[3])
      n_bkg.append(result[4])
      sigma_eff_.append(result[5])

  return {"cuts" : cuts, "za" : za, "za_unc": za_unc, "n_sig" : n_sig, "n_bkg" : n_bkg, "sigma_eff" : sigma_eff_}

def scan_za(cut_combos, signal_events, background_events, is_data, mc_bkg_events, mass_shift):
  testing_frac = 0.5 # this assumes that we always use half the mc for testing and half for training

  za_dict = []
  cuts = []
  za = []
  za_unc = []
  n_sig = []
  n_bkg = []
  sigma_eff_ = []

  for cut_tuple in cut_combos:
    cut_list = dict((x,y) for x,y in cut_tuple)
    print(cut_list)
    # Get events passing cut
    signal_events_passing_cut = events_passing_cut(signal_events, cut_list)
    bkg_events_passing_cut = events_passing_cut(background_events, cut_list)
    if is_data:
      resonant_bkg_events_passing_cut = events_passing_cut(mc_bkg_events, cut_list, True)
    else:
      resonant_bkg_events_passing_cut = events_passing_cut(background_events, cut_list, True)

    # Calculate sigma eff
    mean_eff, sigma_eff = calc_sigma_eff(signal_events_passing_cut["mass"], signal_events_passing_cut["weights"])

    # Calculate s
    signal_events_mass_window = []
    for i in range(len(signal_events_passing_cut["mass"])):
      if signal_events_passing_cut["mass"][i] > mean_eff - (1.645 * sigma_eff) and signal_events_passing_cut["mass"][i] < mean_eff + (1.645 * sigma_eff):
        signal_events_mass_window.append(signal_events_passing_cut["weights"][i])
    signal_events_mass_window = numpy.asarray(signal_events_mass_window)
    s = ( 1. / testing_frac) * numpy.sum(signal_events_mass_window)
    s_unc = ( 1. / testing_frac) * math.sqrt(numpy.sum(signal_events_mass_window**2))

    # Calculate b
    bkg_events_mass_window = []
    for i in range(len(bkg_events_passing_cut["mass"])):
      if bkg_events_passing_cut["mass"][i] > 100 and bkg_events_passing_cut["mass"][i] < 180:
        bkg_events_mass_window.append(bkg_events_passing_cut["weights"][i])
    bkg_events_mass_window = numpy.asarray(bkg_events_mass_window)
    #b = ( 1. / testing_frac) * constant_estimate(bkg_events_mass_window, sigma_eff)
    b_non_res, b_non_res_unc = constant_estimate(bkg_events_mass_window, sigma_eff, is_data)
    if not is_data:
      b_non_res *= ( 1. / testing_frac)
      b_non_res_unc *= ( 1. / testing_frac)

    # Now add resonant bkg contribution
    resonant_bkg_events_mass_window = []
    shift = 2 if mass_shift else 0
    for i in range(len(resonant_bkg_events_passing_cut["mass"])):
      if resonant_bkg_events_passing_cut["mass"][i] > (mean_eff-shift) - (1.645 * sigma_eff) and resonant_bkg_events_passing_cut["mass"][i] < (mean_eff-shift) + (1.645 * sigma_eff): # (mean_eff-shift) because we use the M127 ttH sample but M125 for other processes
        resonant_bkg_events_mass_window.append(resonant_bkg_events_passing_cut["weights"][i])
    resonant_bkg_events_mass_window = numpy.asarray(resonant_bkg_events_mass_window)
    b_res = ( 1. / testing_frac) * numpy.sum(resonant_bkg_events_mass_window)
    b_res_unc = ( 1. / testing_frac) * math.sqrt(numpy.sum(resonant_bkg_events_mass_window**2))

    b = b_non_res + b_res
    b_unc = ((b_non_res_unc**2) + (b_res_unc**2))**(0.5)

    # Calculate Z_A
    z_mc = Z_A(s, b)
    z_mc_unc = unc_ZA(s, b, s_unc, b_unc) if z_mc > 0 else 0
    print(("s: %.3f +/- %.3f, b: %.3f +/- %.3f, mean_eff: %.3f, sigma_eff: %.3f" % (s, s_unc, b, b_unc, mean_eff, sigma_eff)))
    print(("non-resonant background: %.3f +/- %.3f, resonant background: %.3f +/- %.3f" % (b_non_res, b_non_res_unc, b_res, b_res_unc)))
    print(("za: %.3f +/- %.3f" % (z_mc, z_mc_unc)))
    za_dict.append({"z_mc": z_mc, "cut_list" : cut_list}) 
    cuts.append(cut_list)
    za.append(z_mc)
    za_unc.append(z_mc_unc)
    n_sig.append(s)
    n_bkg.append(b)
    sigma_eff_.append(sigma_eff)

  return {"cuts" : cuts, "za" : za, "za_unc": za_unc, "n_sig" : n_sig, "n_bkg" : n_bkg, "sigma_eff" : sigma_eff_}


def take_first(array):
  return array[0]

def find_nearest(array,value):
    val = numpy.ones_like(array)*value
    idx = (numpy.abs(array-val)).argmin()
    return array[idx], idx

def extract_1d_curve(full_za_dict):
  # First sort za_dict by n_sig
  sort_indices = numpy.argsort(full_za_dict["n_sig"])
  #print sort_indices
  for key in list(full_za_dict.keys()):
    array = numpy.asarray(full_za_dict[key])
    #print array
    full_za_dict[key] = array[sort_indices]
  
  # Now pick max Z_A for each successive group of points
  good_indices = []
  n_points_total = len(full_za_dict["n_sig"])
  n_groups = 50
  n_points_per_group = int(n_points_total / n_groups)

  za_group = []
  for i in range(n_points_total):
    za_group.append(full_za_dict["za"][i])
    if (i+1) % n_points_per_group == 0: 
      #print "Group considering %d points" % n_points_per_group
      #print "Z_A values were: ", za_group
      best_index = numpy.argmax(za_group)
      #print "And chose %.3f as best" % (za_group[best_index])
      best_index_global = ((i+1) - n_points_per_group) + best_index
      #print "This should be the same %.3f" % (full_za_dict["za"][best_index_global])
      good_indices.append(best_index_global)
      za_group = []

  print(good_indices)

  za_dict_trimmed = {}
  for key in list(full_za_dict.keys()):
    za_dict_trimmed[key] = full_za_dict[key][good_indices]

  return za_dict_trimmed

  # Takes in all points from N-d scan and constructs Z_A vs. signal curve with highest Z_A value for each signal yield
  #za = full_za_dict["za"]
  #n_sig = full_za_dict["n_sig"]
  
  #za_and_n_sig = [] 
  #for i in range(len(n_sig)):
  #  za_and_n_sig.append([n_sig[i], za[i]])

  #za_ordered_by_n_sig = za_and_n_sig.sort(key=take_first)
  #range = abs(za_ordered_by_n_sig[0] - za_ordered_by_n_sig[-1])
  #print "Range of n_signal %.3f" % range
  
  #sig_points = numpy.linspace(za_ordered_by_n_sig[0], za_ordered_by_n_sig[-1], 20)
  #print "Sampling points near following signal values: ", sig_points

  #sig_groups = {}
  #for point in sig_points:
  #  sig_groups[point] = []
  
  #for pair in za_ordered_by_n_sig:
  #  n_sig_nearest, idx_nearest = find_nearest(sig_points, pair[0])
  #  sig_groups[n_sig_nearest].append([pair])

  #za_dict_trimmed = {}

  #for group in sig_groups:
  #  group = numpy.asarray(group)
  #  idx_max_in_group = numpy.argmax(group[:,1])

  #return n_sig_trimmed, za_trimmed


def za_scores(n_quantiles, signal_events, background_events, is_data, mc_bkg_events = {}, mass_shift = True):
  # Calculate cuts corresponding to each quantile
  mva_cuts = {}
  for mva_name in list(signal_events["mva_score"].keys()):
    quantiles, mva_cut = quantiles_to_mva_score(n_quantiles, signal_events["mva_score"][mva_name])
    print("Evenly spaced signal efficiency cuts:") 
    print((mva_name, mva_cut))
    mva_cut_tuples = []
    for cut in mva_cut:
      mva_cut_tuples.append((mva_name, cut))
    mva_cuts[mva_name] = { "cuts" : mva_cut, "quantiles" : quantiles , "cut_tuples" : mva_cut_tuples}

  # Make list of all cut combinations (N-d)
  mva_cut_list = []
  for mva_name in list(mva_cuts.keys()):
    mva_cut_list.append(mva_cuts[mva_name]["cut_tuples"])
  cut_combos = list(itertools.product(*mva_cut_list)) # all unique combinations of cuts from each of the N MVAs
  print(("Performing %d-D scan with %d points" % (len(list(signal_events["mva_score"].keys())), len(cut_combos))))

  # Perform full N-d scan
  #full_za_dict = scan_za(cut_combos, signal_events, background_events, is_data, mc_bkg_events, mass_shift) # Full N-d scan
  full_za_dict = scan_za_parallel(cut_combos, signal_events, background_events, is_data, mc_bkg_events, mass_shift) # Full N-d scan

  if len(list(mva_cuts.keys())) > 1:
    za_dict_1d = extract_1d_curve(full_za_dict)

    # Now fix all other MVA scores except BDT at the value for which we get max Z_A, and perform 1-d scan on BDT score


    #max_idx = numpy.argmax(full_za_dict["za"])
    #mva_values_at_max = full_za_dict["za_dict"][max_idx]["cut_list"]
    
    #mva_cut_list_1d = [mva_cuts["bdt_score"]["cut_tuples"]]
    #for mva_name in mva_cuts.keys():
    #  if mva_name == "bdt_score":
    #    continue
    #  else:
    #    print "Fixing 1d optimization at %s : %.4f" % (mva_name, mva_values_at_max[mva_name])
    #    mva_cut_list_1d.append([(mva_name, mva_values_at_max[mva_name])])
    #cut_combos_1d = list(itertools.product(*mva_cut_list_1d))
    #za_dict_1d = scan_za(cut_combos_1d, signal_events, background_events, is_data)
  else:
    za_dict_1d = full_za_dict

  return za_dict_1d["za"], za_dict_1d["za_unc"], za_dict_1d["n_sig"], za_dict_1d["n_bkg"], za_dict_1d["sigma_eff"]
