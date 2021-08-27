import numpy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--inputs", help = "csv list of which .npz files to consider", type=str)
parser.add_argument("--labels", help = "csv list of labels for each .npz file", type=str)
parser.add_argument("--colors", nargs='+', help = "<Required> Set colors for curves", type=str, required=True)
parser.add_argument("--ylim", help = "csv list of ylim", type=str)
parser.add_argument("--tag", help = "tag to denote with", type=str)
args = parser.parse_args()

print "------------------------------------------------------------"
print "Hello World! (from MVAs/make_za_comparisons.py)"
print "------------------------------------------------------------"
print "args.inputs               : %s" % args.inputs
print "args.labels               : %s" % args.labels
print "args.colors               : %s" % args.colors
print "args.ylim                 : %s" % args.ylim
print "args.tag                  : %s" % args.tag
print "------------------------------------------------------------"

inputs = (args.inputs.replace(" ","")).split(",")
labels = args.labels.split(",")

#colors = ["black", "blue", "green", "orange", "red", "purple", "brown"]
colors = args.colors

files = []
for input in inputs:
  print input
  files.append(numpy.load(input))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def print_relative_percentage(tag, wp, before, after):
    relative_percentage = 100. *  (after - before) / before
    print 'Improvement in %s (%s) = %.2f %% (before = %.2f, after = %.2f)' % (tag, wp, relative_percentage, before, after)

def print_full_relative_percentage(list_before, list_after):
    list_diff = []
    zip_list = zip(list_before, list_after)
    for ele_1, ele_2 in zip_list:
        relative_difference = 100. * (ele_2 - ele_1) / ele_1
        list_diff.append(relative_difference)
    max = numpy.max(list_diff)
    print 'max improvement from full scan = %.2f' % (max)
    print list_diff

#----------------------------------------------------------------------------------------------------#

fig = plt.figure()
ax1 = fig.add_subplot(111)

list_max = []
list_50_eff = []
for i in range(len(files)):
  print("MC")
  print(("Max ZA", numpy.max(files[i]["za"]), " +/- ", files[i]["za_unc"][numpy.argmax(files[i]["za"])]))
  print(("50 sig eff ZA", files[i]["za"][len(files[i]["za"])/2], " +/- ", files[i]["za_unc"][len(files[i]["za"])/2]))
  list_max.append(numpy.max(files[i]["za"]))
  list_50_eff.append(files[i]["za"][len(files[i]["za"])/2])
  ax1.plot(files[i]["signal"], files[i]["za"], label = labels[i], color = colors[i])
  ax1.fill_between(files[i]["signal"], files[i]["za"] - files[i]["za_unc"], files[i]["za"] + files[i]["za_unc"], color = colors[i], alpha = 0.25)
  plt.xlabel('# Signal Events')
  plt.gca().set_xlim(left=2)
  ax1.set_ylabel('Significance (Z_A)')
  ax1.legend(loc='lower left')
  
print_relative_percentage(args.tag, "max", list_max[-2], list_max[-1])
print_relative_percentage(args.tag, "50_eff", list_50_eff[-2], list_50_eff[-1])
print_full_relative_percentage(files[-2]["za"], files[-1]["za"])
print files[-1]["signal"]
print files[-1]["za"]
print files[-2]["signal"]
print files[-2]["za"]

ymax = numpy.max(list_max) * 1.2
if args.ylim:
  ylim = args.ylim.split(",")
  #plt.ylim([float(ylim[0]), float(ylim[1])])
  plt.ylim([float(ylim[0]), float(ymax)])

plt.savefig('za_comparison_mc_%s.pdf' % (args.tag))

#----------------------------------------------------------------------------------------------------#

fig = plt.figure()
ax1 = fig.add_subplot(111)

for i in range(len(files)):
  print("Data")
  print((numpy.max(files[i]["za_data"]), " +/- ", files[i]["za_unc_data"][numpy.argmax(files[i]["za_data"])]))
  print files[i]["za_data"]
  print((files[i]["za_data"][len(files[i]["za_data"])/2], " +/- ", files[i]["za_unc_data"][len(files[i]["za_data"])/2]))
  ax1.plot(files[i]["signal_data"], files[i]["za_data"], label = labels[i], color = colors[i])
  ax1.fill_between(files[i]["signal_data"], files[i]["za_data"] - files[i]["za_unc_data"], files[i]["za_data"] + files[i]["za_unc_data"], color = colors[i], alpha = 0.25)
  plt.xlabel('# Signal Events')
  plt.gca().set_xlim(left=2.5)
  ax1.set_ylabel('Significance (Z_A)')
  ax1.legend(loc='lower left')

print_relative_percentage(args.tag, "max", list_max[-2], list_max[-1])
print_relative_percentage(args.tag, "50_eff", list_50_eff[-2], list_50_eff[-1])
print_full_relative_percentage(files[-2]["za_data"], files[-1]["za_data"])
print files[-1]["signal_data"]
print files[-1]["za_data"]
print files[-2]["signal_data"]
print files[-2]["za_data"]

ymax = numpy.max(list_max) * 1.2
if args.ylim:
  ylim = args.ylim.split(",")
  #plt.ylim([float(ylim[0]), float(ylim[1])])
  plt.ylim([float(ylim[0]), float(ymax)])

plt.savefig('za_comparison_data_%s.pdf' % (args.tag))

#----------------------------------------------------------------------------------------------------#
