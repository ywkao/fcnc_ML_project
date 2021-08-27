#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)
c1 = ROOT.TCanvas('c1', '', 800, 600)

labels = { #{{{
    0  : "ttHJet / ttHToGG",
    1  : "DY",
    2  : "DiPhoton",
    3  : "GJet_Pt",
    4  : "QCD",
    5  : "TTGG",
    6  : "TTGJets",
    9  : "TT (Jets/2L2Nu/SemiLeptonic)",
    7  : "WG / ZG",
    8  : "WJets",
    10 : "DoubleEG / EGamma",
    11 : "THQ",
    12 : "THW",
    13 : "TGJets",
    14 : "GluGluHToGG",
    15 : "VBF",
    16 : "VHToGG",
    17 : "GJets_HT",
    18 : "imputed_QCD_GJets",
    19 : "TTZ",
    20 : "WW / WZ / ZZ",
    21 : "ST_tW / tZq",
    22 : "TT_FCNC_hut",
    23 : "TT_FCNC_hct",
    24 : "ST_FCNC_hut",
    25 : "ST_FCNC_hct",
    26 : "TTW",
} #}}}
def check_mvababy_var(var, root): #{{{
    filename = "../../%s" % root
    print "# file = ", filename

    f1 = ROOT.TFile.Open(filename, "R")
    t = f1.Get("t")
    
    h = ROOT.TH1D("h_%s" % var, "", 37, 0, 37)
    t.Draw("%s >> h_%s" % (var, var), "")
    h.SetStats(0)
    
    ids = [0, 1, 2, 3, 4, 5, 6, 9, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    for id in sorted(ids):
        print ">>> id: %2d, Entry of %-30s = %d" % (id, labels[id], h.GetBinContent(id+1))
    
    print ""
    #output = "mvaBaby_%s.png" % (var)
    #c1.SaveAs(output)
    #update_to_my_webpage(output)
#}}}
def print_tree(root): #{{{
    filename = "../%s" % root
    print "# file = ", filename

    f1 = ROOT.TFile.Open(filename, "R")
    t = f1.Get("t")
    t.Print()
#}}}

if __name__ == "__main__":
    rootfiles = [
        "MVABaby_ttHHadronic_v5.7_test_impute_hct_BDT_FCNC.root",
        "MVABaby_ttHHadronic_v5.7_test_impute_hut_BDT_FCNC.root",
        "MVABaby_ttHLeptonic_v5.7_test_hct_BDT_FCNC.root",
        "MVABaby_ttHLeptonic_v5.7_test_hut_BDT_FCNC.root",
    ]
    for root in rootfiles:
        check_mvababy_var("process_id_", root)
        #print_tree(root)
        break
