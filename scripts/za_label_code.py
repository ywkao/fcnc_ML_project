def strategy_0():
    return "Default (no top taggers)"

def strategy_1():
    return "Top Tagger BDT"

def strategy_2a():
    return "Minimum Chi-2 Method"

def strategy_2b():
    return "Quadratic Equation"

def strategy_3():
    return "FCNC Reco ANN"

def strategy_4a():
    return "Top Tagger BDT + Miniumum Chi-2 Method"

def strategy_4b():
    return "Top Tagger BDT + Quadratic Equation"

def strategy_5a():
    return "Miniumum Chi-2 Method + FCNC Reco ANN"

def strategy_5b():
    return "Quadratic Equation + FCNC Reco ANN"

def strategy_6():
    return "FCNC Reco ANN + Top Tagger BDT"

def strategy_7():
    return "Three Methods"

def strategy_8():
    return "Three Methods + cvsl and cvsb"

def code_to_label(code, channel):
    switcher_had = {
        0: strategy_0,
        1: strategy_1,
        2: strategy_2a,
        3: strategy_3,
        4: strategy_4a,
        5: strategy_5a,
        6: strategy_6,
        7: strategy_7,
        8: strategy_8
    }

    switcher_lep = {
        0: strategy_0,
        1: strategy_1,
        2: strategy_2b,
        3: strategy_3,
        4: strategy_4b,
        5: strategy_5b,
        6: strategy_6,
        7: strategy_7,
        8: strategy_8
    }

    switcher = switcher_had if channel == "Hadronic" else switcher_lep
    func = switcher.get(code, lambda: "Invalid strategy code")
    label = func()
    return label

