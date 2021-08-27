import os, sys
import json
import numpy

from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

sys.path.append("../Loopers")
import parallel_utils
import train_dnn_core

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "input hdf5 file", type=str)
parser.add_argument("--tag", help = "save name for weights file", type=str)
parser.add_argument("--channel", help = "Hadronic or Leptonic", type=str, default="Hadronic")
parser.add_argument("--no_global", help = "don't use global features", action="store_true")
parser.add_argument("--no_lstm", help = "don't use object features (lstm)", action="store_true")
parser.add_argument("--load", help = "give weights file to use as starting point", type=str)
parser.add_argument("--random", help = "do random exploration instead of bayesian exploration", action="store_true")
parser.add_argument("--no_bootstrap", help = "don't use bootstrapping to estimate unc. in AUC (to save time during hyperparameter opt)", action="store_true")
parser.add_argument("--absolute_weights", help = "set negative weights to positive in *training* only (to improve stats)", action="store_true")
parser.add_argument("--n_points", help = "how many points to probe", type=str, default="200")
#parser.add_argument("--pbounds", help = "which pbounds set to consider", type=str)
parser.add_argument("--preprocess_scheme", help = "json used for preprocessing features", type=str)
parser.add_argument("--no_buildup", help = "don't build up with light, medium, full pbounds", action = "store_true")
parser.add_argument("--fixed", help = "used a fixed set of pbounds (for calculating systematics unc.)", action = "store_true")
parser.add_argument("--xi", help = "exploitation vs exploration parameter for hyperparam opt", type=str, default="0.0005")
parser.add_argument("--alpha", help = "noise parameter for BO", type=str, default="1e-05")

args = parser.parse_args()
args.n_points = int(args.n_points)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x):
    fig = plt.figure()
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process after {} steps'.format(steps),
        fontdict={'size':30}
    )
    
    axis = fig.add_subplot(111)
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    #axis = plt.subplot(gs[0])
    #acq = plt.subplot(gs[1])
    
    x_obs = numpy.array([[res["params"]["learning_rate"]] for res in optimizer.res])
    y_obs = numpy.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    #axis.plot(x, y, linewidth=3, label='Target')
    unc = 0.0033 # calculated for Leptonic ttH vs ttGG
    axis.errorbar(x_obs.flatten(), y_obs, yerr = numpy.ones(len(y_obs)) * unc, label='Observations', color='r', marker='o', markersize=8, ls="none")
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(numpy.concatenate([x, x[::-1]]), 
              numpy.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-5, -1))
    axis.set_ylim((0.7, 0.82))
    axis.set_ylabel('AUC', fontdict={'size':20})
    axis.set_xlabel('Learning Rate', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    #utility_function = UtilityFunction(kind="ei", xi=float(args.xi))
    utility = utility_function.utility(x, optimizer._gp, 0)
    #acq.plot(x, utility, label='Utility Function', color='purple')
    #acq.plot(x[numpy.argmax(utility)], numpy.max(utility), '*', markersize=15, 
             #label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    #acq.set_xlim((-2, 10))
    #acq.set_ylim((0, numpy.max(utility) + 0.5))
    #acq.set_ylabel('Utility', fontdict={'size':20})
    #acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc='upper left')
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.savefig('optimization_step_%d.pdf' % (steps), bbox_inches='tight')
    plt.clf() 

def auc(n_nodes_dense_1, n_nodes_dense_2, n_dense_1, n_dense_2, n_nodes_lstm, n_lstm, maxnorm, dropout_rate, learning_rate, start_batch, batch_momentum, epsilon):
    global idx, log, full_results

    config = {}
    config["n_nodes_dense_1"] = int(n_nodes_dense_1)
    config["n_nodes_dense_2"] = int(n_nodes_dense_2)
    config["n_dense_1"] = int(n_dense_1)
    config["n_dense_2"] = int(n_dense_2)
    config["n_nodes_lstm"] = int(n_nodes_lstm)
    config["n_lstm"] = int(n_lstm)
    config["maxnorm"] = 10**(maxnorm)
    config["epsilon"] = 10**(epsilon)
    config["dropout_rate"] = dropout_rate
    config["learning_rate"] = 10**(learning_rate)
    config["start_batch"] = int(2**(start_batch))
    config["batch_norm"] = True
    config["layer_norm"] = False
    config["batch_momentum"] = batch_momentum

    print(("Index: ", idx))
    print(("Config: " , config))

    #if idx in full_results.keys():
    #    target = full_results[idx]["auc_test"][-1]

    #else:
    trained_dnn = train_dnn_core.train(args, config)
    full_results[str(idx)] = {"config" : config, "results" : {"auc_train" : trained_dnn.auc["train"], "auc_train_unc" : trained_dnn.auc_unc["train"], "auc_test" : trained_dnn.auc["validation"], "auc_test_unc" : trained_dnn.auc_unc["validation"]}}
    target = trained_dnn.auc["validation"][-1]

    with open(log, "w") as f_out:
        json.dump(full_results, f_out, indent=4, sort_keys=True)

    idx += 1
    return target 

idx = 0
log = "bayes_dnn_hyperparam_scan_%s_%s.json" % (args.channel, args.tag)
full_results = { "input" : args.input }

pbounds_light = {
    "n_nodes_dense_1" : (300, 300), 
    "n_nodes_dense_2" : (200, 200), 
    "n_dense_1" : (1,1), 
    "n_dense_2" : (2,2), 
    "n_nodes_lstm" : (100, 100), 
    "n_lstm" : (1,1), 
    "maxnorm" : (0.5, 0.5), # 10**(maxnorm)
    "dropout_rate" : (0.25, 0.25), 
    "learning_rate" : (-5, -1), # 10**(learning_rate)
    "start_batch" : (11, 11), # 2**(start_batch)
    "batch_momentum" : (0.99, 0.99),
    "epsilon" : (-8, -8)
}

pbounds_medium = {
    "n_nodes_dense_1" : (300, 300),
    "n_nodes_dense_2" : (200, 200),
    "n_dense_1" : (1,1),
    "n_dense_2" : (2,2), 
    "n_nodes_lstm" : (25, 250),
    "n_lstm" : (1,1), 
    "maxnorm" : (0.5, 0.5), # 10**(maxnorm)
    "dropout_rate" : (0.0, 0.5), 
    "learning_rate" : (-5, -1), # 10**(learning_rate)
    "start_batch" : (11, 11), # 2**(start_batch)
    "batch_momentum" : (0.99, 0.99),
    "epsilon" : (-8, -8)
}

pbounds_full = {
    "n_nodes_dense_1" : (100, 500),
    "n_nodes_dense_2" : (25, 200),
    "n_dense_1" : (1,3),
    "n_dense_2" : (1,4),  
    "n_nodes_lstm" : (25, 150),
    "n_lstm" : (1,5), 
    "maxnorm" : (-1, 2), # 10**(maxnorm)
    "dropout_rate" : (0.0, 0.5), 
    "learning_rate" : (-5, -1), # 10**(learning_rate)
    "start_batch" : (7, 13), # 2**(start_batch)
    "batch_momentum" : (0.5, 0.999),
    "epsilon" : (-10, -6)
}

pbounds_fixed = {
    "n_nodes_dense_1" : (300, 300),
    "n_nodes_dense_2" : (200, 200),
    "n_dense_1" : (1,1),
    "n_dense_2" : (4,4),
    "n_nodes_lstm" : (100, 100),
    "n_lstm" : (3,3),
    "maxnorm" : (0.5, 0.5), # 10**(maxnorm)
    "dropout_rate" : (0.25, 0.25),
    "learning_rate" : (-3, -3), # 10**(learning_rate)
    "start_batch" : (9, 9.00001), # 2**(start_batch) # dumb hacky way to make sure the points aren't all considered the same
    "batch_momentum" : (0.99, 0.99),
    "epsilon" : (-8, -8)
}

pbounds_dict = {
    "pbounds_full" : pbounds_full,
    "pbounds_light" : pbounds_light,
    "pbounds_medium" : pbounds_medium,
    "pbounds_fixed" : pbounds_fixed 
}

starting_point = {
    "n_nodes_dense_1" : 300, 
    "n_nodes_dense_2" : 200, 
    "n_dense_1" : 1, 
    "n_dense_2" : 4, 
    "n_nodes_lstm" : 100, 
    "n_lstm" : 3, 
    "maxnorm" : 0.5, 
    "dropout_rate" : 0.25, 
    "learning_rate" : -3,
    "start_batch" : 10,
    "batch_momentum" : 0.99
}

optimizer = BayesianOptimization(
    f=auc,
    pbounds=pbounds_light,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

#official_log = "log_%s_%s.json" % (args.channel, args.tag)

#logger = JSONLogger(path=official_log)
#optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

xi = float(args.xi)

x = numpy.linspace(pbounds_light["learning_rate"][0], pbounds_light["learning_rate"][1], 1000).reshape(-1,1)

#optimizer.maximize(init_points=3, n_iter=0)

def maximize(optimizer, pbounds, n_probe, n_points, random, do_plot = False):
    optimizer.set_bounds(new_bounds = pbounds)
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    for i in range(n_points):
        if i % 3 == 0 or random:
            print("Probing a RANDOM point")
            optimizer.maximize(init_points=1, n_iter=0, alpha=float(args.alpha))
        else:
            next_point = optimizer.suggest(utility_function)
            print("Probing this point next: ", next_point)
            target = auc(**next_point)
            print("Found AUC to be: ", target)
            optimizer.register(params = next_point, target = target)
        if do_plot:
            plot_gp(optimizer,x)

#maximize(optimizer, pbounds_light, 0, 10, args.random, True)
maximize(optimizer, pbounds_medium, 0, args.n_points, args.random)
#maximize(optimizer, pbounds_full, 0, args.n_points, args.random)

"""
if args.no_buildup or args.fixed:
    optimizer.set_bounds(new_bounds = pbounds_full)
    if args.fixed:
        optimizer.set_bounds(new_bounds = pbounds_fixed)
    if args.no_buildup:
        args.n_points *= 3
    print(("Probing %d points per pbounds set" % (args.n_points)))
    optimizer.maximize(
        init_points = args.n_points if args.random else 0,
        n_iter = 0 if args.random else args.n_points,
        acq = "ucb", xi  = 0, kappa = 5,
    )

else:
    print(("Probing %d points per pbounds set" % (args.n_points)))

    optimizer.maximize(
            init_points = 3,#args.n_points if args.random else 0,
            n_iter = 0 if args.random else args.n_points,
            acq = "ucb", xi  = 0, kappa = 5,
    )

    x = numpy.linspace(pbounds_light["learning_rate"][0], pbounds_light["learning_rate"][1], 1000).reshape(-1,1)
    plot_gp(optimizer,x)

    optimizer.set_bounds(new_bounds = pbounds_medium)

    optimizer.maximize(
            init_points = args.n_points if args.random else 0,
            n_iter = 0 if args.random else args.n_points,
            acq = "ucb", xi  = 0, kappa = 5,
    )

    optimizer.set_bounds(new_bounds = pbounds_full)

    optimizer.maximize(
            init_points = args.n_points if args.random else 0,
            n_iter = 0 if args.random else args.n_points,
            acq = "ucb", xi  = 0, kappa = 5,
    )


for i, res in enumerate(optimizer.res):
    print(("Iteration {}: \n\t{}".format(i, res)))
"""
