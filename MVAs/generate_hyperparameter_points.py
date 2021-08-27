import os,sys
import json
import numpy
import itertools

#h_params = ["max_depth", "eta", "min_child_weight", "subsample", "colsample_bytree"]
h_params = ["max_depth", "eta", "n_round"]

param_ranges = {
	"max_depth" : 		[4, 5, 6, 7, 8, 10, 12], 
	"eta" : 		[0.1, 0.2, 0.3],
	"n_round" :		[25, 50, 75, 100, 150],
	"min_child_weight" : 	[1],
	"subsample" : 		[1.0],
	"colsample_bytree" :	[1.0],
	"gamma" :		[0.0],
	"lambda" :		[1.0],
	"alpha" :		[1.0],
	#"objective" :		["binary:logistic"],
}

def product_dict(**kwargs):
  keys = list(kwargs.keys())
  vals = list(kwargs.values())
  for instance in itertools.product(*vals):
        yield dict(list(zip(keys, instance)))

hyperparameter_points = list(product_dict(**param_ranges))

with open("hyperparameter_points.json", "w") as f_out:
  f_out.write("{\n")
  for i in range(len(hyperparameter_points)):
    param_dict = hyperparameter_points[i]
    n_rounds = param_dict.pop("n_round")
    new_dict = { "params" : param_dict, "n_rounds" : n_rounds}
    f_out.write(' "%s" : %s%s \n' % (str(i), json.dumps(new_dict), "" if i == len(hyperparameter_points) - 1 else ","))
  f_out.write("}") 


#with open("hyperparameter_points.json", "w") as f:
#  idx = 0
#  n_params = len(h_params)
#  n_points = 1
#  for param in h_params:
#    n_points *= len(h_params[param]) 
#
#  points = []
#  for i in range(n_points):
#    points.append({})
#
#  while len(h_params) > 0:   
#    for param in h_params:
#      values = param_ranges[param]
#      for value in values:
        
     

 
