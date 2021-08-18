# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:38:05 2021

@author: Anselmo Ramalho Pitombeira Neto
website: www.opl.ufc.br
e-mail: anselmo.pitombeira@ufc.br
"""
from os import path
import json
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from routing_challenge_final_submission import *


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


#%%OPEN FILES
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

##PATHS
path1 = path.join(BASE_DIR,'data/model_build_inputs/')
path2 = path.join(BASE_DIR,'data/model_build_outputs/')

##SPLIT BIG TIME DATA JSON
print("Split big json")
process_big_json(path1,path2)

##PREPARE DATA FOR APPLICATION OF OPTIMIZATION ALGORITHM
print("Data treatment")
n_sample = -1    ##For all samples, use n_sample = -1
n_initial_points = 10
rotas_id, data = data_treatment(path1,n_sample)

#%%BAYESIAN-OPTIMIZATION
#  
def my_function(teta1,teta2,teta3,teta4,teta5,teta6,teta7):
    
    """
    Interface with Bayesian optimization
    """
    
    teta = np.array([teta1,teta2,teta3,teta4,teta5,teta6,teta7])
#    teta = np.array([teta1,teta2])
    
    scores = compute_and_evaluate(rotas_id,path1,path2,data,teta)
    value = scores['submission_score']
    
    return -value ##BO module assumes maximization

bounds = {'teta1':[0.0,1.0],'teta2':[0.0,1.0],
          'teta3':[0.0, 1.0], 'teta4':[0.0, 1.0],
          'teta5':[0.0, 1.0], 'teta6':[0.0, 1.0],
          'teta7':[0.0, 1.0]}

optimizer = BayesianOptimization(
    f = None,
    pbounds=bounds,
    verbose=2, 
    random_state=42)
    
utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)


##Probe initial solutions
print("Probe a initial set of parameters values")
##
params1 = {'teta1':1.0,
           'teta2':0.0, 
           'teta3':0.0, 
           'teta4':0.0,
           'teta5':0.0,
           'teta6':0.0,
           'teta7':0.0}

target1 = my_function(**params1)
optimizer.register(params=params1, target=target1)
print(params1, target1)

with open(path2+'model.json','w') as fil:
    json.dump(optimizer.max,fil)

print("Random sample of initial points")

for it in range(1,n_initial_points):
    print("Initial point #",it)
    sampled_point = optimizer.space.random_sample()
    next_point = optimizer.space.array_to_params(sampled_point)
    target = my_function(**next_point)
    optimizer.register(params=next_point, target=target)
    print(target, next_point)
    with open(path2+'model.json','w') as fil:
        json.dump(optimizer.max,fil)

##Search for some time
print("Start loop")
for it in range(1,200):
    print("#",it)
    next_point = optimizer.suggest(utility)
    target = my_function(**next_point)
    optimizer.register(params=next_point, target=target)
    print(target, next_point)
    with open(path2+'model.json','w') as fil:
        json.dump(optimizer.max,fil)


