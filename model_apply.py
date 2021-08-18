from os import path
import json
import numpy as np
from routing_challenge_final_submission import *

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Model Build output
model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')

with open(model_path, newline='') as in_file:
    model_build_out = json.load(in_file)
    
    
teta = []
teta.append(model_build_out['params']['teta1'])
teta.append(model_build_out['params']['teta2'])
teta.append(model_build_out['params']['teta3'])
teta.append(model_build_out['params']['teta4'])
teta.append(model_build_out['params']['teta5'])
teta.append(model_build_out['params']['teta6'])
teta.append(model_build_out['params']['teta7'])

teta = np.array(teta)
    
path1 = path.join(BASE_DIR,'data/model_apply_inputs/')
path2 = path.join(BASE_DIR,'data/model_apply_outputs/')

print("Split big json")
process_big_json_apply(path1,path2)

print("Data treatment")
rotas_id, data = data_treatment_apply(path1)

print("Apply model")
apply_model(rotas_id,path1,path2,data,teta)

