# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:01:12 2021

@author: Anselmo Ramalho Pitombeira Neto
website: www.opl.ufc.br
e-mail: anselmo.pitombeira@ufc.br
"""
import json
import ijson
import numpy as np
from math import ceil
from random import sample,seed,shuffle
import pandas as pd
from numba import njit
from multiprocessing import Pool
from datetime import datetime
from score import *

def identify_stations(route_data):
    rotas_id = list(route_data.keys())
    stations_dict = {}
    for r_id in rotas_id:
        rota_dict = route_data[r_id]

        for stop in rota_dict['stops']:
            if rota_dict['stops'][stop]['type'] == 'Station':
                station = stop
                break
        
        stations_dict[r_id] = station
    
    return stations_dict

@njit(fastmath=False)
def greedy_rollout(depot,
                   initial_stop,
                   stops,
                   route_clock,
                   travel_time_data,
                   serv_time_data,
                   pack_dim_data,
                   start_time_windows,
                   end_time_windows,
                   latest_due_date,
                   teta=None):
    
    """
    Build a tail route from a current stop by following greedily to the next
    stop with lowest cost.
    
    depot: initial stop in the route (station) in which the rollout is being computed
    initial_stop: stop from which the rollout is started
    stops: list of remaining stops in the route
    teta: Parameter vector
    """
    ##Penalty for not fulfilling time window
    window_penalty = travel_time_data.sum()
    ##Clock of the route
    rollout_clock = route_clock
    
    current_stop = initial_stop
    route = []
    route.append(current_stop)
    remaining_stops = list(stops)
    remaining_stops.remove(current_stop)

    rollout_cost = 0
    while len(remaining_stops) > 0:
        
        best_cost = np.inf
        for stop in remaining_stops:
            
            cost = cost_function(current_stop,
                                      stop,
                                      travel_time_data,
                                      serv_time_data,
                                      pack_dim_data,
                                      start_time_windows,
                                      end_time_windows,
                                      rollout_clock,
                                      latest_due_date,
                                      teta)
            
            if cost < best_cost:
                best_cost = cost
                next_stop = stop
        
        ##Take travel time between stops
        time_between_stops = travel_time_data[current_stop,next_stop]
        ##Update clock
        rollout_clock+=time_between_stops
        ##Update current city
        current_stop = next_stop
        route.append(current_stop)
        remaining_stops.remove(current_stop)
        ##Update rollout cost
        rollout_cost+=best_cost
        ##Check time window
        start_window = start_time_windows[current_stop]
        end_window = end_time_windows[current_stop]
        
        ##Apply windows penalty
        if start_window > -1:
            if rollout_clock < start_window:
                rollout_cost+= teta[-1]*window_penalty
            
        if end_window > -1:
            if rollout_clock > end_window:
                rollout_cost+= teta[-1]*window_penalty
            
        ##Update clock after serving stop
        rollout_clock+=serv_time_data[current_stop]

    return rollout_cost


@njit(fastmath=False)
def rollout_policy(stops,
                   initial_stop,
                   departure_time,
                   travel_time_data,
                   serv_time_data,
                   pack_dim_data,
                   start_time_windows,
                   end_time_windows,
                   teta=None):    
    
    """
    Build a route by following a rollout policy with a greedy
    base policy.
    
    teta: Parameter vector
    """
    
    current_stop = initial_stop
    route = []
    route.append(initial_stop)
    remaining_stops = list(stops)
    remaining_stops.remove(initial_stop)
    
    ##This is a clock variable which determines the estimated time
    ##when the vehicle arrives at a stop
    ##It is updated at every stop
    route_clock = departure_time
    latest_due_date = np.max(end_time_windows)
    
    
    while len(remaining_stops) > 0:
        
        best_cost = np.inf
        for stop in remaining_stops:
            
            ##This is the myopic cost from current stop to candidate next stop
            myopic_cost = cost_function(current_stop,
                                        stop,
                                        travel_time_data,
                                        serv_time_data,
                                        pack_dim_data,
                                        start_time_windows,
                                        end_time_windows,
                                        route_clock,
                                        latest_due_date,
                                        teta)
            
            trial_travel_time = travel_time_data[current_stop,stop]
            trial_serv_time = serv_time_data[stop]
            ##This is a trial clock in case stop is chosen as next stop
            trial_route_clock = route_clock+trial_travel_time+trial_serv_time
            
            ##This is the rollout cost from starting at the candidate stop and acting greedily
            rollout_cost = greedy_rollout(initial_stop,
                                           stop,
                                           remaining_stops,
                                           trial_route_clock,
                                           travel_time_data,
                                           serv_time_data,
                                           pack_dim_data,
                                           start_time_windows,
                                           end_time_windows,
                                           latest_due_date,
                                           teta)

            cost = myopic_cost+teta[5]*rollout_cost
            
            if cost < best_cost:
                best_cost = cost
                next_stop = stop
                
        ##Take travel time between stops
        time_between_stops = travel_time_data[current_stop,next_stop]
        ##Update clock
        route_clock+=time_between_stops
        ##Update current stop
        current_stop = next_stop
        route.append(current_stop)
        remaining_stops.remove(current_stop)
        ##Update clock after serving stop
        route_clock+=serv_time_data[current_stop]
        
    route.append(initial_stop)
    return route

@njit(fastmath=True)
def cost_function(a,b,
                  travel_time_data,
                  serv_time_data,
                  pack_dim_data,
                  start_time_window_data,
                  end_time_window_data,
                  clock,
                  latest_due_date,
                  teta):
    
    """
    a - origin stop
    b - destination stop
    end_time_window_data - array of time windows endtime
    clock - current time in route
    """
    
    travel_time = travel_time_data[a, b]
    serv_time = serv_time_data[b]
    pack_dim = pack_dim_data[b]
    end_time_window = end_time_window_data[b]   ##Endtime window of destination stop. Treated as a due date.
    start_time_window = start_time_window_data[b]   ##starttime window of destination stop.
    
    ##Compute time interval from current clock to endtime window
    ##The logic here is that the longer the time inverval, less
    ##desirable is to go to that next stop
    ##On the other hand, the smallest this interval, more desirable
    
    ##Compute time to start of time window
    if start_time_window > -1: ##Start time window exists. -1 indicates no start time_window
        time_to_start = max(0,start_time_window-(clock+travel_time))
    else:    ##No time window
        time_to_start = 0
        
    #Compute time left to end of time window
    if end_time_window > -1: ##End time window exists. -1 indicates no end time_window
        time_to_end = end_time_window-(clock+travel_time)    ##Notice that it may be negative
    else:    ##No time window, interval is attributed to largest interval
        time_to_end = max(0,latest_due_date-(clock+travel_time))
        
    
    cost = teta[0]*travel_time+teta[1]*serv_time+teta[2]*pack_dim+teta[3]*time_to_start+teta[4]*time_to_end
    
    return cost


def compute_route(args):
    
    """
    This function computes a route
    route_id: route hash
    path: path for json files
    data: route data dictionary
    teta: parameter vector
    """
    
    # route_id,station_id,path = args
    route_id,path,data,teta = args
    
    ##General route data
    station_id = data['station']
    departure_time = data['departure_timestamp']   ##POSIX time
    
    ##read travel time matrix
    with open(path+route_id+'.json') as fil2:
        tempos_rota = json.load(fil2)
    
    ##Extrai matriz de tempos usando o Pandas
    df = pd.DataFrame(tempos_rota)
    time_matrix = df.values
    stops_map = dict(zip(df.index.values,range(len(df.index.values))))
    inverted_stops_map = {value: key for key, value in stops_map.items()}
    station_index = stops_map[station_id]
    stops = list(stops_map.values())
    travel_time_data = time_matrix
    
    ##Assemble stops data arrays
    serv_time_data = []
    pack_dim_data = []
    start_time_windows = []
    end_time_windows = []
    
    for stop in data['stops']:
        serv_time_data.append(data['stops'][stop]['service_time'])
        pack_dim_data.append(data['stops'][stop]['max_dim'])
        start_time_windows.append(data['stops'][stop]['start_time_window'])
        end_time_windows.append(data['stops'][stop]['end_time_window'])
    
    serv_time_data = np.array(serv_time_data)
    pack_dim_data = np.array(pack_dim_data)
    start_time_windows = np.array(start_time_windows)
    end_time_windows = np.array(end_time_windows)

    rota = rollout_policy(stops,
                          station_index,
                          departure_time,
                          travel_time_data,
                          serv_time_data,
                          pack_dim_data,
                          start_time_windows,
                          end_time_windows,
                          teta)

    
    ##Exclude initial station
    rota = rota[:-1]
    
    nomes_rota = []
    for i in rota:
        nomes_rota.append(inverted_stops_map[i])
    
    ##Returns a tuple with route hash in a list with stop sequence
    return (route_id, nomes_rota)


def gen_proposed_route_json(proposed_routes):
    
    formatted_dict = {}
    
    for route in proposed_routes:
        formatted_dict[route] = {}
        formatted_dict[route]['proposed'] = {}
        
        k = 0
        for stop in proposed_routes[route]:
            stop_string = stop
            formatted_dict[route]['proposed'][stop] = k
            k+=1
            
    return formatted_dict
        
    
def compute_all_routes(route_ids,path,data,teta):
    
    """
    Compute all routes in parallel
    route_ids: list with route hashes
    
    """
    
    ##Monta lista de argumentos para aplicar o map do POOL
    args=[]
    # teta = np.ones(3)
    for r_id in route_ids:
        args.append([r_id,path,data[r_id],teta])
    # print("Inicia Pool de processos")
    # n_processes = 4
    with Pool() as p:
        rotas= p.map(compute_route, args)

    ##Transforma em dicionário
    rotas = dict(rotas)
    
    ##Gera o dicionário formatado (pode ser convertido diretament no json)
    rotas_formatadas = gen_proposed_route_json(rotas)
    
    return rotas_formatadas

    
def my_evaluate(actual_routes_json,submission_json,invalid_scores_json,path,**kwargs):
    '''
    Calculates score for a submission.
    
    This is a modification of the original implementation of the evaluate function
    so that we load the correspoding cost matrices of each route from separate
    jsons, instead of the full json.

    Parameters
    ----------
    actual_routes_json : str
        filepath of JSON of actual routes.
    submission_json : str
        filepath of JSON of participant-created routes.
    invalid_scores_json : str
        filepath of JSON of scores assigned to routes if they are invalid.
    **kwargs :
        Inputs placed in output. Intended for testing_time_seconds and
        training_time_seconds

    Returns
    -------
    scores : dict
        Dictionary containing submission score, individual route scores, feasibility
        of routes, and kwargs.

    '''
    actual_routes=read_json_data(actual_routes_json)
    good_format(actual_routes,'actual',actual_routes_json)
    submission=read_json_data(submission_json)
    good_format(submission,'proposed',submission_json)
    # cost_matrices=read_json_data(cost_matrices_json)
    # good_format(cost_matrices,'costs',cost_matrices_json)
    invalid_scores=read_json_data(invalid_scores_json)
    good_format(invalid_scores,'invalids',invalid_scores_json)
    scores={'submission_score':'x','route_scores':{},'route_feasibility':{}}
    for kwarg in kwargs:
        scores[kwarg]=kwargs[kwarg]
    k = 1
    for route in actual_routes:
        print("Evaluating route ", k)
        if route not in submission:
            scores['route_scores'][route]=invalid_scores[route]
            scores['route_feasibility'][route]=False
        else:
            actual_dict=actual_routes[route]
            actual=route2list(actual_dict)
            try:
                sub_dict=submission[route]
                sub=route2list(sub_dict)
            except:
                scores['route_scores'][route]=invalid_scores[route]
                scores['route_feasibility'][route]=False
            else:
                if isinvalid(actual,sub):
                    scores['route_scores'][route]=invalid_scores[route]
                    scores['route_feasibility'][route]=False
                else:
                     with open(path+route+'.json') as fil2:
                         cost_mat = json.load(fil2)    
                     
                     # cost_mat=cost_matrices[route]
                     scores['route_scores'][route]=score(actual,sub,cost_mat)
                     scores['route_feasibility'][route]=True
        k+=1
    submission_score=np.mean(list(scores['route_scores'].values()))
    scores['submission_score']=submission_score
    return scores


def my_parallel_evaluate(actual_routes_json,
                         submission_json,
                         invalid_scores_json,
                         path):
    '''
    Calculates score for a submission (parallel version).
    
    This is a modification of the original implementation of the evaluate function
    so that we load the correspoding cost matrices of each route from separate
    jsons, instead of the full json.

    Parameters
    ----------
    actual_routes_json : str
        filepath of JSON of actual routes.
    submission_json : str
        filepath of JSON of participant-created routes.
    invalid_scores_json : str
        filepath of JSON of scores assigned to routes if they are invalid.
    **kwargs :
        Inputs placed in output. Intended for testing_time_seconds and
        training_time_seconds

    Returns
    -------
    scores : dict
        Dictionary containing submission score, individual route scores, feasibility
        of routes, and kwargs.

    '''
    actual_routes=read_json_data(actual_routes_json)
    good_format(actual_routes,'actual',actual_routes_json)
    submission=read_json_data(submission_json)
    good_format(submission,'proposed',submission_json)
    # cost_matrices=read_json_data(cost_matrices_json)
    # good_format(cost_matrices,'costs',cost_matrices_json)
    invalid_scores=read_json_data(invalid_scores_json)
    good_format(invalid_scores,'invalids',invalid_scores_json)
    # scores={'submission_score':'x','route_scores':{},'route_feasibility':{}}
    # for kwarg in kwargs:
    #     scores[kwarg]=kwargs[kwarg]
    
    ##Argument list
    args=[]
    # for route in actual_routes:
    for route in submission:
        args.append([route,actual_routes,submission,path])

    ##Parallelize
    with Pool() as p:
        scores_list= p.map(evaluate_single_route, args)
        
    ##Merge dicts
    # scores = {}
    # for d in scores_list:
    #     scores.update(d)
    
    # submission_score=np.mean(list(scores['route_scores'].values()))
    # scores['submission_score']=submission_score
    return scores_list


def evaluate_single_route(args):
    
    route,actual_routes,submission,path = args
    
    scores={'submission_score':'x','route_scores':{},'route_feasibility':{}}
    
    if route not in submission:
            scores['route_scores'][route]=invalid_scores[route]
            scores['route_feasibility'][route]=False
    else:
        actual_dict=actual_routes[route]
        actual=route2list(actual_dict)
        try:
            sub_dict=submission[route]
            sub=route2list(sub_dict)
        except:
            scores['route_scores'][route]=invalid_scores[route]
            scores['route_feasibility'][route]=False
        else:
            if isinvalid(actual,sub):
                scores['route_scores'][route]=invalid_scores[route]
                scores['route_feasibility'][route]=False
            else:
                with open(path+route+'.json') as fil2:
                    cost_mat = json.load(fil2)
                    
                actual = tuple(actual)
                sub = tuple(sub)
                 
                # cost_mat=cost_matrices[route]
                scores['route_scores'][route]=score(actual,sub,cost_mat)
                scores['route_feasibility'][route]=True
                     
    return scores
    
    
def create_scores_dict(scores_list):
    scores = {'submission_score':'x','route_scores':{},'route_feasibility':{}}

    for dic in scores_list:
        keys = list(dic['route_scores'].keys())
        values1 = list(dic['route_scores'].values())
        values2 = list(dic['route_feasibility'].values())
        route = keys[0]    ##route name
        route_score = values1[0]
        route_feas = values2[0]
        scores['route_scores'][route] = route_score
        scores['route_feasibility'][route] = route_feas
        
    submission_score=np.mean(list(scores['route_scores'].values()))
    scores['submission_score']=submission_score
    
    return scores


def compute_and_evaluate(rotas_id,path,path2,data,teta):
    
    """
    This function computes all routes and evaluate them after that.
    """
    
    print("Compute routes")
    rotas_formatadas = compute_all_routes(rotas_id,path2,data,teta)
    print("Evaluate routes")
    
    with open(path2+"submission.json",'w') as fil:
        json.dump(rotas_formatadas,fil)
        
    scores_list = my_parallel_evaluate(path+'actual_sequences.json',
                                       path2+'submission.json',
                                       path+'invalid_sequence_scores.json',
                                       path2)
    
    
    scores = create_scores_dict(scores_list)
    
    return scores

def apply_model(rotas_id,path,path2,data,teta):
    
    """
    This function computes all routes and saves a submission.
    """
    print("Compute routes")
    rotas_formatadas = compute_all_routes(rotas_id,path2,data,teta)
    
    with open(path2+"proposed_sequences.json",'w') as fil:
        json.dump(rotas_formatadas,fil)
        

        

    
#def data_treatment(path):
#
#    with open(path+"route_data.json",'rb') as fil:
#        # route_data = pd.read_json(fil,orient='index')
#        route_data = json.load(fil)
#        
#    with open(path+"package_data.json",'rb') as fil:
#        # route_data = pd.read_json(fil,orient='index')
#        package_data = json.load(fil)
#        
#    data = {}    ##Dicionário que reúne os dados relevantes por rota
#    
#    for route in route_data:
#        data[route] = {}
#        data[route]['stops'] = {}
#        dep_time = route_data[route]['date_YYYY_MM_DD']+" "+route_data[route]['departure_time_utc']
#        # data[route]['departure_time'] = route_data[route]['departure_time_utc']
#        #data[route]['date'] = route_data[route]['date_YYYY_MM_DD']
#        data[route]['departure_time'] = dep_time
#        dep_time_obj = datetime.strptime(dep_time, '%Y-%m-%d %H:%M:%S')
#        data[route]['departure_timestamp'] = dep_time_obj.timestamp()    ##Posix time utc
#        for stop in package_data[route]:
#            data[route]['stops'][stop] = {}
#
#    ##Extra os tempos de serviço dos pacotes
#    service_times = {}
#    # all_service_times = []    ##Guarda todos os tempos de serviço para fins de exploração de dados
#
#    for route in package_data:
#        service_times[route] = {}
#        for stop in package_data[route]:
#            service_times[route][stop] = 0    ##Inicia em 0. Caso não haja package, será mantido em 0
#            total_s_time = 0
#            for package in package_data[route][stop]:    
#                s_time = package_data[route][stop][package]['planned_service_time_seconds']
#                total_s_time+=s_time
#                # all_service_times.append(total_s_time)
#            service_times[route][stop] = total_s_time
#            data[route]['stops'][stop]['service_time'] = total_s_time
#                
#    # all_service_times = np.array(all_service_times)
#    
#    ##Extrai dimensão máxima dos pacotes
#    max_dimentions = {}
#    ##all_max_dims = []    ##Guarda todos os tempos de serviço para fins de exploração de dados
#
#    for route in package_data:
#        max_dimentions[route] = {}
#        for stop in package_data[route]:
#            max_dimentions[route][stop] = 0    ##Inicia em 0. Caso não haja package, será mantido em 0
#            max_dim = 0
#            for package in package_data[route][stop]:    
#                h_dim = package_data[route][stop][package]['dimensions']['height_cm']
#                d_dim = package_data[route][stop][package]['dimensions']['depth_cm']
#                w_dim = package_data[route][stop][package]['dimensions']['width_cm']
#                ##all_max_dims.append(max(h_dim,d_dim,w_dim))
#                max_dim = max(h_dim,d_dim,w_dim,max_dim)
#                
#            max_dimentions[route][stop] = max_dim
#            data[route]['stops'][stop]['max_dim'] = max_dim
#
#    for route in package_data:
#        for stop in package_data[route]:
#            flag = False
#            start_time_utc = -1
#            end_time_utc = -1
#            for package in package_data[route][stop]:
#                time_window = package_data[route][stop][package]['time_window']
#                
#                # print("time window = ", time_window)
#                
#                try: np.isnan(time_window["start_time_utc"])
#                
#                except:
#                    flag = True
#                    time_obj = datetime.strptime(time_window["start_time_utc"], '%Y-%m-%d %H:%M:%S')
#                    start_time_utc = time_obj.timestamp()   ##POSIX time
#                    
#                    
#                    
#                try: np.isnan(time_window["end_time_utc"])
#            
#                except:
#                    flag = True                    
#                    time_obj = datetime.strptime(time_window["end_time_utc"], '%Y-%m-%d %H:%M:%S')
#                    end_time_utc = time_obj.timestamp()   ##POSIX time
#                    
#                if flag == True:
#                    break
#            # print("route = ",route)
#            # print("stop =", stop)
#            data[route]['stops'][stop]['start_time_window'] = start_time_utc
#            # print('start_time_window', start_time_utc)
#            data[route]['stops'][stop]['end_time_window'] = end_time_utc
#            # print('end_time_window', end_time_utc)
#                
#    ##all_max_dims = np.array(all_max_dims)
#
#    rotas_id = list(route_data.keys())
#    n_routes = len(rotas_id)
#    
#    ##Extrai stations de cada rota (depot)
#    stations_dict = identify_stations(route_data)
#    
#    stations_id = []
#    for r_id in rotas_id:
#        stations_id.append(stations_dict[r_id])
#        data[r_id]['station'] = stations_dict[r_id]
#        
#        
###%%FILTER ROUTES WITH HIGH SCORE
#    rotas_high_score = []
#    for r_id in rotas_id:
#        if route_data[r_id]['route_score'] == 'High':
#            rotas_high_score.append(r_id)
#        
#    rotas_id = rotas_high_score
#
#    return rotas_id, data

def data_treatment(path,n_sample=0):

    with open(path+"route_data.json",'rb') as fil:
        # route_data = pd.read_json(fil,orient='index')
        route_data = json.load(fil)
        
    with open(path+"package_data.json",'rb') as fil:
        # route_data = pd.read_json(fil,orient='index')
        package_data = json.load(fil)
        

    data = {}    ##Data dictionary
    
    for route in route_data:
        data[route] = {}
        data[route]['stops'] = {}
        dep_time = route_data[route]['date_YYYY_MM_DD']+" "+route_data[route]['departure_time_utc']
        # data[route]['departure_time'] = route_data[route]['departure_time_utc']
        #data[route]['date'] = route_data[route]['date_YYYY_MM_DD']
        data[route]['departure_time'] = dep_time
        dep_time_obj = datetime.strptime(dep_time, '%Y-%m-%d %H:%M:%S')
        data[route]['departure_timestamp'] = dep_time_obj.timestamp()    ##Posix time utc
        for stop in package_data[route]:
            data[route]['stops'][stop] = {}


    ##Extract service times
    service_times = {}
    # all_service_times = []    ##Guarda todos os tempos de serviço para fins de exploração de dados

    for route in package_data:
        service_times[route] = {}
        for stop in package_data[route]:
            service_times[route][stop] = 0    ##Inicia em 0. Caso não haja package, será mantido em 0
            total_s_time = 0
            for package in package_data[route][stop]:    
                s_time = package_data[route][stop][package]['planned_service_time_seconds']
                total_s_time+=s_time
                # all_service_times.append(total_s_time)
            service_times[route][stop] = total_s_time
            data[route]['stops'][stop]['service_time'] = total_s_time
                
    
    ##extract largest dimension of packages
    max_dimentions = {}

    for route in package_data:
        max_dimentions[route] = {}
        for stop in package_data[route]:
            max_dimentions[route][stop] = 0    
            max_dim = 0
            for package in package_data[route][stop]:    
                h_dim = package_data[route][stop][package]['dimensions']['height_cm']
                d_dim = package_data[route][stop][package]['dimensions']['depth_cm']
                w_dim = package_data[route][stop][package]['dimensions']['width_cm']
                ##all_max_dims.append(max(h_dim,d_dim,w_dim))
                max_dim = max(h_dim,d_dim,w_dim,max_dim)
                
            max_dimentions[route][stop] = max_dim
            data[route]['stops'][stop]['max_dim'] = max_dim

    for route in package_data:
        for stop in package_data[route]:
            flag = False
            start_time_utc = -1
            end_time_utc = -1
            for package in package_data[route][stop]:
                time_window = package_data[route][stop][package]['time_window']
                
                
                try: np.isnan(time_window["start_time_utc"])
                
                except:
                    flag = True
                    time_obj = datetime.strptime(time_window["start_time_utc"], '%Y-%m-%d %H:%M:%S')
                    start_time_utc = time_obj.timestamp()   ##POSIX time
                    
                    
                    
                try: np.isnan(time_window["end_time_utc"])
            
                except:
                    flag = True                    
                    time_obj = datetime.strptime(time_window["end_time_utc"], '%Y-%m-%d %H:%M:%S')
                    end_time_utc = time_obj.timestamp()   ##POSIX time
                    
                if flag == True:
                    break
            # print("route = ",route)
            # print("stop =", stop)
            data[route]['stops'][stop]['start_time_window'] = start_time_utc
            # print('start_time_window', start_time_utc)
            data[route]['stops'][stop]['end_time_window'] = end_time_utc
            # print('end_time_window', end_time_utc)
                
    ##all_max_dims = np.array(all_max_dims)
    
    rotas_id = list(route_data.keys())
    n_routes = len(rotas_id)
    
    ##Extrai stations de cada rota (depot)
    stations_dict = identify_stations(route_data)
    
    stations_id = []
    for r_id in rotas_id:
        stations_id.append(stations_dict[r_id])
        data[r_id]['station'] = stations_dict[r_id]
        
        
##%%FILTER ROUTES WITH HIGH SCORE
    rotas_high_score = []
    for r_id in rotas_id:
        if route_data[r_id]['route_score'] == 'High':
            rotas_high_score.append(r_id)
        
    rotas_id = rotas_high_score
    
    ##Sample a set of routes
    if n_sample>0:
        print("Sample n = ", n_sample, " routes")
        seed(39)
        rotas_id = sample(rotas_id, n_sample)

    return rotas_id, data



def data_treatment_apply(path):

    with open(path+"new_route_data.json",'rb') as fil:
        # route_data = pd.read_json(fil,orient='index')
        route_data = json.load(fil)
        
    with open(path+"new_package_data.json",'rb') as fil:
        # route_data = pd.read_json(fil,orient='index')
        package_data = json.load(fil)
        
    data = {}    ##Dicionário que reúne os dados relevantes por rota
    
    for route in route_data:
        data[route] = {}
        data[route]['stops'] = {}
        dep_time = route_data[route]['date_YYYY_MM_DD']+" "+route_data[route]['departure_time_utc']
        # data[route]['departure_time'] = route_data[route]['departure_time_utc']
        #data[route]['date'] = route_data[route]['date_YYYY_MM_DD']
        data[route]['departure_time'] = dep_time
        dep_time_obj = datetime.strptime(dep_time, '%Y-%m-%d %H:%M:%S')
        data[route]['departure_timestamp'] = dep_time_obj.timestamp()    ##Posix time utc
        for stop in package_data[route]:
            data[route]['stops'][stop] = {}

    ##Extra os tempos de serviço dos pacotes
    service_times = {}
    # all_service_times = []    ##Guarda todos os tempos de serviço para fins de exploração de dados

    for route in package_data:
        service_times[route] = {}
        for stop in package_data[route]:
            service_times[route][stop] = 0    ##Inicia em 0. Caso não haja package, será mantido em 0
            total_s_time = 0
            for package in package_data[route][stop]:    
                s_time = package_data[route][stop][package]['planned_service_time_seconds']
                total_s_time+=s_time
                # all_service_times.append(total_s_time)
            service_times[route][stop] = total_s_time
            data[route]['stops'][stop]['service_time'] = total_s_time
                
    # all_service_times = np.array(all_service_times)
    
    ##Extrai dimensão máxima dos pacotes
    max_dimentions = {}
    ##all_max_dims = []    ##Guarda todos os tempos de serviço para fins de exploração de dados

    for route in package_data:
        max_dimentions[route] = {}
        for stop in package_data[route]:
            max_dimentions[route][stop] = 0    ##Inicia em 0. Caso não haja package, será mantido em 0
            max_dim = 0
            for package in package_data[route][stop]:    
                h_dim = package_data[route][stop][package]['dimensions']['height_cm']
                d_dim = package_data[route][stop][package]['dimensions']['depth_cm']
                w_dim = package_data[route][stop][package]['dimensions']['width_cm']
                ##all_max_dims.append(max(h_dim,d_dim,w_dim))
                max_dim = max(h_dim,d_dim,w_dim,max_dim)
                
            max_dimentions[route][stop] = max_dim
            data[route]['stops'][stop]['max_dim'] = max_dim

    for route in package_data:
        for stop in package_data[route]:
            flag = False
            start_time_utc = -1
            end_time_utc = -1
            for package in package_data[route][stop]:
                time_window = package_data[route][stop][package]['time_window']
                
                # print("time window = ", time_window)
                
                try: np.isnan(time_window["start_time_utc"])
                
                except:
                    flag = True
                    time_obj = datetime.strptime(time_window["start_time_utc"], '%Y-%m-%d %H:%M:%S')
                    start_time_utc = time_obj.timestamp()   ##POSIX time
                    
                    
                    
                try: np.isnan(time_window["end_time_utc"])
            
                except:
                    flag = True                    
                    time_obj = datetime.strptime(time_window["end_time_utc"], '%Y-%m-%d %H:%M:%S')
                    end_time_utc = time_obj.timestamp()   ##POSIX time
                    
                if flag == True:
                    break
            # print("route = ",route)
            # print("stop =", stop)
            data[route]['stops'][stop]['start_time_window'] = start_time_utc
            # print('start_time_window', start_time_utc)
            data[route]['stops'][stop]['end_time_window'] = end_time_utc
            # print('end_time_window', end_time_utc)
                
    ##all_max_dims = np.array(all_max_dims)
    
    rotas_id = list(route_data.keys())
    n_routes = len(rotas_id)
    
    ##Extrai stations de cada rota (depot)
    stations_dict = identify_stations(route_data)
    
    stations_id = []
    for r_id in rotas_id:
        stations_id.append(stations_dict[r_id])
        data[r_id]['station'] = stations_dict[r_id]
        
    return rotas_id, data

def train_test_split(routes_id):
    
    n = len(routes_id)
    shuffle(routes_id)    ##Shuffle list inplace
    train = routes_id[:ceil(0.7*n)]
    test = routes_id[ceil(0.7*n):]
    
    return train, test


def process_big_json(path,path2):
    

    with open(path+"route_data.json",'rb') as fil:
        # route_data = pd.read_json(fil,orient='index')
        route_data = json.load(fil)
        

    rotas = list(route_data.keys())
    n_rotas = len(rotas)

    fil= open(path+"travel_times.json",'rb')
    parser = ijson.parse(fil, use_float=True) 


    for i in range(n_rotas):
        flag = 0
        comma_flag = False
        while True:
            if flag == 0:
                while True:
                    prefix, event,value = next(parser)
                    # print("Prefixo =", prefix)
                    # print("Evento =",event)
                    # print("Value = ", value)
                    if prefix != "":
                        flag = 1
                        route_ID = prefix
                        fil2 = open(path2+route_ID+'.json', 'w')
                        fil2.write('{\n')
                        break
            else:
                prefix, event,value = next(parser)
                if prefix == "":
                    fil2.close()
                    break
                if event == 'start_map':
                    fil2.write('{\n')
                    comma_flag = False
                elif event == 'end_map':
                    fil2.write('}\n')
                elif event == 'map_key':
                    if comma_flag == False:
                        fil2.write('"'+value+'"'+':')
                        comma_flag = True
                    else:
                        fil2.write(',"'+value+'"'+':')
                elif event == 'number':
                    fil2.write(str(value))
                    #fil2.write(',')
                
    fil.close()
    

def process_big_json_apply(path,path2):
    

    with open(path+"new_route_data.json",'rb') as fil:
        # route_data = pd.read_json(fil,orient='index')
        route_data = json.load(fil)
        

    rotas = list(route_data.keys())
    n_rotas = len(rotas)

    fil= open(path+"new_travel_times.json",'rb')
    parser = ijson.parse(fil, use_float=True) 


    for i in range(n_rotas):
        flag = 0
        comma_flag = False
        while True:
            if flag == 0:
                while True:
                    prefix, event,value = next(parser)
                    # print("Prefixo =", prefix)
                    # print("Evento =",event)
                    # print("Value = ", value)
                    if prefix != "":
                        flag = 1
                        route_ID = prefix
                        fil2 = open(path2+route_ID+'.json', 'w')
                        fil2.write('{\n')
                        break
            else:
                prefix, event,value = next(parser)
                if prefix == "":
                    fil2.close()
                    break
                if event == 'start_map':
                    fil2.write('{\n')
                    comma_flag = False
                elif event == 'end_map':
                    fil2.write('}\n')
                elif event == 'map_key':
                    if comma_flag == False:
                        fil2.write('"'+value+'"'+':')
                        comma_flag = True
                    else:
                        fil2.write(',"'+value+'"'+':')
                elif event == 'number':
                    fil2.write(str(value))
                    #fil2.write(',')
                
    fil.close()


