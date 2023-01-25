# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:56:18 2022

@author: S79158
"""


# =============================================================================
#                               Packages
# =============================================================================

import Optimisation
import random
import numpy as np 
import concurrent.futures
from itertools import combinations
from itertools import chain
import pandas as pd 


def AUC_PC_pickle(y, X, Pred_Formula, Eval_Metric, var_interet, N_coalition_sampled = 1000, seed = 42):

    random.seed(seed) # Fix the seed 
    
    p = np.size(X,1)  # Number of variables 
    
    N = np.size(X,0)  # Sample size 
    
    if Eval_Metric == ["AUC"]: 
        
        liste = list(range(p))       # list with numbers from 0 to p-1
        liste = [x+1 for x in liste] # To move index from (0,p-1) to (1,p) / list from 1 to p 
        
        #variable = liste.copy()
        
        # Code specific to the measure 
        
        delta_n3 = 2*np.mean(1-y)*np.mean(y) # Denominator of the AUC 
        
        shuffle = np.array(range(N)) # list with numbers from 0 to (N-1) 
        random.shuffle(shuffle)      # Shuffle the list with numbers from 0 to (N-1)
        
        X_shuffle = X[shuffle,:]     # Retrieve the feature values according to
                                     # order given by the shuffled list
        
        # =============================================================================
        variable = liste.copy()      # Copy the list with numbers from 1 to p
        
        variable.remove(var_interet+1) # Remove the variable of interest from "variable"
        
        # List of every possible coalitions
        combination_list = [list(combinations(variable, combination)) for combination in range(len(variable)+1)]
        combination_list = list(chain.from_iterable(combination_list))
                
        #N_combination_list = len(combination_list)
        
        combination_list_sampled = random.sample(combination_list, N_coalition_sampled) # sample without replacement (no need in reality)
        
        print("Nombre de coalitions totales:",len(combination_list))
        print("Part du nombre de coalitions:",round(100*(N_coalition_sampled/len(combination_list)),4),"%")
        # =============================================================================

        weight = np.zeros(shape=(N_coalition_sampled,1))          # column vector with N_coalition_sampled 0 elements
        Metric = np.zeros(shape=(N_coalition_sampled))            # vector with N_coalition_sampled 0 elements
        Metric_ind = np.zeros(shape=(N_coalition_sampled,len(y))) # Matrix N_coalition_sampled x len(y) filled with 0 

        Benchmark_ind = np.zeros(shape=(N_coalition_sampled,len(y))) # Matrix N_coalition_sampled x len(y) filled with 0 
        EM_ind = np.zeros(shape=(N_coalition_sampled,len(y)))        # Matrix N_coalition_sampled x len(y) filled with 0 
        
        ### Ajout pour calculer avec variable d'intérêt
        
        Metric_vinteret = np.zeros(shape=(N_coalition_sampled))   # vector with N_coalition_sampled 0 elements
        Metric_ind_vinteret = np.zeros(shape=(N_coalition_sampled,len(y))) # Matrix N_coalition_sampled x len(y) filled with 0 
        
        benchmark = np.zeros(shape=(N_coalition_sampled)) # vector with N_coalition_sampled 0 elements
        sample_EM = np.zeros(shape=(N_coalition_sampled)) # vector with N_coalition_sampled 0 elements
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
          
            results = [executor.submit(Optimisation.loop_AUC_pickle,s,combination_list_sampled,p,X,y,Pred_Formula,delta_n3,Metric,Metric_ind,N,X_shuffle,Metric_vinteret,Metric_ind_vinteret,var_interet) for s in list(range(N_coalition_sampled))]
        
            for result in concurrent.futures.as_completed(results):
                s = result.result()[0]
                print("Coalition (results)",s)
                weight[s,:] = result.result()[1]
                Metric[s] = result.result()[2]
                Metric_ind[s,:] = result.result()[3]
                Metric_vinteret[s] = result.result()[4]
                Metric_ind_vinteret[s,:] = result.result()[5]
                benchmark[s] = result.result()[6]
                sample_EM[s] = result.result()[7]
                Benchmark_ind[s,:] = result.result()[8]
                EM_ind[s,:] = result.result()[9]
    else: 
        
        phi = []
        print("Error")
        
    
    weight = np.reshape(weight, (weight.shape[0]))
    
    phi = sum(weight*(Metric_vinteret - Metric))    # Compute XPER values
    
    
    benchmark = benchmark[np.isnan(benchmark) == 0] # Benchmark value
    
    sample_EM = sample_EM[np.isnan(sample_EM) == 0] # Observed AUC
    
    
    marginal = pd.DataFrame(Metric_ind_vinteret - Metric_ind ) 
    
    phi_i_j = marginal.multiply(weight,axis=0)
    
    phi_i_j = phi_i_j.sum(axis=0) # Individual XPER values
    
    return phi, phi_i_j, benchmark, sample_EM,Benchmark_ind,EM_ind


