# -*- coding: utf-8 -*-
"""
Created on Thu May  5 19:56:18 2022

@author: S79158
"""


# =============================================================================
#                               Packages
# =============================================================================

#from Optimisation import OptimizationClass
from XPER.compute.Optimisation import OptimizationClass
import random
import numpy as np 
import concurrent.futures
from itertools import combinations
from itertools import chain
import pandas as pd 
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor


def main(nb_var):
  
    # Draw a number randomly from 1 to (nb_var-1)
   
    random_int = random.randint(1, nb_var-1)
   
    # Create a list which contains "random_int" elements. Each element is
    # a scalar from 1 to nb_var
   
    elements = makelist(random_int,nb_var)
  
    return elements


def makelist(random_int,nb_var):
   
    num_list = []
   
    for count in range(random_int):
       
        num_list.append(random.randint(1,nb_var)) # Length of the variable
       
    return num_list


def sampled_coalitions(N_coalition_sampled, nb_var):
  
    # Condition to avoid an infinity loop

    nb_coal_max = 2**(nb_var) - 2
   
    if N_coalition_sampled <= nb_coal_max:
   
        while_condition = N_coalition_sampled
       
    else: # If the number of coalitions to sample is higher than possible
          # as computed with "nb_coal_max" then we set the number of coalitions
          # sampled to "nb_coal_max".
       
        while_condition = nb_coal_max
   
    combination = set() # We use a set to avoid having duplicates
  
    # While the number of coalitions created is inferior to the number of coalitions
    # to create, i.e., "N_coalition_sampled" we continue to create new coalitions.
   
    while len(combination) < while_condition:
       
        tuples = tuple(set(sorted(main(nb_var)))) # set to avoid duplicate values
      
        combination.add(tuples)
      
    combination = list(combination) # List containing tuples representing
                                    # the coaltions of variable.
  
    return combination


    

def XPER_choice(y, X, model, Eval_Metric, var_interet=None, N_coalition_sampled = 1000,
         seed = 42, CFP = None, CFN = None, intercept=True,kernel=True): # Add the kernel parameter
    
    '''
    
    y: target variable. Format = array
    
    X: Features values. Format = array
    
    Eval_Metric: One of the following metrics at the moment 
                
                ["AUC","BS","Balanced_accuracy","Accuracy","MC"].
                
                MC = missclassications cost. Must specify a value for CFP and
                     CFN in this case
                
    var_interet: None only for Kernel estimation because it computes all of 
                 the XPER values at once. Otherwise, integer with the corresponding
                 number of the variable of interest.
                 
    N_coalition_sampled: Number of coalitions to consider to compute XPER values
                         Reminds that for exact computation it requires 2**(p-1)
                         coalitions and for the kernel 2**p coalitions.
                 
    CFP (cost false positive) : specific to the missclassication cost
    
    CFN (cost false negative) : specific to the missclassication cost
    
    intercept=True : The model includes an intercept in the first column of the 
                     database (X)
    
    kernel: if true the kernel SHAP/XPER method is implemented                       
    
    
    '''
    
    # =============================================================================
    #                           Prediction from the model
    #
    # Adapt the code to take into account the case where the model includes the 
    # method "predict" or "predict_proba" + the case where the model includes or not
    # an intercept. If an intercept is included, it must be in the first column 
    # of the inputs (X).
    # =============================================================================

    if (getattr(model,"predict_proba","No") != "No") and (getattr(model,"predict","No") != "No"):
        # The model includes a "predict_proba" method and a "predict" method
        # The predict method returns the predicted class 
        
        if Eval_Metric[0] in ["AUC","BS"]: # Use "predict_proba"
            
            
            if intercept == True: # If an intercept is on the first column of the database
                                  # used to estimate the model
                
                X = np.delete(X,0,axis=1) # delete the first column (intercept) from the 
                                          # features to avoid taking it into account for
                                          # the computation of XPER
                       
                model_predict = OptimizationClass.model_predict2
                                          
            else:
                 
                model_predict = OptimizationClass.model_predict3
         
        else: # If the metric does not use predicted probabilities 
            
            if intercept == True: # If an intercept is on the first column of the database
                                  # used to estimate the model
                
                X = np.delete(X,0,axis=1) # delete the first column (intercept) from the 
                                          # features to avoid taking it into account for
                                          # the computation of XPER
                
                model_predict = OptimizationClass.model_predict1
                
            else:
                
                
                model_predict = OptimizationClass.model_predict4
                
    elif (getattr(model,"predict","No") != "No"): 
        # The model only includes a "predict" method
        # It can be a regression or classification model
        # For instance with statsmodels package, if we estimate a Probit model
        # the predict method returns estimated probabilities of the positive
        # class.
        
        if Eval_Metric[0] in ["R2","MSE","MAE","AUC","BS"]: 
            # Use the predict method, do not need to specify a threshold 
            # to transform predicted probabilities into predicted class 
            # for classification models (without a predict_proba method).
        
            if intercept == True: # If an intercept is on the first column of the database
                                  # used to estimate the model
                
                X = np.delete(X,0,axis=1) # delete the first column (intercept) from the 
                                          # features to avoid taking it into account for
                                          # the computation of XPER
                
                model_predict = OptimizationClass.model_predict1
                
            else:
                
                
                model_predict = OptimizationClass.model_predict4
        
        else: # Case where the model only includes a predict method 
              # which gives predicted probabilities and that the metric 
              # requires predicted class. Therefore, we transform the 
              # predicted probabilities into predicted class by taking a threshold
              # value = 0.5. Add in a later version a parameter to control this 
              # threshold value for the user.
            
            if intercept == True: # If an intercept is on the first column of the database
                                  # used to estimate the model
                
                X = np.delete(X,0,axis=1) # delete the first column (intercept) from the 
                                          # features to avoid taking it into account for
                                          # the computation of XPER
                
                model_predict = OptimizationClass.model_predict5
                
            else:
                
                
                model_predict = OptimizationClass.model_predict6
           
            
    elif (getattr(model,"predict_proba","No") == "No") and (getattr(model,"predict","No") == "No"):      
                
        print("\nThe model does not include a predict method or a predict_proba method\n")
              


    # =============================================================================
    #     
    # =============================================================================


    random.seed(seed) # Fix the seed 
    
    p = np.size(X,1)  # Number of variables 
    
    N = np.size(X,0)  # Sample size 
    

    liste = list(range(p))       # list with numbers from 0 to p-1
    liste = [x+1 for x in liste] # To move index from (0,p-1) to (1,p) / list from 1 to p 
    
    #variable = liste.copy()
    
    if Eval_Metric == ["AUC"]: 
        
        # Code specific to the measure 
        
        delta = [2*np.mean(1-y)*np.mean(y)] # Denominator of the AUC 
        
        shuffle = np.array(range(N)) # list with numbers from 0 to (N-1) 
        random.shuffle(shuffle)      # Shuffle the list with numbers from 0 to (N-1)
        
        X_shuffle = X[shuffle,:]     # Retrieve the feature values according to
                                     # order given by the shuffled list
                                     
    elif Eval_Metric == ["Balanced_accuracy"]:
        
        delta = [np.mean(y), np.mean(1-y)]
        
        X_shuffle = None

    elif Eval_Metric == ["Sensitivity"]:
        
        delta = [np.mean(y)]
        
        X_shuffle = None
        
    elif Eval_Metric == ["Specificity"]:
        
        delta = [1-np.mean(y)]
        
        X_shuffle = None
        
    elif Eval_Metric == ["Precision"]:
        
        delta = []

        shuffle = np.array(range(N)) # list with numbers from 0 to (N-1) 
        random.shuffle(shuffle)      # Shuffle the list with numbers from 0 to (N-1)
        
        X_shuffle = X[shuffle,:]     # Retrieve the feature values according to
                                     # order given by the shuffled list
        
    elif Eval_Metric == ["R2"]:
        
        delta = [np.mean((y-np.mean(y))**2)]
        
        X_shuffle = None
        
    else:
        
        delta = []
        
        X_shuffle = None
        
    # =============================================================================
    
    if kernel == True:
        
        # liste = list(range(p))
        # liste = [x+1 for x in liste] # To move index from (0,p-1) to (1,p)

        # variable = liste.copy()
        
        # combination_list = [list(combinations(variable, combination)) for combination in range(len(variable)+1)]
        # combination_list = list(chain.from_iterable(combination_list))
        # combination_list.pop(0) #Remove the first coalition to avoid infinite weight
        # combination_list.pop(-1) #Remove the last coalition to avoid infinite weight

        # #combination_list_sampled = random.sample(combination_list, N_coalition_sampled)
        # combination_list_sampled = combination_list[:N_coalition_sampled]

        combination_list_sampled = sampled_coalitions(N_coalition_sampled=N_coalition_sampled, nb_var=p)
        print(len(combination_list_sampled))

    else:
    
        variable = liste.copy()      # Copy the list with numbers from 1 to p
        
        variable.remove(var_interet+1) # Remove the variable of interest from "variable"
        
        # List of every possible coalitions
        combination_list = [list(combinations(variable, combination)) for combination in range(len(variable)+1)]
        combination_list = list(chain.from_iterable(combination_list))
                
        #N_combination_list = len(combination_list)
        
        combination_list_sampled = random.sample(combination_list, N_coalition_sampled) # sample without replacement (no need in reality)
        
        #print("Nombre de coalitions totales:",len(combination_list))
        #print("Part du nombre de coalitions:",round(100*(N_coalition_sampled/len(combination_list)),4),"%")
    # =============================================================================

    weight = np.zeros(shape=(N_coalition_sampled,1))          # column vector with N_coalition_sampled 0 elements
    Metric = np.zeros(shape=(N_coalition_sampled))            # vector with N_coalition_sampled 0 elements
    Metric_ind = np.zeros(shape=(N_coalition_sampled,len(y))) # Matrix N_coalition_sampled x len(y) filled with 0 

    if kernel != True:
        
        Benchmark_ind = np.zeros(shape=(N_coalition_sampled,len(y))) # Matrix N_coalition_sampled x len(y) filled with 0 
        EM_ind = np.zeros(shape=(N_coalition_sampled,len(y)))        # Matrix N_coalition_sampled x len(y) filled with 0 
        
        ### Ajout pour calculer avec variable d'intérêt
        
        Metric_vinteret = np.zeros(shape=(N_coalition_sampled))   # vector with N_coalition_sampled 0 elements
        Metric_ind_vinteret = np.zeros(shape=(N_coalition_sampled,len(y))) # Matrix N_coalition_sampled x len(y) filled with 0 
        
        benchmark = np.zeros(shape=(N_coalition_sampled)) # vector with N_coalition_sampled 0 elements
        sample_EM = np.zeros(shape=(N_coalition_sampled)) # vector with N_coalition_sampled 0 elements
        
    # with concurrent.futures.ProcessPoolExecutor() as executor:



    #     if kernel == True:
    #         results = [executor.submit(OptimizationClass.loop_choice, s, combination_list_sampled, p, X, y, model, model_predict, delta, Metric, Metric_ind, N, X_shuffle, Eval_Metric=Eval_Metric, Metric_vinteret=None, Metric_ind_vinteret=None, var_interet=None, CFP=CFP, CFN=CFN, kernel=True) for s in range(N_coalition_sampled)]
    #     else:
    #         results = [executor.submit(OptimizationClass.loop_choice, s, combination_list_sampled, p, X, y, model, model_predict, delta, Metric, Metric_ind, N, X_shuffle, Metric_vinteret, Metric_ind_vinteret, var_interet, Eval_Metric=Eval_Metric, CFP=CFP, CFN=CFN, kernel=False) for s in range(N_coalition_sampled)]



    #     for result in concurrent.futures.as_completed(results):
    #         s, weight_s, Metric_s, Metric_ind_s, *extra_results = result.result()

    #         weight[s, :] = weight_s
    #         Metric[s] = Metric_s
    #         Metric_ind[s, :] = Metric_ind_s

    #     if kernel != True:
    #         Metric_vinteret[s] = extra_results[0]
    #         Metric_ind_vinteret[s, :] = extra_results[1]
    #         benchmark[s] = extra_results[2]
    #         sample_EM[s] = extra_results[3]
    #         Benchmark_ind[s, :] = extra_results[4]
    #         EM_ind[s, :] = extra_results[5]

    with concurrent.futures.ThreadPoolExecutor() as executor:

        if kernel == True:
           
            results = [executor.submit(OptimizationClass.loop_choice,s,combination_list_sampled,p,X,y,model,model_predict,delta,Metric,Metric_ind,N,X_shuffle,Eval_Metric=Eval_Metric,Metric_vinteret=None,Metric_ind_vinteret=None,var_interet=None,CFP=CFP, CFN = CFN, kernel=True) for s in list(range(N_coalition_sampled))]
           
        else: # if exact computation
            
            results = [executor.submit(OptimizationClass.loop_choice,s,combination_list_sampled,p,X,y,model,model_predict,delta,Metric,Metric_ind,N,X_shuffle,Metric_vinteret,Metric_ind_vinteret,var_interet,Eval_Metric = Eval_Metric,CFP=CFP, CFN = CFN,kernel=False) for s in list(range(N_coalition_sampled))]
    
        for result in concurrent.futures.as_completed(results):
            s = result.result()[0]
            #print("Coalition (results)",s)
            weight[s,:] = result.result()[1]
            Metric[s] = result.result()[2]
            Metric_ind[s,:] = result.result()[3]
            
            if kernel != True:
                
                Metric_vinteret[s] = result.result()[4]
                Metric_ind_vinteret[s,:] = result.result()[5]
                benchmark[s] = result.result()[6]
                sample_EM[s] = result.result()[7]
                Benchmark_ind[s,:] = result.result()[8]
                EM_ind[s,:] = result.result()[9]
         
    if kernel == True:

        weight = np.asarray(weight)

        Z = np.zeros(shape=(N_coalition_sampled,p))
        z = combination_list_sampled
        for ii in list(range(N_coalition_sampled)):
            z[ii] = [x-1 for x in z[ii]] #To move index from (1,p) to (0,p-1)
            Z[ii,z[ii]] = 1
            
        const = np.zeros(shape=(N_coalition_sampled,1))+1
        Z = np.concatenate((const, Z), axis=1) #To add a constant
            
        # Estimation of phi (global level)
        
        WLS = sm.WLS(Metric, Z, weight)
        WLS_fit = WLS.fit()
            
        phi = WLS_fit.params
        
        # Estimation of phi_i (individual level)
            
        phi_i_j = np.zeros(shape=(len(y),p+1))
            
        for i in range(len(y)):
                
            WLS = sm.WLS(np.transpose(Metric_ind[:,i]), Z, weight)
            WLS_fit_ind = WLS.fit()
            phi_i_j[i,:] = WLS_fit_ind.params    
        
        
        return phi, phi_i_j
    
    else: # If exact computation
        
        weight = np.reshape(weight, (weight.shape[0]))
        
        phi = sum(weight*(Metric_vinteret - Metric))    # Compute XPER values
        
        
        benchmark = benchmark[np.isnan(benchmark) == 0] # Benchmark value
        
        sample_EM = sample_EM[np.isnan(sample_EM) == 0] # Observed AUC
        
        
        marginal = pd.DataFrame(Metric_ind_vinteret - Metric_ind ) 
        
        phi_i_j = marginal.multiply(weight,axis=0)
        
        phi_i_j = phi_i_j.sum(axis=0) # Individual XPER values
        
        return phi, phi_i_j, benchmark, sample_EM,Benchmark_ind,EM_ind
    
  
