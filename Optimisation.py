# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:11:49 2022

@author: S79158
"""

# =============================================================================
#                               Packages
# =============================================================================

import numpy as np 
from datetime import datetime


def loop_AUC_pickle(s,combination_list_sampled,p,X,y,Pred_Formula,delta_n3,Metric,Metric_ind,N,X_shuffle,Metric_vinteret,Metric_ind_vinteret,var_interet):
    
    
    '''
    
    s : number of the iteration (coalition)
    
    combination_list_sampled : combination of features in which to compute the 
                               marginal effect on the AUC
                               
    p : number of features
    
    X : feature values
    
    y : target values 
    
    Pred_Formula : estimated model
    
    delta_n3 : denominator of the AUC
    
    Metric : AUC(s) computed without the variable of interest for each combination 
             of feature (filled as it goes along the loop)
    
    Metric_ind ; Individual AUC computed without the variable of interest for 
                 each combination of feature (filled as it goes along the loop)
    
    N : Sample size 
    
    X_shuffle : Feature values shuffled
    
    Metric_vinteret : AUC(s) computed with the variable of interest for each 
                      combination of feature (filled as it goes along the loop)
        
    Metric_ind_vinteret : Individual AUC computed with the variable of interest 
                          for each combination of feature 
                          (filled as it goes along the loop)
    
    var_interet : number of the variable of interest 
    
    '''
    
    import math
    import pandas as pd
    
    S = combination_list_sampled[s]   # Retrieve the sth combination of features 
    S = [x-1 for x in S]              # To move index from (1,p) to (0,p-1)
    
    weight_s = ((math.factorial(len(S)))*(math.factorial(p-len(S)-1)))/math.factorial(p) 
        # Weight associated to the coalition
    
    # Code specific to the measure            
    
    X_shuffle_combination = X_shuffle.copy()      
    X_shuffle_combination[:,S] = X[:,S].copy()   # Change the value of the feature 
                                                 # values shuffled in column S 
                                                 # to the feature values not shuffled
    
    #
    
    G_i_j = []             # Empty list to store G(y_i, x_ij, x_i^S; delta) 
    
    G_i_j_vinteret = []    # Empty list to store G(y_i, x_i^S; delta) 
    
    Metric_ind_j = np.zeros(shape=(1,len(y))) 
        # row vector of size N to store E_Xsbar[G(y_i, x_ij, x_i^S; delta)]
    
    Metric_ind_j_vinteret = np.zeros(shape=(1,len(y))) 
        # row vector of size N to store E_x_j,Xsbar[G(y_i, x_ij, x_i^S; delta)]
    
    for j in range(len(y)):              # Loop for each individual 
        
        start_time_AUC = datetime.now()
        
        # =====================================================================
        #      Variable of interest unknown/ feature values of the variable of 
        #                interest is unknown for individual i 
        # =====================================================================
        
        X_tirage_i = X.copy()            # Feature values   
        X_tirage_i[:,S] = X[j,S].copy()  # Change the values of feature S for all individuals
                                         # to the corresponding value of individual j
        
        # Code specific to the measure
            
        y_hat_tirage = Pred_Formula.predict_proba(X_shuffle_combination)[:,1] 
            # N predictions / for all individuals 
        
        y_hat_proba_i = Pred_Formula.predict_proba(X_tirage_i)[:,1] 
            # N predictions / Individual i feature values in subset S
    
                        
        y_hat_proba_i = pd.DataFrame(y_hat_proba_i,columns=["proba"]) 
            # Put the predictions made from Individual i feature values in subset S in 
            # a DataFrame / Column name "proba"
        
        ####
        
        y_hat_tirage = pd.DataFrame({"tirage": [y_hat_tirage]}) 
            # Put the predictions made from all individuals feature values in 
            # subset S (shuffled values) in a DataFrame / Column name "tirage"
        
        y_hat_tirage = pd.DataFrame(np.repeat(y_hat_tirage.values, N, axis=0),columns=["tirage"]) 
          # Create a DataFrame where each row contains all of the predictions 
          # from all individuals feature values in subset S (shuffled values)
          # Necessary to use the method apply on a DataFrame to implement a function
          # on each row. Specifically, we want to compute the number of concordant 
          # pairs for each individual in the database.
        
        ####
        
        y_temp = pd.DataFrame({"y": [y]}) # DataFrame with target values / column name "y"
        
        y_temp = pd.DataFrame(np.repeat(y_temp.values, N, axis=0),columns=["y"])
             # As for the object "y_hat_tirage" we create a DataFrame where each
             # row contrains all of the target values. Necessary to use the method 
             # apply on a DataFrame to implement a function on each row. 
             # Specifically, we want to compute the number of concordant 
             # pairs for each individual in the database.
        
        df_temp = pd.concat([y_hat_proba_i,y_hat_tirage,y_temp],axis=1)
        
        delta_n1 = df_temp.apply(lambda row : np.mean((1-row["y"])*(row["proba"] > row["tirage"])),axis=1)
            # Compute delta_n1 for individual i / scalar value
        
        delta_n2 = df_temp.apply(lambda row : np.mean(row["y"]*(1 - (row["proba"] > row["tirage"]))),axis=1)
            # Compute delta_n2 for individual i / scalar value
            
        G = (y[j]*delta_n1 + (1-y[j])*delta_n2)/delta_n3 
            # Compute the individual i contribution to the performance metric 
            # for the subset S and without knowing the feature value of interest of 
            # individual i 
        
        ###
        
        # =====================================================================
        #      Variable of interest known / feature values of the variable of 
        #                interest is known for individual i 
        # =====================================================================
        
        X_tirage_i_vinteret = X.copy()            # Feature values   
        X_tirage_i_vinteret[:,S] = X[j,S].copy()  
            # Change the values of feature S for all individuals
            # to the corresponding value of individual j
            
        X_tirage_i_vinteret[:,var_interet] = X[j,var_interet].copy()
            # Change the value of the variable of interest, for all individuals,
            # to the one of individual i
        
        X_shuffle_combination_vinteret = X_shuffle_combination.copy()
        X_shuffle_combination_vinteret[:,var_interet] = X[:,var_interet].copy()
            # Change the value of the feature of interest in the shuffled database 
            # to the one of individual i 
        
        y_hat_tirage_vinteret = Pred_Formula.predict_proba(X_shuffle_combination_vinteret)[:,1] 
            # N predictions / for all individuals 
            
        y_hat_proba_i_vinteret = Pred_Formula.predict_proba(X_tirage_i_vinteret)[:,1] # N predictions 
            # N predictions / Individual i feature values in subset S
                        
        y_hat_proba_i_vinteret = pd.DataFrame(y_hat_proba_i_vinteret,columns=["proba"])
            # Put the predictions made from Individual i feature values in subset S in 
            # a DataFrame / Column name "proba"
            
        ####
        
        y_hat_tirage_vinteret = pd.DataFrame({"tirage": [y_hat_tirage_vinteret]})
            # Put the predictions made from all individuals feature values in 
            # subset S (shuffled values) in a DataFrame / Column name "tirage"
            
        y_hat_tirage_vinteret = pd.DataFrame(np.repeat(y_hat_tirage_vinteret.values, N, axis=0),columns=["tirage"])
            # Create a DataFrame where each row contains all of the predictions 
            # from all individuals feature values in subset S (shuffled values)
            # Necessary to use the method apply on a DataFrame to implement a function
            # on each row. Specifically, we want to compute the number of concordant 
            # pairs for each individual in the database.
          
            
        ####
        
        y_temp_vinteret = pd.DataFrame({"y": [y]}) # DataFrame with target values / column name "y"
        
        y_temp_vinteret = pd.DataFrame(np.repeat(y_temp_vinteret.values, N, axis=0),columns=["y"])
            # As for the object "y_hat_tirage" we create a DataFrame where each
            # row contrains all of the target values. Necessary to use the method 
            # apply on a DataFrame to implement a function on each row. 
            # Specifically, we want to compute the number of concordant 
            # pairs for each individual in the database.
       
        df_temp_vinteret = pd.concat([y_hat_proba_i_vinteret,y_hat_tirage_vinteret,y_temp_vinteret],axis=1)
        
        delta_n1_vinteret = df_temp_vinteret.apply(lambda row : np.mean((1-row["y"])*(row["proba"] > row["tirage"])),axis=1)
            # Compute delta_n1 for individual i / scalar value
            
        delta_n2_vinteret = df_temp_vinteret.apply(lambda row : np.mean(row["y"]*(1 - (row["proba"] > row["tirage"]))),axis=1)
            # Compute delta_n2 for individual i / scalar value
            
        G_vinteret = (y[j]*delta_n1_vinteret + (1-y[j])*delta_n2_vinteret)/delta_n3
            # Compute the individual i contribution to the performance metric 
            # for the subset S while knowing the feature value of interest of 
            # individual i 
            
        ###
                
        G_i_j.append(G)
        
        G_i_j_vinteret.append(G_vinteret)
        
        Metric_ind_j[:,j] = np.sum(G)/N
        
        Metric_ind_j_vinteret[:,j] = np.sum(G_vinteret)/N
        
        time_elapsed_AUC = datetime.now() - start_time_AUC

        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed_AUC))
        
    Metric_s = np.sum(G_i_j)/((N)**2) # np.mean(Metric_ind_j)
    Metric_s_vinteret = np.sum(G_i_j_vinteret)/((N)**2) # np.mean(Metric_ind_j_vinteret)
    
    if S == []: # If empty coalitions (feature values unknown)
        phi_0 = Metric_s         # Benchmark of the AUC
        phi_i_0 = Metric_ind_j   # "Individual" benchmarks of the AUC
    else:
        phi_0 = np.nan
        phi_i_0 = np.zeros(shape=(1,len(y)))
        phi_i_0[:] = np.nan 
        
    if len(S) == p-1:      
        E_XY = Metric_s_vinteret        # Observed AUC
        E_XY_i = Metric_ind_j_vinteret  # Observed AUC_i
    else:
        E_XY = np.nan
        E_XY_i = np.zeros(shape=(1,len(y)))
        E_XY_i[:] = np.nan 
        
    return s, weight_s,Metric_s,Metric_ind_j, Metric_s_vinteret,Metric_ind_j_vinteret,  phi_0  ,E_XY,phi_i_0,E_XY_i








