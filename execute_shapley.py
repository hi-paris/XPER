# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:14:45 2022

@author: S79158
"""

# =============================================================================
#                              Need to install XGBoost 
# =============================================================================

# =============================================================================
#                 Import the workspace "modelisation.spydata" located in 
#                               the folder "Workspace" 
# =============================================================================

# =============================================================================
#                Current path should be the one of this file     
# =============================================================================

# Import the file

import EM

# =============================================================================
#                               Packages
# =============================================================================

from sklearn.metrics import roc_auc_score,accuracy_score,brier_score_loss,balanced_accuracy_score,classification_report
import numpy as np
from IPython import get_ipython
from datetime import datetime
import pandas as pd 

N_coalition_sampled = 1 # 512


# =============================================================================
#                               Selected model + predictions
# =============================================================================

model = gridXGBOOST

# Predicted probabilites on the test sample
y_hat_proba = model.predict_proba(X_test_dummies)[:,1]

# Binary predictions on the test sample with a cutoff at 0.5
y_pred = (y_hat_proba > 0.5)

# Predicted probabilities on the training sample
y_hat_proba_train = model.predict_proba(X_train_dummies)[:,1]

# Binary predictions on the training sample with a cutoff at 0.5
y_pred_train = (y_hat_proba_train > 0.5)



# =============================================================================
#                                  AUC
# =============================================================================

AUC = roc_auc_score(y_test, y_hat_proba)  # Compute the AUC on the test sample   

Eval_Metric = ["AUC"] # Name of the chosen metricgest



start_time = datetime.now()

all_contrib_AUC = [] # List to store all the result of the function "AUC_PC_pickle" 
                     # from the python file "EM.py"

all_phi_j_AUC = []   # List to store the XPER value of each feature + the benchmark 

for var in np.arange(10): # loop on the number of variables
    
    get_ipython().magic('clear')   # Clear the console

    print("Variable numéro:", var)    
    
    Contrib_AUC = EM.AUC_PC_pickle(y = y_test.iloc[:50].values,          # Target values
                                   X = X_test_dummies.iloc[:50].values,  # Feature values
                                   Pred_Formula = model,       # Estimated model
                                   Eval_Metric = Eval_Metric,  # Name of the performance metric
                                   var_interet=var,            # Variable for which to compute XPER value
                                   N_coalition_sampled = N_coalition_sampled) # Number of coalitions taken into account for XPER computation
    
    if var == 0:                             # Ajout du benchmark
        
        all_phi_j_AUC.append(Contrib_AUC[2]) # Add the benchmark to the list of XPER values
        
    all_contrib_AUC.append(Contrib_AUC)      
    
    all_phi_j_AUC.append(Contrib_AUC[0])     # Add the XPER value to "all_contrib_AUC"
    
time_elapsed = datetime.now() - start_time

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# Retrieve the benchmark performance metric for each individual
# Note: We pick "Contrib_AUC" instead of each "all_contrib_AUC" as the benchmark
# is the same for each individual. Indeed, the benchmark does not depend on the 
# feature of interest as it corresponds to the empty coalition.
benchmark_ind_AUC = pd.DataFrame(Contrib_AUC[4][np.isnan(Contrib_AUC[4]) == False],columns=["Individual Benchmark"])
EM_ind_AUC = pd.DataFrame(Contrib_AUC[5][np.isnan(Contrib_AUC[5]) == False],columns=["Individual EM"])


df_phi_i_j_AUC = pd.DataFrame(index=np.arange(len(y_test.iloc[:50])),columns=np.arange(X_test_dummies.iloc[:50].shape[1]))

for i,contrib in enumerate(all_contrib_AUC):
    
    phi_i_j = contrib[1].copy()
    
    df_phi_i_j_AUC.iloc[:,i] = phi_i_j.copy()
