
'''

In this file we use the data generated with the file "Beta_model" (see the file
for a description of the data) and the estimated model to  

'''



# =============================================================================
#                              Need to install XGBoost 
# =============================================================================


# =============================================================================
#                Current path should be the one of this file     
# =============================================================================

# Import the file

import EM
import Beta_model # We need to have a separate file to be able to launch the 
                  # parallel code because the "model" belongs to a class object.

# Execute the file "Beta_model" which contains the Data Generating Process (DGP)
# of a probit model and the estimation of the model. We retrieve from it the 
# data and the estimated model.

exec(open('Beta_model.py').read())

# =============================================================================
#                               Packages
# =============================================================================

from sklearn.metrics import roc_auc_score,accuracy_score,brier_score_loss,balanced_accuracy_score,classification_report
import numpy as np
from IPython import get_ipython
from datetime import datetime
import pandas as pd 

# As we have only 3 features we have 2^2 coalitions to consider for each variable
# to compute the XPER value. Reminds that from k variables we compute 2^(k-1) coalitions
# to compute the XPER value of a given feature.

N_coalition_sampled = 4 # 512


# =============================================================================
#                               Selected model + predictions
# =============================================================================

model = Beta_model.model_

# Predicted probabilites on the test sample
y_hat_proba = model.predict_proba(X_test[:,1:])[:,1] # X_test[:,1:] : exlude the intercept (already added by the function)

# Binary predictions on the test sample with a cutoff at 0.5
y_pred = (y_hat_proba > 0.5)

# Predicted probabilities on the training sample
y_hat_proba_train = model.predict_proba(X_train[:,1:])[:,1] # X_train[:,1:] : exlude the intercept (already added by the function)

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

for var in np.arange(X_test[:,1:].shape[1]): # loop on the number of variables
    
    get_ipython().magic('clear')   # Clear the console

    print("Variable numéro:", var)    
    
    Contrib_AUC = EM.AUC_PC_pickle(y = y_test,          # Target values
                                   X = X_test[:,1:],       # Feature values / exclude the constant
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


df_phi_i_j_AUC = pd.DataFrame(index=np.arange(len(y_test)),columns=np.arange(X_test[:,1:].shape[1]))

for i,contrib in enumerate(all_contrib_AUC):
    
    phi_i_j = contrib[1].copy()
    
    df_phi_i_j_AUC.iloc[:,i] = phi_i_j.copy()
    
    
    
print(all_phi_j_AUC)
