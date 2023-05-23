'''

In this file we use the data generated with the file "OLS" (see the file
for a description of the data) and the estimated model to  

'''

# =============================================================================
#             Only parameter to change in the code / Performance metric 
# =============================================================================

Eval_Metric = ["MAE"] 
                     # Name of the chosen metric 
                     # ["MSE","MAE","R2"].

# =============================================================================
#                Current path should be the one of this file     
# =============================================================================

# Import the file

import EM

# Execute the file "Probit_model" which contains the Data Generating Process (DGP)
# of a probit model and the estimation of the model. We retrieve from it the 
# data and the estimated model.

exec(open('OLS_model.py').read())

# =============================================================================
#                               Packages
# =============================================================================

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import numpy as np
from IPython import get_ipython
from datetime import datetime
import pandas as pd 

# As we have only 6 features we have 2^5 coalitions to consider for each variable
# to compute the XPER value. Reminds that from k variables we compute 2^(k-1) coalitions
# to compute the XPER value of a given feature.

p = X_test.shape[1] - 1 # withdraw the intercept

N_coalition_sampled = 2**(p-1)

# =============================================================================
#                               Selected model + predictions
# =============================================================================

model = result

# Predicted probabilites on the test sample
y_pred = model.predict(X_test)

# Predicted probabilities on the training sample
y_pred_train = model.predict(X_train) 






if Eval_Metric == ["MSE"]:
    
    PM = -mean_squared_error(y_test, y_pred)  # Compute the MSE on the test sample   
    
elif Eval_Metric == ["MAE"]:
    
    PM = -mean_absolute_error(y_test, y_pred)  # Compute the MAE on the test sample   

elif Eval_Metric == ["R2"]:
    
    PM = r2_score(y_test, y_pred)  # Compute the R2 on the test sample   
        
# =============================================================================
#                                  PM 
# =============================================================================

start_time = datetime.now()

all_contrib = [] # List to store all the result of the function "AUC_PC_pickle" 
                     # from the python file "EM.py"

all_phi_j = []   # List to store the XPER value of each feature + the benchmark 

for var in np.arange(p): # loop on the number of variables
    
    get_ipython().magic('clear')   # Clear the console

    print("Variable numéro:", var)    
    
    Contrib = EM.XPER_choice(y = y_test,          # Target values
                          X = X_test,       # Feature values / include the intercept 
                          model = model,       # Estimated model
                          Eval_Metric = Eval_Metric,  # Name of the performance metric
                          var_interet=var,            # Variable for which to compute XPER value
                          N_coalition_sampled = N_coalition_sampled,# Number of coalitions taken into account for XPER computation
                          intercept=True,
                          kernel=False) 
    
    if var == 0:                             # Ajout du benchmark
        
        all_phi_j.append(Contrib[2]) # Add the benchmark to the list of XPER values
        
    all_contrib.append(Contrib)      
    
    all_phi_j.append(Contrib[0])     # Add the XPER value to "all_contrib_AUC"
    
time_elapsed = datetime.now() - start_time

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

# Retrieve the benchmark performance metric for each individual
# Note: We pick "Contrib_AUC" instead of each "all_contrib_AUC" as the benchmark
# is the same for each individual. Indeed, the benchmark does not depend on the 
# feature of interest as it corresponds to the empty coalition.
benchmark_ind = pd.DataFrame(Contrib[4][np.isnan(Contrib[4]) == False],columns=["Individual Benchmark"])
EM_ind = pd.DataFrame(Contrib[5][np.isnan(Contrib[5]) == False],columns=["Individual EM"])


df_phi_i_j = pd.DataFrame(index=np.arange(len(y_test)),columns=np.arange(p))

for i,contrib in enumerate(all_contrib):
    
    phi_i_j = contrib[1].copy()
    
    df_phi_i_j.iloc[:,i] = phi_i_j.copy()
        
PM_XPER = Contrib[3][0]

Benchmark_XPER = Contrib[2][0]

phi_j_XPER = np.insert(all_phi_j[1:], 0,all_phi_j[0])

phi_j_XPER_pct = 100*(phi_j_XPER[1:] / (phi_j_XPER.sum() - phi_j_XPER[0]))

efficiency_XPER = PM - (phi_j_XPER[1:].sum() + Benchmark_XPER)

efficiency_bench_XPER = np.array([Benchmark_XPER,efficiency_XPER])


# =============================================================================
#                                PM: Kernel XPER
# =============================================================================


N_coalition_sampled = (2**p) - 2   # use all of coalitions: (2**p) - 2 


start_time = datetime.now()


    
Contrib_Kernel = EM.XPER_choice(y = y_test,          # Target values
                                   X = X_test,  # Feature values
                                   model = model,       # Estimated model
                                   Eval_Metric = Eval_Metric,  # Name of the performance metric
                                   N_coalition_sampled = N_coalition_sampled, # Number of coalitions taken into account for XPER computation
                                   intercept=True,
                                   kernel=True) 
    
time_elapsed = datetime.now() - start_time

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


phi, phi_i_j = Contrib_Kernel

efficiency_Kernel = PM - (phi.sum())

efficiency_bench_kernel = np.array([phi[0],efficiency_Kernel])



# =============================================================================
#               Give a name to the variable in the dataset
# =============================================================================

variable_name = ["X" + str(i+1) for i in range(p)] # Give a name to the 6 variables: X1, ..., X6


# =============================================================================
#             Comparison of XPER "exact" computation vs Kernel XPER
#             Not that this part (comparison) is not to include in the package
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(1,2)

fig.suptitle("Difference between exact XPER values and Kernel approximation")

axs[0].bar(variable_name,phi[1:],color="red",label="Kernel",align="edge",width=0.6)
axs[0].bar(variable_name,all_phi_j[1:],label="Exact",width=0.6) 
axs[0].legend() # Add the legend 
sns.despine() # Remove the right and to bar of the graphic


axs[1].bar(["Benchmark","PM - sum(phi_j)"],efficiency_bench_kernel,color="red",label="Kernel",align="edge",width=0.6)
axs[1].bar(["Benchmark","PM - sum(phi_j)"],efficiency_bench_XPER,label="Exact",width=0.6)
sns.despine() # Remove the right and to bar of the graphic
fig.tight_layout()

plt.show()

# =============================================================================
#                              End of the comparison
# =============================================================================

# =============================================================================
#                           Data visualisation
# =============================================================================

# Run the file "Visualisation.py" 

exec(open('Visualisation.py').read())



# =============================================================================
#           Choice between the results of exact XPER and kernel XPER
# =============================================================================

kernel = True # False

if kernel == True:
    
    XPER_v = phi.copy() 
        # XPER value for each feature (global level)
        # Format: array of size p + 1 (include the benchmark)
    
    XPER_v_ind = pd.DataFrame(phi_i_j[:,1:])
        # Dataframe with the XPER values for each feature
        # Careful: it does not include the benchmark values
    
    benchmark_v_ind = pd.DataFrame(phi_i_j[:,0],columns=["Individual Benchmark"])
    
else:
    
    XPER_v = np.insert(all_phi_j[1:],0,all_phi_j[0][0])
        # XPER value for each feature (global level)
        # Format: array of size p + 1 (include the benchmark)
        
    XPER_v_ind = df_phi_i_j.copy() 
        # Dataframe with the XPER values for each feature
        # Careful: it does not include the benchmark values
        
    benchmark_v_ind = benchmark_ind.copy()
    













########################## phi_j contributions ################################
##########################       Bar plot      ################################

X_df = pd.DataFrame(X_test[:,1:])

# Contribution of the features to the performance metric "AUC": phi_j
feature_imp(XPER_v,data=X_df,labels=variable_name,metric="MSE",nb_var=p,percentage=False,echantillon="test")

# Contribution of the features to the performance metric "AUC": phi_j / (AUC - benchmark)
# Note that the AUC corresponds to the sum of the phi_j for j=0 to p, j=0 being the benchmark.
feature_imp(XPER_v,data=X_df,labels=variable_name,metric="MSE",nb_var=p,percentage=True,echantillon="test")








########################## phi_i_j contributions ##############################
##########################     Beeswarn plot     ##############################


#### Now we change the values of "shap_values" to those of XPER

df_phi_i_j = XPER_v_ind.copy()

shap_values.values = df_phi_i_j.to_numpy()  # XPER values for each observation
                                            # and for each feature 
shap_values.base_values = np.reshape(benchmark_v_ind.to_numpy(),benchmark_v_ind.shape[0])
                                            # Base_value = benchmark values
shap_values.data = X_test[:,1:]                   # Data/Inputs

shap_values.feature_names = variable_name   # Label of the features displayed on
                                            # the y-axis


##### Summary plots

ordering = shap_values.mean(0)              # By default the features are ordered
                                            # using shap_values.abs.mean(0). We
                                            # change it to the mean value of the
                                            # XPER values = phi_j (global ones)

shap.plots.beeswarm(shap_values, order=ordering,show=False)
plt.xlabel("Contribution")
#plt.savefig('./Figures/Summary_plots.pdf', format='pdf', dpi = 1200, bbox_inches='tight')
plt.show()



########################## phi_i_j contributions ##############################
##########################      Force plot       ##############################

# =============================================================================
ind_i = 3 # Pick an invidual 

benchmark_ind_i = benchmark_v_ind.iloc[ind_i][0] 
    # Benchmark value on the individual "ind_i" / scalar value
    
perf_value_ind = XPER_v_ind.sum(axis=1) + benchmark_ind_i
    # The individual performance metric value is equal to the sum of the 
    # XPER values 

perf_value_ind = round(perf_value_ind.iloc[ind_i],2) 
    # We pick the value of the individual performance metric value for individual
    # "ind_i" and we round the number to the 2nd decimal.
    
XPER_values = XPER_v_ind.iloc[ind_i].values 
    # Array with the XPER values of individual "ind_i"

    
X_ind_i = pd.Series(np.round(X_test[ind_i,1:],3),index=variable_name).astype(object)
    # Feature values of individual "ind_i" in a pandas series object with the
    # index being equal to the name of the variable ("variable_name"). 
    # Dtype object allows to have both integers and floats on the same pandas series

X_names_ind_i = np.array(variable_name) 
    # Numpy array with the name of the variables ("variable_name")
    # It needs to be a numpy array.

force_plot(XPER=XPER_values,
           perf_value=perf_value_ind,
           base_value=benchmark_ind_i,
           X= X_ind_i,
           X_names=X_names_ind_i,
           savefig_name="force_plot", # Name of the figure which is saved
                                      # Careful: the user need to have a folder 
                                      # named "Figures" in the working directory.
           figsize=(16,2),
           min_perc=0.01) # min_perc: parameter to control the display of the 
                          # the features. If the contribution of a feature 
                          # is very small then we do not display the feature 
                          # to avoid an overloaded graphic.
# =============================================================================

# =============================================================================
# Look at the result and confirm that we display what we wanted 
# =============================================================================

perf_value_ind

XPER_values.sum()

benchmark_ind_i

X_ind_i

# =============================================================================
# End of the verification
# =============================================================================

