# ########################## phi_i_j contributions ##############################
# ##########################      Force plot       ##############################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    
X_ind_i = pd.Series(np.round(X_test[ind_i,:],3),index=variable_name).astype(object)
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