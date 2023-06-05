# ########################## phi_i_j contributions ##############################
# ##########################     Beeswarn plot     ##############################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#### Now we change the values of "shap_values" to those of XPER

df_phi_i_j = XPER_v_ind.copy()

shap_values.values = df_phi_i_j.to_numpy()  # XPER values for each observation
                                            # and for each feature 
shap_values.base_values = np.reshape(benchmark_v_ind.to_numpy(),benchmark_v_ind.shape[0])
                                            # Base_value = benchmark values
shap_values.data = X_test                   # Data/Inputs

shap_values.feature_names = variable_name   # Label of the features displayed on
                                            # the y-axis

