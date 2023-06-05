# ########################## phi_j contributions ################################
# ##########################       Bar plot      ################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

X_df = pd.DataFrame(X_test)

# Contribution of the features to the performance metric "AUC": phi_j
feature_imp(XPER_v,data=X_df,labels=variable_name,metric="AUC",nb_var=p,percentage=False,echantillon="test")

# Contribution of the features to the performance metric "AUC": phi_j / (AUC - benchmark)
# Note that the AUC corresponds to the sum of the phi_j for j=0 to p, j=0 being the benchmark.
feature_imp(XPER_v,data=X_df,labels=variable_name,metric="AUC",nb_var=p,percentage=True,echantillon="test")

