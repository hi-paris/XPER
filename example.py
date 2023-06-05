from sklearn.metrics import roc_auc_score,brier_score_loss,balanced_accuracy_score,accuracy_score
import numpy as np
from IPython import get_ipython
from datetime import datetime
import pandas as pd 

from XPER.datasets.sample import sample_generation
from XPER.models.XPER_XGBoost import evaluate_model_performance
from XPER.models.XGBoost_model import gridXGBOOST
from XPER.models.EM import *

X_train, y_train, X_test, y_test, seed  = sample_generation(N=500,p=6,seed=123456)
print(seed)


Eval_Metric = ["Precision"] 
                     # Name of the chosen metric 
                     # ["AUC","BS","Balanced_accuracy","Accuracy","MC",
                     #  "Sensitivity","Specificity","Precision"].

CFP = None # Specific to MC / 1 
CFN = None # Specific to MC / 5 
N=500
p=6 
seed=123456
PM = evaluate_model_performance(Eval_Metric, X_train, y_train, X_test, y_test, gridXGBOOST)        
print("Performance Metrics: ",PM)

