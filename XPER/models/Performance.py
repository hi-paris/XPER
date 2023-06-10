'''
In this file we use the data generated with the file "XGBoost_model" (see the file
for a description of the data) and the estimated model to  
'''

# =============================================================================
#             Only parameter to change in the code / Performance metric 
# =============================================================================

Eval_Metric = ["Precision"] 
                     # Name of the chosen metric 
                     # ["AUC","BS","Balanced_accuracy","Accuracy","MC",
                     #  "Sensitivity","Specificity","Precision"].

CFP = None # Specific to MC / 1 
CFN = None # Specific to MC / 5 

# =============================================================================
#                              Need to install XGBoost 
# =============================================================================


# =============================================================================
#                Current path should be the one of this file     
# =============================================================================

# Import the file


#import XGBoost_model

# Execute the file "XGBoost_model" which contains the Data Generating Process (DGP)
# of a probit model and the estimation of the model. We retrieve from it the 
# data and the estimated model.

#exec(open('XGBoost_model.py').read())

# =============================================================================
#                               Packages
# =============================================================================
from XPER.models.EM import XPER_choice
from sklearn.metrics import roc_auc_score,brier_score_loss,balanced_accuracy_score,accuracy_score
import numpy as np
from IPython import get_ipython
from datetime import datetime
import pandas as pd 

def evaluate_model_performance(Eval_Metric, X_train, y_train, X_test, y_test, model):
    """
     Evaluate the performance of a model using various evaluation metrics.

     Parameters:
         Eval_Metric (str or list): Evaluation metric(s) to compute. Options: "AUC", "Accuracy",
             "Balanced_accuracy", "BS" (Brier Score), "MC" (Misclassification Cost),
             "Sensitivity", "Specificity", "Precision".
         X_train (ndarray): Training set features.
         y_train (ndarray): Training set labels.
         X_test (ndarray): Test set features.
         y_test (ndarray): Test set labels.
         gridXGBOOST : Model used for predictions.

     Returns:
         PM (float): Performance measure(s) computed based on the specified evaluation metric(s).
    """
    p = X_test.shape[1]

    N_coalition_sampled = 2**(p-1)

    # # =============================================================================
    # #                               Selected model + predictions
    # # =============================================================================

    model = model

    # # Predicted probabilites on the test sample
    y_hat_proba = model.predict_proba(X_test)[:,1] 

    # # Binary predictions on the test sample with a cutoff at 0.5
    y_pred = (y_hat_proba > 0.5)

    # # Predicted probabilities on the training sample
    y_hat_proba_train = model.predict_proba(X_train)[:,1] 

    # Binary predictions on the training sample with a cutoff at 0.5
    y_pred_train = (y_hat_proba_train > 0.5)





    if Eval_Metric == ["AUC"]:
        
        PM = roc_auc_score(y_test, y_hat_proba)  # Compute the AUC on the test sample   
        
    elif Eval_Metric == ["Accuracy"]:
        
        PM = accuracy_score(y_test, y_pred)  # Compute the PM on the test sample 
        
    elif Eval_Metric == ["Balanced_accuracy"]:
        
        PM = balanced_accuracy_score(y_test, y_pred)  # Compute the BS on the test sample   

    elif Eval_Metric == ["BS"]:
        
        PM = -brier_score_loss(y_test, y_hat_proba)  # Compute the BS on the test sample   

    elif Eval_Metric == ["MC"]:

        N = len(y_pred)
        CFP = 1
        CFN = 5
        FP, FN = np.zeros(shape=N), np.zeros(shape=(N))
        for i in list(range(N)):
            FP[i] = (y_pred[i] == 0 and y_test[i] == 1)
            FN[i] = (y_pred[i] == 1 and y_test[i] == 0)
        FPR = np.mean(FP)
        FNR = np.mean(FN)
        
        PM = - (CFP*FPR + CFN*FNR) # Compute the PM on the test sample 

    elif Eval_Metric == ["Sensitivity"]:
        
        PM = np.mean((y_test*y_pred)/np.mean(y_test))  # Compute the sensitivity on the test sample   

        #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        #sensitivity = tp / (tp + fn) # Same result that with PM
        
    elif Eval_Metric == ["Specificity"]:
        
        PM = np.mean(((1-y_test)*(1-y_pred))/np.mean((1-y_test)))  # Compute the specificity on the test sample   

        #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        #specificity = tn / (tn+fp) # Same result that with PM
        
    elif Eval_Metric == ["Precision"]:
        
        PM = np.mean((y_test*y_pred)/np.mean(y_pred))  # Compute the precision on the test sample   

        #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        #precision = tp / (tp+fp) # Same result that with PM
        
    return PM
#PM = evaluate_model_performance(Eval_Metric, X_train, y_train, X_test, y_test, gridXGBOOST)        
#print("Performance Metrics: ",PM)


def calculate_XPER_values(X_test, y_test, model, Eval_Metric, CFP, CFN,PM):
    """
    Calculates XPER (Extended Partial-Expected Ranking) values for each feature based on the given inputs.

    Parameters:
        X_test (numpy.ndarray): Array of shape (n_samples, n_features) containing the feature values.
        y_test (numpy.ndarray): Array of shape (n_samples,) containing the target values.
        model: The estimated model object.
        Eval_Metric: Name of the performance metric.
        CFP: Cost of false positive.
        CFN: Cost of false negative.

    Returns:
        tuple: A tuple containing the following elements:
            - all_contrib (list): List to store all the result of the function "AUC_PC_pickle" from the python file "EM.py".
            - all_phi_j (list): List to store the XPER value of each feature + the benchmark.
            - df_phi_i_j (pandas.DataFrame): DataFrame of shape (n_samples, n_features) containing the XPER values for each feature.
            - benchmark_ind (pandas.DataFrame): DataFrame containing the benchmark performance metric for each individual.
            - EM_ind (pandas.DataFrame): DataFrame containing the EM (Expected Metric) performance metric for each individual.
            - efficiency_bench_XPER (numpy.ndarray): Array containing the efficiency benchmark XPER values.
    """

    start_time = datetime.now()

    all_contrib = []  # List to store all the result of the function "AUC_PC_pickle" from the python file "EM.py"
    all_phi_j = []    # List to store the XPER value of each feature + the benchmark
    p = X_test.shape[1]
    N_coalition_sampled = 2**(p-1)

    for var in np.arange(p):  # loop on the number of variables
        print("Variable numéro:", var)

        Contrib = XPER_choice(y=y_test,          # Target values
                                 X=X_test,       # Feature values / include the intercept
                                 model=model,       # Estimated model
                                 Eval_Metric=Eval_Metric,  # Name of the performance metric
                                 var_interet=var,            # Variable for which to compute XPER value
                                 N_coalition_sampled=N_coalition_sampled,  # Number of coalitions taken into account for XPER computation
                                 CFP=CFP,
                                 CFN=CFN,
                                 intercept=False,
                                 kernel=False)

        if var == 0:  # Ajout du benchmark
            all_phi_j.append(Contrib[2])  # Add the benchmark to the list of XPER values

        all_contrib.append(Contrib)
        all_phi_j.append(Contrib[0])  # Add the XPER value to "all_contrib_AUC"

    time_elapsed = datetime.now() - start_time

    benchmark_ind = pd.DataFrame(Contrib[4][np.isnan(Contrib[4]) == False], columns=["Individual Benchmark"])
    EM_ind = pd.DataFrame(Contrib[5][np.isnan(Contrib[5]) == False], columns=["Individual EM"])

    df_phi_i_j = pd.DataFrame(index=np.arange(len(y_test)), columns=np.arange(p))

    for i, contrib in enumerate(all_contrib):
        phi_i_j = contrib[1].copy()
        df_phi_i_j.iloc[:, i] = phi_i_j.copy()

    PM_XPER = Contrib[3][0]
    Benchmark_XPER = Contrib[2][0]
    phi_j_XPER = np.insert(all_phi_j[1:], 0, all_phi_j[0])
    phi_j_XPER_pct = 100 * (phi_j_XPER[1:] / (phi_j_XPER.sum() - phi_j_XPER[0]))
    efficiency_XPER = PM - (phi_j_XPER[1:].sum() + Benchmark_XPER)
    efficiency_bench_XPER = np.array([Benchmark_XPER, efficiency_XPER])

    return all_contrib, all_phi_j, df_phi_i_j, benchmark_ind, EM_ind, efficiency_bench_XPER

#result = calculate_XPER_values(X_test, y_test, gridXGBOOST, Eval_Metric, CFP, CFN)
#print("Efficiency bench XPER: ", result[-1])


# # =============================================================================
# #                                PM: Kernel XPER
# # =============================================================================
# all_contrib = result[0]
# all_phi_j = result[1]
# df_phi_i_j = result[2]
# efficiency_bench_XPER = result[-1]
# p = X_test.shape[1]
# N_coalition_sampled = (2**p) - 2   # use all of coalitions: (2**p) - 2 
# model = gridXGBOOST

# start_time = datetime.now()


    
# Contrib_Kernel = EM.XPER_choice(y = y_test,          # Target values
#                                    X = X_test,  # Feature values
#                                    model = model,       # Estimated model
#                                    Eval_Metric = Eval_Metric,  # Name of the performance metric
#                                    N_coalition_sampled = N_coalition_sampled, # Number of coalitions taken into account for XPER computation
#                                    CFP=CFP,
#                                    CFN=CFN,
#                                    intercept=False,
#                                    kernel=True) 
    
# time_elapsed = datetime.now() - start_time

# #print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


# phi, phi_i_j = Contrib_Kernel

# efficiency_Kernel = PM - (phi.sum())

# efficiency_bench_kernel = np.array([phi[0],efficiency_Kernel])



# # # =============================================================================
# # #               Give a name to the variable in the dataset
# # # =============================================================================

# variable_name = ["X" + str(i+1) for i in range(p)] # Give a name to the 6 variables: X1, ..., X6


# # # =============================================================================
# # #             Comparison of XPER "exact" computation vs Kernel XPER
# # #             Not that this part (comparison) is not to include in the package
# # # =============================================================================

# import matplotlib.pyplot as plt
# import seaborn as sns

# fig, axs = plt.subplots(1,2)

# fig.suptitle("Difference between exact XPER values and Kernel approximation")

# axs[0].bar(variable_name,phi[1:],color="red",label="Kernel",align="edge",width=0.6)
# axs[0].bar(variable_name,all_phi_j[1:],label="Exact",width=0.6) 
# axs[0].legend() # Add the legend 
# sns.despine() # Remove the right and to bar of the graphic


# axs[1].bar(["Benchmark","PM - sum(phi_j)"],efficiency_bench_kernel,color="red",label="Kernel",align="edge",width=0.6)
# axs[1].bar(["Benchmark","PM - sum(phi_j)"],efficiency_bench_XPER,label="Exact",width=0.6)
# sns.despine() # Remove the right and to bar of the graphic
# fig.tight_layout()

# plt.show()

# # # =============================================================================
# # #                              End of the comparison
# # # =============================================================================

# # # =============================================================================
# # #                           Data visualisation
# # # =============================================================================

# # Run the file "Visualisation.py" 

# #exec(open('Visualisation.py').read())



# # # =============================================================================
# # #           Choice between the results of exact XPER and kernel XPER
# # # =============================================================================


