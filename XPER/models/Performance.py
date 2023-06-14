# =============================================================================
#                               Packages
# =============================================================================
from XPER.models.EM import XPER_choice
from sklearn.metrics import roc_auc_score,brier_score_loss,balanced_accuracy_score,accuracy_score
import numpy as np
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
         model : Model used for predictions.

     Returns:
         PM (float): Performance measure(s) computed based on the specified evaluation metric(s).
    """
    p = X_test.shape[1]

    # # =============================================================================
    # #                               Selected model + predictions
    # # =============================================================================

    model = model

    # # Predicted probabilites on the test sample
    y_hat_proba = model.predict_proba(X_test)[:,1] 

    # # Binary predictions on the test sample with a cutoff at 0.5
    y_pred = (y_hat_proba > 0.5)

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
        
    elif Eval_Metric == ["Specificity"]:
        
        PM = np.mean(((1-y_test)*(1-y_pred))/np.mean((1-y_test)))  # Compute the specificity on the test sample   
    
    elif Eval_Metric == ["Precision"]:
        
        PM = np.mean((y_test*y_pred)/np.mean(y_pred))  # Compute the precision on the test sample   
        
    return PM

def calculate_XPER_values(X_test, y_test, model, Eval_Metric, CFP, CFN, N_coalition_sampled = None , kernel=False, intercept=False):
    """
    Calculates XPER (Extended Partial-Expected Ranking) values for each feature based on the given inputs.

    Parameters:
        X_test (numpy.ndarray): Array of shape (n_samples, n_features) containing the feature values.
        y_test (numpy.ndarray): Array of shape (n_samples,) containing the target values.
        model: The estimated model object.
        Eval_Metric: Name of the performance metric.
        CFP: Cost of false positive.
        CFN: Cost of false negative.
        N_coalition_sampled: Number of coalitions considered to compute the XPER values. Minimum = 1 and maximum = (2**p) - 2.
        kernel: True if we approximate the XPER values (appropriate when the number of features is large), False otherwise
        intercept: True if the model and the features include an intercept, False otherwise
 
    Returns:
        tuple: A tuple containing the following elements:
            - all_contrib (list): List to store all the result of the function "XPER_choice" from the python file "EM.py".
            - all_phi_j (list): List to store the XPER value of each feature + the benchmark.
            - df_phi_i_j (pandas.DataFrame): DataFrame of shape (n_samples, n_features) containing the XPER values for each feature.
            - benchmark_ind (pandas.DataFrame): DataFrame containing the benchmark performance metric for each individual.
    """

    start_time = datetime.now()

    all_contrib = []  # List to store all the result of the function "XPER_choice" from the python file "EM.py"
    all_phi_j = []    # List to store the XPER value of each feature + the benchmark
    p = X_test.shape[1]
    
    if kernel == False:
     
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
                                   intercept=intercept,
                                   kernel=kernel)

          if var == 0:  # Ajout du benchmark
              all_phi_j.append(Contrib[2])  # Add the benchmark to the list of XPER values

          all_contrib.append(Contrib)
          all_phi_j.append(Contrib[0])  # Add the XPER value to "all_contrib_AUC"

      time_elapsed = datetime.now() - start_time

      phi_j = np.insert(all_phi_j[1:], 0,all_phi_j[0])
      
      benchmark_ind = pd.DataFrame(Contrib[4][np.isnan(Contrib[4]) == False], columns=["Individual Benchmark"])

      df_phi_i_j = pd.DataFrame(index=np.arange(len(y_test)), columns=np.arange(p))

      for i, contrib in enumerate(all_contrib):
          phi_i_j = contrib[1].copy()
          df_phi_i_j.iloc[:, i] = phi_i_j.copy()
         
      phi_i_j = pd.concat([benchmark_ind,df_phi_i_j],axis=1).values
      
      return phi_j, phi_i_j

    # =============================================================================
    #                                Kernel XPER
    # =============================================================================

    else:
     
      if N_coalition_sampled == None:
       
        N_coalition_sampled = (2**p) - 2 # Maximum number of coalitions 
       
      Contrib_Kernel = EM.XPER_choice(y = y_test,          # Target values
                                         X = X_test,  # Feature values
                                         model = model,       # Estimated model
                                         Eval_Metric = Eval_Metric,  # Name of the performance metric
                                         N_coalition_sampled = N_coalition_sampled, # Number of coalitions taken into account for XPER computation
                                         CFP=CFP,
                                         CFN=CFN,
                                         intercept=intercept,
                                         kernel=kernel) 

      phi, phi_i_j = Contrib_Kernel

      return phi, phi_i_j
