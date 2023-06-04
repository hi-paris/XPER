# =============================================================================
#                           Setting of the simulation
# =============================================================================

'''

In this file we generate data according to the Data Generating Process (DGP) of 
a three-feature probit model. Then, we split the data in two parts, a training (70%) 
and a test set (30%). We estimate an XGBoost model using the training data and we 
compute the AUC of the model on the test set. 

'''

import random
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import statsmodels.api as sm



def sample_generation(N=500, p=6, seed=123456):
    """
    Generate a synthetic dataset for binary classification using a multivariate normal distribution.

    Parameters:
        N (int): Size of the sample (default: 500)
        p (int): Number of explanatory variables (default: 6)
        seed (int): Random seed for reproducibility (default: 123456)

    Returns:
        X_train (ndarray): Training set features
        y_train (ndarray): Training set labels
        X_test (ndarray): Test set features
        y_test (ndarray): Test set labels
        seed (int): Random seed used
    """

    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Set the sample size and number of variables
    N = 500  # Size of the sample
    p = 6    # Number of explanatory variables

    # Generate a multivariate normal distribution
    mean = np.repeat(0, p)  # Mean equal to 0 for all variables
    var = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1])  # Variances for each variable
    cov = var * np.eye(p)  # Covariance matrix (diagonal)

    X = np.random.multivariate_normal(mean, cov, N)
    print(X)

    # Simulate the model

    # Coefficients
    beta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).T
    print("Number of coefficients (including the intercept):", np.shape(beta)[0])

    beta_0 = 0.75  # Intercept

    # Errors
    mu_error, std_error = 0, 1
    error = np.random.normal(mu_error, std_error, N)

    # Calculate the index
    index = beta_0 + np.matmul(X, beta)

    # Generate the true labels
    y = index.copy() + error
    y[y > 0] = 1
    y[y <= 0] = 0

    # Calculate the proportion of positive labels
    print("Calculate the proportion of positive labels: ",np.sum(y == 1) / len(y))

    # Calculate the probabilities
    proba = norm.cdf(index)

    # Train/Test split
    N_train = int(0.7 * N)  # 70% of the sample for training

    X_train = X[:N_train, :]
    y_train = y[:N_train]

    X_test = X[N_train:, :]
    y_test = y[N_train:]

    return X_train, y_train, X_test, y_test, seed

X_train, y_train, X_test, y_test, seed  = sample_generation(N=500,p=6,seed=123456)
# =============================================================================
#                              #### XGBOOST ####
# =============================================================================
           
import xgboost as xgb                         
from sklearn.model_selection import RandomizedSearchCV

clf = xgb.XGBClassifier(eval_metric="error") # ,scale_pos_weight=sum(y_train == 0)/sum(y_train == 1)

#x["gender"] = pd.to_numeric(x["gender"])

# Grille d'hyperparamètres
# =============================================================================
 
parameters = {
     "eta"    : np.arange(0,1,0.1) ,                           # Learning rate 
    
     "max_depth"        : np.arange(1,11,1),                    # The maximum depth of the tree.
    
     "min_child_weight" : np.arange(1,100,10),                    # Minimum sum of instance weight (hessian) needed in a child
    
     "gamma"            : np.arange(0,1,0.1),                  # Minimum loss reduction required to make a further partition on 
                                                               # a leaf node of the tree. The larger gamma is, the more 
                                                               # conservative the algorithm will be
    
     "colsample_bytree" : np.arange(0,1,0.1),                  # what percentage of features ( columns ) will be used for 
                                                               # building each tree 
                                                               # Subsampling occurs once for every tree constructed.
     
    "colsample_bylevel" : np.arange(0,1,0.1),                  # This comes into play every time when we achieve the new level 
                                                               # of depth in a tree. Before making any further splits we take 
                                                               # all the features that are left after applying colsample_bytree
                                                               # and filter them again using colsample_bylevel.
    
    "colsample_bynode"  : np.arange(0,1,0.1),                   # The final possible step of choosing features is when we set 
                                                               # colsample_bynode hyperparameter. Before making the next split 
                                                               # we filter all the features left after applying colsample_bylevel. 
                                                               # We choose features for each split on the same level of depth 
                                                               # separately.
    
    "n_estimators": np.arange(1,16,1)                            # Number of boosting rounds.
}

gridXGBOOST = RandomizedSearchCV(clf,
                            parameters, n_jobs=-2,
                            random_state = seed,
                            n_iter=100,#,
                            #scoring=scoring,
                            cv=5,
                            return_train_score=True)


from datetime import datetime 

start_time = datetime.now() 

gridXGBOOST.fit(X_train, y_train)

time_elapsed_XG = datetime.now() - start_time 

#print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed_XG))


# =============================================================================
#                                   AUC
# =============================================================================

# AUC of the model out-of-sample

AUC = roc_auc_score(y_test, gridXGBOOST.predict_proba(X_test)[:,1:])

print("AUC: {}\n".format(round(AUC,4)))

print(gridXGBOOST.best_estimator_)


import joblib
joblib.dump(gridXGBOOST, 'xgboost_model.joblib')
# load the model from disk
loaded_model = joblib.load('xgboost_model.joblib')
result = loaded_model.score(X_test, y_test)
print("Model performance: ",result)
