# # =============================================================================
# #                           Setting of the simulation
# # =============================================================================

# '''

# In this file we generate data according to the Data Generating Process (DGP) of 
# a three-feature probit model. Then, we split the data in two parts, a training (70%) 
# and a test set (30%). We estimate an XGBoost model using the training data and we 
# compute the AUC of the model on the test set. 

# '''




# # =============================================================================
# #                              #### XGBOOST ####
# # =============================================================================
# #import sys
# #sys.path.insert(0, "../../datasets")
# from XPER.datasets.sample import *
# import random
# import numpy as np
# from sklearn.metrics import roc_auc_score
# from scipy.stats import norm
# import statsmodels.api as sm
# import xgboost as xgb                         
# from sklearn.model_selection import RandomizedSearchCV

# import random
# import numpy as np
# import xgboost as xgb

# # Set the random seed
# random_seed = 42
# random.seed(random_seed)
# np.random.seed(random_seed)



# X_train, y_train, X_test, y_test, p, N, seed  = sample_generation(N=500,p=6,seed=123456)

# clf = xgb.XGBClassifier(eval_metric="error") # ,scale_pos_weight=sum(y_train == 0)/sum(y_train == 1)

# #x["gender"] = pd.to_numeric(x["gender"])

# # Grille d'hyperparamètres
# # =============================================================================
 
# parameters = {
#      "eta"    : np.arange(0,1,0.1) ,                           # Learning rate 
    
#      "max_depth"        : np.arange(1,11,1),                    # The maximum depth of the tree.
    
#      "min_child_weight" : np.arange(1,100,10),                    # Minimum sum of instance weight (hessian) needed in a child
    
#      "gamma"            : np.arange(0,1,0.1),                  # Minimum loss reduction required to make a further partition on 
#                                                                # a leaf node of the tree. The larger gamma is, the more 
#                                                                # conservative the algorithm will be
    
#      "colsample_bytree" : np.arange(0,1,0.1),                  # what percentage of features ( columns ) will be used for 
#                                                                # building each tree 
#                                                                # Subsampling occurs once for every tree constructed.
     
#     "colsample_bylevel" : np.arange(0,1,0.1),                  # This comes into play every time when we achieve the new level 
#                                                                # of depth in a tree. Before making any further splits we take 
#                                                                # all the features that are left after applying colsample_bytree
#                                                                # and filter them again using colsample_bylevel.
    
#     "colsample_bynode"  : np.arange(0,1,0.1),                   # The final possible step of choosing features is when we set 
#                                                                # colsample_bynode hyperparameter. Before making the next split 
#                                                                # we filter all the features left after applying colsample_bylevel. 
#                                                                # We choose features for each split on the same level of depth 
#                                                                # separately.
    
#     "n_estimators": np.arange(1,16,1)                            # Number of boosting rounds.
# }

# gridXGBOOST = RandomizedSearchCV(clf,
#                             parameters, n_jobs=-2,
#                             random_state = seed,
#                             n_iter=100,#,
#                             #scoring=scoring,
#                             cv=5,
#                             return_train_score=True)


# from datetime import datetime 

# start_time = datetime.now() 

# gridXGBOOST.fit(X_train, y_train)

# time_elapsed_XG = datetime.now() - start_time 

# #print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed_XG))


# # =============================================================================
# #                                   AUC
# # =============================================================================

# # AUC of the model out-of-sample

# AUC = roc_auc_score(y_test, gridXGBOOST.predict_proba(X_test)[:,1:])

# #print("AUC: {}\n".format(round(AUC,4)))

# #print(gridXGBOOST.best_estimator_)


# import joblib
# joblib.dump(gridXGBOOST, 'xgboost_model.joblib')
# # load the model from disk
# loaded_model = joblib.load('xgboost_model.joblib')
# result = loaded_model.score(X_test, y_test)
# #print("Model performance: ",result)
