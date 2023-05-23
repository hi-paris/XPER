# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:08:31 2023

@author: sebsa
"""

# =============================================================================
#                           Setting of the simulation
# =============================================================================

'''

In this file we generate data according to the Data Generating Process (DGP) of 
a three-feature probit model. Then, we split the data in two parts, a training (70%) 
and a test set (30%). We estimate a probit model using the training data and we 
compute the AUC of the model on the test set. 

'''





import random
import numpy as np
import pandas as pd
from itertools import combinations
from itertools import chain
from sklearn.metrics import precision_score,r2_score,roc_auc_score,mean_squared_error,accuracy_score,confusion_matrix,brier_score_loss
from scipy.stats import norm
import statsmodels.api as sm

# Data simulation 

## Inputs 

# Fixation du seed 

seed = 123456
random.seed(seed)
np.random.seed(seed)

N = 500 # Taille de l'échantillon
p = 3  # Nombre de variables explicatives

# Tirage loi normale multivariée


mean = np.repeat(0, p) # Moyenne égale à 0 pour tous le monde 
var  = np.array([1.2,1,0.1])# np.repeat(8, p) # Variance égale à 1 pour tous le monde 
cov = var*np.eye(p)

X = np.random.multivariate_normal(mean, cov, N)


## Model simulation 

# Coefficients

beta = np.array([0.5, 0.5, 0]).T

print("Nombre de coefficients (constante incluse) :",np.shape(beta)[0])

beta_0 = 0.05

# Errors 

mu_error, std_error= 0,1

error = np.random.normal(mu_error, std_error, N)

# Index 

index = beta_0 + np.matmul(X,beta)

# True y 

y = index.copy() + error # Ne pas oublier de rater le terme d'erreur 

y[y > 0] = 1
y[y<= 0] = 0

np.sum(y==1)/len(y)

# Probabilités

proba = norm.cdf(index)

# =============================================================================
#                             Train / Test
# =============================================================================

N_train = int(0.7*N)


# 70% de l'échantillon pour l'apprentissage du modèle

X_train = X[:N_train,:]

y_train = y[:N_train]

# 30% de l'échantillon pour le test 

X_test = X[N_train:,:]

y_test = y[N_train:]

# =============================================================================
#                  Estimation du modèle de régression logistique
# =============================================================================

# Ajout d'une constante
 
X_train =  sm.add_constant(X_train)

X_test =  sm.add_constant(X_test)

# Estimation du modèle sur l'échantillon d'apprentissage

model = sm.Logit(y_train, X_train)

result = model.fit(method='newton')

print(result.summary(),"\n")


# =============================================================================
#                                   AUC
# =============================================================================

# AUC of the model out-of-sample

AUC = roc_auc_score(y_test, result.predict(X_test))

print("AUC: {}\n".format(round(AUC,4)))



class model_:
    
    def predict_proba(X_w):
        
        # Add an intercept to the database because the model has been
        # estimated with a constant. Therefore, the input database must exclude
        # the intercept
        
        N = X_w.shape[0] # Number of observations
        
        intercept = np.repeat(1,N).reshape((N,-1)) # Array of the form: array([[1],[1],...,[1]])
        
        X = np.concatenate((intercept,X_w),axis=1) # Array of the form: array([[1, x_11, x_12, x_13],
                                                   #                             ... ,
                                                   #                           [1 , x_N1, x_N2, x_N3]])
        
        # Predicted the positive class 
        
        prediction_class_1 = result.predict(X).reshape((N,-1)) # Array of the form: array([[y_hat_1_1],[y_hat_1_2],...,[y_hat_1_N]])
        
        prediction_class_0 = 1 - result.predict(X).reshape((N,-1)) # Array of the form: array([[y_hat_0_1],[y_hat_0_2],...,[y_hat_0_N]])
        
        all_pred = np.concatenate((prediction_class_0,prediction_class_1),axis=1) # Array of the form: array([[y_hat_0_1, y_hat_1_1],
                                                                                  #                                  ...
                                                                                  #                           [y_hat_0_N],...,[y_hat_1_N]])
        
        # =============================================================================
        #       Example of the use of reshape 
        #   
        #        toto = np.array([1,2,3,4]).reshape((4,-1))
        #        tata = np.array([5,6,7,8]).reshape((4,-1))
        #
        #        np.concatenate((toto,tata),axis=1)
        # =============================================================================

        return all_pred


