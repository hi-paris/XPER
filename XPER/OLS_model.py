# =============================================================================
#                           Setting of the simulation
# =============================================================================

'''

In this file we generate data according to the Data Generating Process (DGP) of 
a three-feature linear regression model. Then, we split the data in two parts, 
a training (70%) and a test set (30%). We estimate a linear regression model 
using the training data and we compute the MSE of the model on the test set. 

'''

import random
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Data simulation 

## Inputs 

# Fixation du seed 

seed = 123456
random.seed(seed)
np.random.seed(seed)

N = 500 # Taille de l'échantillon
p = 6  # Nombre de variables explicatives

# Tirage loi normale multivariée


mean = np.repeat(0, p) # Moyenne égale à 0 pour tous le monde 
var  = np.array([1.5,1.4,1.3,1.2,1.1,1])# np.repeat(8, p) # Variance égale à 1 pour tous le monde 
cov = var*np.eye(p)

X = np.random.multivariate_normal(mean, cov, N)


## Model simulation 

# Coefficients

beta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).T

print("Nombre de coefficients (constante incluse) :",np.shape(beta)[0])

beta_0 = 0.75

# Errors 

mu_error, std_error= 0,1

error = np.random.normal(mu_error, std_error, N)

# Index 

index = beta_0 + np.matmul(X,beta)

# True y 

y = index.copy() + error # Ne pas oublier de rater le terme d'erreur 


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
#                  Estimation du modèle de régression linéaire
# =============================================================================

# Ajout d'une constante
 
X_train =  sm.add_constant(X_train)

X_test =  sm.add_constant(X_test)

# Estimation du modèle sur l'échantillon d'apprentissage

model = sm.OLS(y_train, X_train)

result = model.fit()

print(result.summary(),"\n")


# =============================================================================
#                                   AUC
# =============================================================================

# MSE of the model out-of-sample

MSE = mean_squared_error(y_test, result.predict(X_test))

print("MSE: {}\n".format(round(MSE,4)))
