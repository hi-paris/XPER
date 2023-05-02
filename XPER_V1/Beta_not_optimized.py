'''

This file includes the same DGP as the one in "Beta_model.py" and implement
XPER specifically for a three-feature model and without any optimization of the 
code. The results obtained are the same as the one from "Beta.py". The objective
is to show that the optimized code in "Beta.py" for XPER delivers the same 
results as the one from this code which is not optimized and only adapted for 
a three-feature model.

23/03/2023: I attest that the results for XPER provided by "Beta.py" and this file 
            are identical (for XPE)

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



# =============================================================================
# 
# =============================================================================

# XPER sur l'échantillon test

X = X_test[:,1:] # constante dissociée dans le calcul de XPER ci-dessous

N = X.shape[0]

y = y_test.copy()

beta_0_est = result.params[0]

beta_est = result.params[1:]

norm.cdf(beta_0_est + np.matmul(X,beta_est))


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
    


random.seed(42) # VERY important because we change the seed at this moment in the
                # optimized code 

# Lancement du calcul des XPER values

weights = pd.DataFrame([1/3,1/6,1/6,1/3])

liste = [1,2,3]

phi = []

shuffle = np.array(range(N))
random.shuffle(shuffle) #np.random.randint(N-1, size=N)

X_shuffle = X[shuffle,:]

delta_n3 = 2*np.mean(1-y)*np.mean(y) 

all_marginal_contribution_i_j = []

all_phi_i_j = pd.DataFrame(columns=range(4),index=range(N))

all_E_XY_i = []

all_phi_i_0 = []

for elements in liste:
    
    print(elements)
    
    variable_interet = elements
    
    variable = liste.copy()
    
    variable.remove(elements)

    #print(variable_interet,variable)
    # Code 1 optimisé
    
    combination_list = [list(combinations(variable, combination)) for combination in range(len(variable)+1)]
    
    # Unlist of combination_list (exemple: [[1,2], [3,4]] to [1, 2, 3, 4])
    
    combination_list = list(chain.from_iterable(combination_list))
    
    #print(combination_list)

    marginal_contribution = []
    
    MC_i = []
    
    for j,items in enumerate(combination_list):
        
        X_shuffle_combination = X_shuffle.copy()

        S = tuple(np.asarray(items) - 1) 
        
        if S != (): 
            X_shuffle_combination[:,S] = X[:,S].copy()
            
        G_i_j = []
        G_i_j_var = []
        
        all_G_i = []
        all_G_i_var = []
        all_marginal_contribution_i = []
        
        for j in range(len(y)):
            
            #print(i)
            G_i = []
            G_i_var = []
            
            for i in range(len(y)):
                
                y_i = y[j].copy()
                X_tirage_i = X[np.array([i]),:].copy()
                
                if S != (): 
                    X_tirage_i[:,S] = X[j,S].copy()
                    
                                # Variable d'intérêt inconnue

                # j 
                 
                y_hat_tirage = model_.predict_proba(X_shuffle_combination)[:,1] # norm.cdf(beta_0_est + np.matmul(X_shuffle_combination,beta_est))
                
                # i 
                y_hat_proba_i = model_.predict_proba(X_tirage_i)[:,1] #  # norm.cdf(beta_0_est + np.matmul(X_tirage_i,beta_est))
                
                indicatrice = y_hat_proba_i > y_hat_tirage
                
                delta_n1 = np.mean((1-y)*indicatrice)  
                
                delta_n2 = np.mean(y*(1-indicatrice)) 
                
                G = (y_i*delta_n1 + (1-y_i)*delta_n2)/delta_n3
                
                G_i.append(G)
                
                G_i_j.append(G)
                
                                # Variable d'intérêt connue
                 
                X_shuffle_combination_var = X_shuffle_combination.copy()
                # j
                X_shuffle_combination_var[:,variable_interet-1] = X[:,variable_interet-1].copy()
                y_hat_tirage = model_.predict_proba(X_shuffle_combination_var)[:,1] # norm.cdf(beta_0_est + np.matmul(X_shuffle_combination_var,beta_est))
                # i
                X_tirage_i[:,variable_interet-1] = X[j,variable_interet-1].copy()
                y_hat_proba_i = model_.predict_proba(X_tirage_i)[:,1] # norm.cdf(beta_0_est + np.matmul(X_tirage_i,beta_est))
                
                indicatrice = y_hat_proba_i > y_hat_tirage
                
                delta_n1 = np.mean((1-y)*indicatrice)  
                
                delta_n2 = np.mean(y*(1-indicatrice)) 
                
                G = (y_i*delta_n1 + (1-y_i)*delta_n2)/delta_n3
                
                G_i_var.append(G)
                
                G_i_j_var.append(G)
                
            all_G_i.append(np.mean(G_i))
            
            all_G_i_var.append(np.mean(G_i_var))
            
            marginal_contribution_i = np.mean(G_i_var) - np.mean(G_i) 
            
            all_marginal_contribution_i.append(marginal_contribution_i)
            
        expected = np.sum(G_i_j)/(N**2)
        expected_var = np.sum(G_i_j_var)/(N**2)
        
        marginal_contribution_j = expected_var - expected
        
        print("S: ",S,"\n Variable ",variable_interet,"\n Marginal contribution: ",marginal_contribution_j,"\n ",expected_var,"-",expected)
        
        marginal_contribution.append(marginal_contribution_j)
        
        MC_i.append(all_marginal_contribution_i)
        
        if S == ():
            phi_0 = expected
            phi_i_0 = all_G_i.copy()
            all_phi_i_0.append(phi_i_0)
        if len(S) == 2:
            E_XY = expected_var
            E_XY_i = all_G_i_var.copy()
            all_E_XY_i.append(E_XY_i)
            
    phi_j = np.sum(np.array([1/3,1/6,1/6,1/3])*(marginal_contribution))
    
    df_MC_i = pd.DataFrame(MC_i)
    
    phi_i_j = df_MC_i.multiply([1/3,1/6,1/6,1/3],axis=0)
    
    all_phi_i_j.iloc[:,elements] = phi_i_j.sum(axis=0)

    phi.append(phi_j)

E_XY - phi_0 - np.sum(phi)

round(E_XY,4) - round(phi_0,4) - round(np.sum(phi),4)
round(phi_0,4) + round(np.sum(phi),4)

round(phi_0,4) 
round(phi[0],4)
round(phi[1],4)
round(phi[2],4)
