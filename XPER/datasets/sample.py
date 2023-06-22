
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import statsmodels.api as sm
#def sample_generation(N=500, p=6, seed=123456, pct_train=0.7, mean=np.repeat(0, p), cov=np.repeat(1.5,p)*np.eye(p), beta=np.repeat(0.5,p), beta_0=0.75):
def sample_generation(N=500, p=6, seed=123456, pct_train=0.7, mean=None, cov=None, beta=None, beta_0=None):
    """
    Generate a synthetic dataset for binary classification using a multivariate normal distribution.

    Parameters:
        N (int): Size of the sample (default: 500)
        p (int): Number of explanatory variables (default: 6)
        seed (int): Random seed for reproducibility (default: 123456)
        pct_train (float): Percentage of the dataset devoted to the training of the algorithm (default: 70%)
        mean (ndarray): Vector of size p containing the expected value of the variables (default: np.array([0, 0, ..., 0, 0]))
        cov (ndarray): Covariance matrix of size p x p  (default: array([[1.5, 0. , 0. , 0. , 0. , 0. ],
                                                                         [0. , 1.5, 0. , 0. , 0. , 0. ],
                                                                         [...],
                                                                         [0. , 0. , 0. , 0. , 0. , 1.5]]) )
        beta (ndarray): Vector of parameters of size p (default: np.array([0.5, 0.5, ..., 0.5, 0.5])
        beta_0 (float): Intercept value (default: 0.75)
        
    Returns:
        X_train (ndarray): Training set features
        y_train (ndarray): Training set labels
        X_test (ndarray): Test set features
        y_test (ndarray): Test set labels
        N (int): Size of the sample (default: 500)
        p (int): Number of explanatory variables (default: 6)
        seed (int): Random seed used
    """
    if mean is None:
        mean = np.repeat(0, p)
    if cov is None:
        cov = np.repeat(1.5, p) * np.eye(p)
    if beta is None:
        beta = np.repeat(0.5, p)
    if beta_0 is None:
        beta_0 = 0.75
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    X = np.random.multivariate_normal(mean, cov, N)

    # Simulate the model

    # Errors
    mu_error, std_error = 0, 1
    error = np.random.normal(mu_error, std_error, N)

    # Calculate the index
    index = beta_0 + np.matmul(X, beta)

    # Generate the true labels
    y = index.copy() + error
    y[y > 0] = 1
    y[y <= 0] = 0

    # Train/Test split
    
    N_train = int(pct_train* N)  # 70% of the sample for training

    X_train = X[:N_train, :]
    y_train = y[:N_train]

    X_test = X[N_train:, :]
    y_test = y[N_train:]

    return X_train, y_train, X_test, y_test, p, N, seed
