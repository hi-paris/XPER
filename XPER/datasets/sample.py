
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
        N (int): Size of the sample (default: 500)
        p (int): Number of explanatory variables (default: 6)
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

    return X_train, y_train, X_test, y_test, p, N, seed

#X_train, y_train, X_test, y_test, p, N, seed  = sample_generation(N=500,p=6,seed=123456)