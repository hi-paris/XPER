# =============================================================================
#                               Packages
# =============================================================================
from XPER.compute.EM import XPER_choice
from sklearn.metrics import roc_auc_score,brier_score_loss,balanced_accuracy_score,accuracy_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import pandas as pd 
from tqdm import tqdm
import warnings
import sys

import warnings
warnings.filterwarnings("ignore")

class ModelPerformance():
    """
    Class to evaluate the performance of a model using various evaluation metrics.
    """

    def __init__(self, X_train, y_train, X_test, y_test, model,sample_size=1000):
        """
        Initialize the ModelEvaluator instance.

        Parameters:
            X_train (ndarray): Training set features.
            y_train (ndarray): Training set labels.
            X_test (ndarray): Test set features.
            y_test (ndarray): Test set labels.
            model : Model used for predictions.
        """

        if len(X_test) <= sample_size:
                X_test = X_test
                y_test = y_test
        else:
            indices = np.arange(len(X_test))
            np.random.shuffle(indices)
            sample_indices = indices[:sample_size]
            X_test = X_test.iloc[sample_indices]
            y_test = y_test.iloc[sample_indices]

        self.X_train = X_train.values
        self.y_train = y_train.values
        self.X_test = X_test.values
        self.y_test = y_test.values
        self.model = model




    def evaluate(self, Eval_Metric, CFP=None, CFN=None):
        """
        Evaluate the performance of the model using various evaluation metrics.

        Parameters:
            Eval_Metric (str or list): Evaluation metric(s) to compute. Options: "AUC", "Accuracy",
                "Balanced_accuracy", "BS" (Brier Score), "MC" (Misclassification Cost),
                "Sensitivity", "Specificity", "Precision".
            CFP: Cost of false positive.
            CFN: Cost of false negative.

        Returns:
            PM (float): Performance measure(s) computed based on the specified evaluation metric(s).
        """
        p = self.X_test.shape[1]

        model = self.model
        y_pred = None

        if Eval_Metric == ["MSE"]:
            y_pred = model.predict(self.X_test)
            PM = mean_squared_error(self.y_test, y_pred)
        elif Eval_Metric == ["RMSE"]:
            y_pred = model.predict(self.X_test)
            PM = np.sqrt(mean_squared_error(self.y_test, y_pred))
        elif Eval_Metric == ["MAE"]:
            y_pred = model.predict(self.X_test)
            PM = mean_absolute_error(self.y_test, y_pred)
        elif Eval_Metric == ["AUC"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            PM = roc_auc_score(self.y_test, y_hat_proba)
        elif Eval_Metric == ["Accuracy"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            PM = accuracy_score(self.y_test, y_pred)
        elif Eval_Metric == ["Balanced_accuracy"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            PM = balanced_accuracy_score(self.y_test, y_pred)
        elif Eval_Metric == ["BS"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            PM = -brier_score_loss(self.y_test, y_hat_proba)
        elif Eval_Metric == ["MC"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            N = len(y_pred)
            FP, FN = np.zeros(shape=N), np.zeros(shape=(N))
            for i in range(N):
                FP[i] = (y_pred[i] == 0 and self.y_test[i] == 1)
                FN[i] = (y_pred[i] == 1 and self.y_test[i] == 0)
            FPR = np.mean(FP)
            FNR = np.mean(FN)
            PM = -(CFP * FPR + CFN * FNR)
        elif Eval_Metric == ["Sensitivity"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            PM = np.mean((self.y_test * y_pred) / np.mean(self.y_test))
        elif Eval_Metric == ["Specificity"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            PM = np.mean(((1 - self.y_test) * (1 - y_pred)) / np.mean((1 - self.y_test)))
        elif Eval_Metric == ["Precision"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_hat_proba > 0.5)
            PM = np.mean((self.y_test * y_pred) / np.mean(y_pred))

        return PM

    def calculate_XPER_values(self, Eval_Metric, CFP=None, CFN=None, N_coalition_sampled=None, kernel=True, intercept=False):
        """
        Calculates XPER (Extended Partial-Expected Ranking) values for each feature based on the given inputs.

        Parameters:
            Eval_Metric: Name of the performance metric.
            CFP: Cost of false positive.
            CFN: Cost of false negative.
            N_coalition_sampled: Number of coalitions considered to compute the XPER values.
                Minimum = 1 and maximum = (2**p) - 2.
            kernel: True if we approximate the XPER values (appropriate when the number of features is large),
                False otherwise.
            intercept: True if the model and the features include an intercept, False otherwise.

        Returns:
            tuple: A tuple containing the following elements:
                - phi (numpy.ndarray): Array of shape (n_features + 1) containing the XPER value of each feature
                    and the benchmark value of the performance metric (first value).
                - phi_i_j (numpy.ndarray): Array of shape (n_samples, n_features + 1) containing the individual XPER
                    values of each feature for all individuals and the corresponding benchmark values of the performance
                    metric (first column).
        """
        start_time = datetime.now()
        all_contrib = []
        all_phi_j = []
        p = self.X_test.shape[1]

        if kernel is False:
            N_coalition_sampled = 2**(p - 1)
            total_iterations = p
            progress_bar = tqdm(total=total_iterations, desc="Performing computation")
            for var in range(p):
                Contrib = XPER_choice(
                    y=self.y_test,
                    X=self.X_test,
                    model=self.model,
                    Eval_Metric=Eval_Metric,
                    var_interet=var,
                    N_coalition_sampled=N_coalition_sampled,
                    CFP=CFP,
                    CFN=CFN,
                    intercept=intercept,
                    kernel=kernel
                )

                progress_bar.update(1)
                sys.stdout.flush()
                if var == 0:
                    all_phi_j.append(Contrib[2])

                all_contrib.append(Contrib)
                all_phi_j.append(Contrib[0])
            progress_bar.close()
            time_elapsed = datetime.now() - start_time
            phi_j = np.insert(all_phi_j[1:], 0, all_phi_j[0])

            benchmark_ind = pd.DataFrame(
                Contrib[4][np.isnan(Contrib[4]) == False],
                columns=["Individual Benchmark"]
            )

            df_phi_i_j = pd.DataFrame(index=np.arange(len(self.y_test)), columns=np.arange(p))

            for i, contrib in enumerate(all_contrib):
                phi_i_j = contrib[1].copy()
                df_phi_i_j.iloc[:, i] = phi_i_j.copy()

            phi_i_j = pd.concat([benchmark_ind, df_phi_i_j], axis=1).values

            return phi_j, phi_i_j

        else:
            for _ in tqdm(range(1), desc="Performing Computation", leave=True):
                if N_coalition_sampled is None:
                    if self.X_test.shape[1] == 11:
                        N_coalition_sampled = 2046

                    elif self.X_test.shape[1] > 11:
                        N_coalition_sampled = 2 * p + 2048
                    #elif self.X_test.shape[1]==11:
                    #    N_coalition_sampled = 1024 #+ (2*p)
                    else:
                        N_coalition_sampled = (2**p) - 2

                Contrib_Kernel = XPER_choice(
                    y=self.y_test,
                    X=self.X_test,
                    model=self.model,
                    Eval_Metric=Eval_Metric,
                    N_coalition_sampled=N_coalition_sampled,
                    CFP=CFP,
                    CFN=CFN,
                    intercept=intercept,
                    kernel=kernel
                )

                phi, phi_i_j = Contrib_Kernel

        return phi, phi_i_j
