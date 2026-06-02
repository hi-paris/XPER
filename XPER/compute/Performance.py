# =============================================================================
#                               Packages
# =============================================================================
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from tqdm import tqdm

from XPER.compute.EM import XPER_choice

# warnings.filterwarnings("ignore")


def _to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        return arr.to_numpy()
    return arr.to_numpy()


def _metadata(arr):
    if isinstance(arr, (np.ndarray, pd.Series)):
        return None

    if isinstance(arr, pd.DataFrame):
        metadata = {
            "columns": arr.columns,
            "dtypes": arr.dtypes,
        }
        return metadata

    return None


def _to_df(arr, metadata):

    if metadata is None:

        return arr

    df = pd.DataFrame(arr, columns=metadata["columns"])

    if "dtypes" in metadata:

        for col, dtype in metadata["dtypes"].items():

            df[col] = df[col].astype(dtype)

    return df


class ModelPerformance:
    """
    Class to evaluate the performance of a model using various evaluation metrics.
    """

    def __init__(
        self, X_train, y_train, X_test, y_test, model, sample_size=500, seed=42
    ):

        self.metadata_train = _metadata(X_train)
        self.metadata_test = _metadata(X_test)

        self.model = self._model(model, X_train, self.metadata_train)

        X_train = _to_numpy(X_train)
        X_test = _to_numpy(X_test)
        y_train = _to_numpy(y_train)
        y_test = _to_numpy(y_test)

        if len(X_test) > sample_size:
            indices = np.arange(len(X_test))
            np.random.seed(seed)
            np.random.shuffle(indices)
            sample_indices = indices[:sample_size]

            X_test = X_test[sample_indices]
            y_test = y_test[sample_indices]

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    class _model:

        def __init__(self, model, X_train, metadata_train):

            self.model = model

            if isinstance(X_train, np.ndarray):
                self.input = "array"

            elif isinstance(X_train, pd.DataFrame):
                self.input = "df"

                warnings.warn(
                    "The input data was provided as a pandas DataFrame. "
                    "For internal computations, the data will be converted to NumPy arrays to improve speed. "
                    "However, because the model may have been trained with DataFrame feature names, "
                    "predictions may require converting arrays back to DataFrames, which can slow down repeated "
                    "prediction calls. If your model can make predictions directly from NumPy arrays without "
                    "raising feature-name or dtype errors, provide X_train and X_test directly as NumPy arrays "
                    "for faster computation.",
                    UserWarning,
                    stacklevel=2,
                )

            else:
                raise TypeError("X_train must be a NumPy array or a pandas DataFrame.")

            self.metadata = metadata_train

        def predict(
            self,
            X,
        ):
            if self.input == "array":
                return self.model.predict(X)

            if self.input == "df":
                X = _to_df(X, self.metadata)
                return self.model.predict(X)

        def predict_proba(self, X):
            if self.input == "array":
                return self.model.predict_proba(X)

            if self.input == "df":
                X = _to_df(X, self.metadata)
                return self.model.predict_proba(X)

    def evaluate(self, Eval_Metric, CFP=None, CFN=None):
        """
        Evaluate the performance of the model using various evaluation metrics.

        Parameters:
            Eval_Metric (str or list): Evaluation metric(s) to compute.
                Options in alphabetical order:
                    - for regression models: "MAE", "MSE", "R2", "RMSE".
                    - for classification models: "AUC", "Accuracy", "Balanced_accuracy", "BS" (Brier Score), "MC" (Misclassification Cost),
                                                 "Precision", "Sensitivity", "Specificity".
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
        elif Eval_Metric == ["R2"]:
            y_pred = model.predict(self.X_test)
            PM = r2_score(self.y_test, y_pred)
        elif Eval_Metric == ["RMSE"]:
            y_pred = model.predict(self.X_test)
            PM = np.sqrt(mean_squared_error(self.y_test, y_pred))
        elif Eval_Metric == ["MAE"]:
            y_pred = model.predict(self.X_test)
            PM = mean_absolute_error(self.y_test, y_pred)
        elif Eval_Metric == ["AUC"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            PM = roc_auc_score(self.y_test, y_hat_proba)
        elif Eval_Metric == ["Accuracy"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            PM = accuracy_score(self.y_test, y_pred)
        elif Eval_Metric == ["Balanced_accuracy"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            PM = balanced_accuracy_score(self.y_test, y_pred)
        elif Eval_Metric == ["BS"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            PM = -brier_score_loss(self.y_test, y_hat_proba)
        elif Eval_Metric == ["MC"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            N = len(y_pred)
            FP, FN = np.zeros(shape=N), np.zeros(shape=(N))
            for i in range(N):
                FP[i] = y_pred[i] == 0 and self.y_test[i] == 1
                FN[i] = y_pred[i] == 1 and self.y_test[i] == 0
            FPR = np.mean(FP)
            FNR = np.mean(FN)
            PM = -(CFP * FPR + CFN * FNR)
        elif Eval_Metric == ["Sensitivity"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            PM = np.mean((self.y_test * y_pred) / np.mean(self.y_test))
        elif Eval_Metric == ["Specificity"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            PM = np.mean(
                ((1 - self.y_test) * (1 - y_pred)) / np.mean((1 - self.y_test))
            )
        elif Eval_Metric == ["Precision"]:
            y_hat_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = y_hat_proba > 0.5
            PM = np.mean((self.y_test * y_pred) / np.mean(y_pred))

        return PM

    def calculate_XPER_values(
        self,
        Eval_Metric,
        CFP=None,
        CFN=None,
        N_coalition_sampled=None,
        kernel=True,
        intercept=False,
        execution_type="ThreadPoolExecutor",
        chunk_size=1,
        seed=42,
    ):
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
            seed (int): Random seed for reproducibility.
            chunk_size : int, default=1
                Number of coalition indices `s` grouped into a single submitted future.
                Using `chunk_size > 1` reduces the number of futures submitted to the
                executor, which can lower scheduling/management overhead and make the
                tqdm progress estimate more stable.

                Trade-off:
                - smaller values, e.g. 1 or 2, give smoother progress updates and better
                load balancing when coalition runtimes are heterogeneous;

                - larger values, e.g. 5 or 10, reduce executor overhead but progress is
                updated only when a whole batch completes.

                `chunk_size=1` is equivalent to the original behavior: one future per
                coalition.

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
            N_coalition_sampled = 2 ** (p - 1)
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
                    kernel=kernel,
                    execution_type=execution_type,
                    chunk_size=chunk_size,
                    seed=seed,
                )
                if var == 0:
                    all_phi_j.append(Contrib[2])

                all_contrib.append(Contrib)
                all_phi_j.append(Contrib[0])
            phi_j = np.insert(all_phi_j[1:], 0, all_phi_j[0])

            benchmark_ind = pd.DataFrame(
                Contrib[4][np.isnan(Contrib[4]) == False],
                columns=["Individual Benchmark"],
            )

            df_phi_i_j = pd.DataFrame(
                index=np.arange(len(self.y_test)), columns=np.arange(p)
            )

            for i, contrib in enumerate(all_contrib):
                phi_i_j = contrib[1].copy()
                df_phi_i_j.iloc[:, i] = phi_i_j.copy()

            phi_i_j = pd.concat([benchmark_ind, df_phi_i_j], axis=1).values.astype(
                np.float64
            )

            return phi_j, phi_i_j

        else:
            # for _ in tqdm(range(1), desc="Performing Computation", leave=True):
            if N_coalition_sampled is None:
                if self.X_test.shape[1] == 11:
                    N_coalition_sampled = 2046

                elif self.X_test.shape[1] > 11:
                    N_coalition_sampled = 2 * p + 2048

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
                kernel=kernel,
                execution_type=execution_type,
                chunk_size=chunk_size,
                seed=seed,
            )

            phi, phi_i_j = Contrib_Kernel

        return phi, phi_i_j
