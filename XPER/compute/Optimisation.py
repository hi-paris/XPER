
"""
Created on Thu May  5 19:56:18 2022

@author: S79158
"""


import math
import pandas as pd
import numpy as np
from datetime import datetime


class OptimizationClass:
    def __init__(self, s, combination_list_sampled, p, X, y, Pred_Formula, delta_n3, Metric, Metric_ind, N, X_shuffle, Metric_vinteret, Metric_ind_vinteret, var_interet):
        self.s = s
        self.combination_list_sampled = combination_list_sampled
        self.p = p
        self.X = X
        self.y = y
        self.Pred_Formula = Pred_Formula
        self.delta_n3 = delta_n3
        self.Metric = Metric
        self.Metric_ind = Metric_ind
        self.N = N
        self.X_shuffle = X_shuffle
        self.Metric_vinteret = Metric_vinteret
        self.Metric_ind_vinteret = Metric_ind_vinteret
        self.var_interet = var_interet
   
    def model_predict1(X, model):
        '''
        X: feature values but exclude the variable of interest
        '''
        N = X.shape[0]  # Number of observations
        intercept = np.repeat(1, N).reshape((N, -1))
        # Array of the form: array([[1],[1],...,[1]])
        X_intercept = np.concatenate((intercept, X), axis=1)
        # The intercept is supposed to be in the first column of the database
        # as mentioned in the function when the parameter intercept = True
        return model.predict(X_intercept)

    def model_predict2(X, model):
        '''
        X: feature values but exclude the variable of interest
        '''
        N = X.shape[0]  # Number of observations
        intercept = np.repeat(1, N).reshape((N, -1))
        # Array of the form: array([[1],[1],...,[1]])
        X_intercept = np.concatenate((intercept, X), axis=1)
        # The intercept is supposed to be in the first column of the database
        # as mentioned in the function when the parameter intercept = True
        return model.predict_proba(X_intercept)[:, 1]


    def model_predict3(X, model):
        return model.predict_proba(X)[:, 1]  # only keep the predicted probabilities
                                              # for the positive class (P(y=1|X))

    def model_predict4(X, model):
        return model.predict(X)

    def model_predict5(X, model):
        '''
        X: feature values but exclude the variable of interest
        '''
        N = X.shape[0]  # Number of observations
        intercept = np.repeat(1, N).reshape((N, -1))
        # Array of the form: array([[1],[1],...,[1]])
        X_intercept = np.concatenate((intercept, X), axis=1)
        # The intercept is supposed to be in the first column of the database
        # as mentioned in the function when the parameter intercept = True
        pred_proba = model.predict(X_intercept)
        pred = pred_proba.copy()
        pred = (pred > 0.5).astype(int)
        return pred

    def model_predict6(X, model):
        pred_proba = model.predict(X)
        pred = pred_proba.copy()
        pred = (pred > 0.5).astype(int)
        return pred

    def loop_choice(s, combination_list_sampled, p, X, y, model, model_predict,
                    delta, Metric, Metric_ind, N, X_shuffle,
                    Metric_vinteret, Metric_ind_vinteret, var_interet,
                    Eval_Metric, CFP, CFN,
                    kernel=True):
        '''
        s : number of the iteration (coalition)
        combination_list_sampled : combination of features in which to compute the
                                   marginal effect on the AUC
        p : number of features
        X : feature values
        y : target values
        model_predict : predictions of the estimated model
        delta : nuisance parameter
        Metric : AUC(s) computed without the variable of interest for each combination
                 of feature (filled as it goes along the loop)
        Metric_ind : Individual AUC computed without the variable of interest for
                     each combination of feature (filled as it goes along the loop)
        N : Sample size
        X_shuffle : Feature values shuffled / Only for the AUC for the moment
        Metric_vinteret : AUC(s) computed with the variable of interest for each
                          combination of feature (filled as it goes along the loop)
        Metric_ind_vinteret : Individual AUC computed with the variable of interest
                              for each combination of feature
                              (filled as it goes along the loop)
        var_interet : number of the variable of interest
        '''
        import math
        import pandas as pd

        for d, v in enumerate(delta):
            globals()['delta_{}'.format(d + 1)] = v

        S = combination_list_sampled[s]  # Retrieve the sth combination of features
        S = [x - 1 for x in S]  # To move index from (1,p) to (0,p-1)

        if kernel == True:
            # Specific to Kernel XPER
            weight_s = (p - 1) / (
                (math.factorial(p) / (math.factorial(len(S)) * math.factorial(p - len(S)))) * (len(S)) * (
                    p - len(S)))
        else:
            weight_s = ((math.factorial(len(S))) * (math.factorial(p - len(S) - 1))) / math.factorial(p)

        G_i_j = []  # Empty list to store G(y_i, x_ij, x_i^S; delta)

        if kernel != True:
            G_i_j_vinteret = []  # Empty list to store G(y_i, x_i^S; delta)

        Metric_ind_j = np.zeros(shape=(1, len(y)))
        # row vector of size N to store E_Xsbar[G(y_i, x_ij, x_i^S; delta)]

        if kernel != True:
            Metric_ind_j_vinteret = np.zeros(shape=(1, len(y)))
            # row vector of size N to store E_x_j,Xsbar[G(y_i, x_ij, x_i^S; delta)]

        for j in range(len(y)):  # Loop for each individual
            start_time_AUC = datetime.now()

            # =====================================================================
            #      Variable of interest unknown/ feature values of the variable of
            #                interest is unknown for individual i
            # =====================================================================

            X_tirage_i = X.copy()  # Feature values
            X_tirage_i[:, S] = X[j, S].copy()  # Change the values of feature S for all individuals
            # to the corresponding value of individual j

            # Code specific to the measure

            if Eval_Metric == ["AUC"]:
                X_shuffle_combination = X_shuffle.copy()
                X_shuffle_combination[:, S] = X[:, S].copy()  # Change the value of the feature
                # values shuffled in column S to the feature values not shuffled

                y_hat_tirage = model_predict(X_shuffle_combination, model)
                # N predictions / for all individuals

                y_hat_proba_i = model_predict(X_tirage_i, model)
                # N predictions / Individual i feature values in subset S

                y_hat_proba_i = pd.DataFrame(y_hat_proba_i, columns=["proba"])
                # Put the predictions made from Individual i feature values in subset S in
                # a DataFrame / Column name "proba"

                ####

                # Calculate the complement of y_hat_tirage and store it in one_y_hat_tirage
                one_y_hat_tirage=1-y_hat_tirage

                # Extract the values of the first column of y_hat_proba_i and transpose the result
                y_hat_proba_i_vec=y_hat_proba_i.values.T[0,:] 

                # Initialize empty lists to store delta_n1 and delta_n2 values
                delta_n1_=[]
                delta_n2_=[]

                # Iterate over each probability in y_hat_proba_i_vec
                for pob in y_hat_proba_i_vec: 
                    # Calculate delta_n1: mean of one_y_hat_tirage for values greater than y_hat_tirage
                    delta_n1_.append(np.mean(one_y_hat_tirage*(pob > y_hat_tirage)))  

                    # Calculate delta_n2: mean of y for values less than or equal to y_hat_tirage
                    delta_n2_.append(np.mean(y*(1-(pob > y_hat_tirage)))) 
                

                # Convert the lists to pandas Series
                delta_n1=pd.Series(delta_n1_)
                delta_n2=pd.Series(delta_n2_)


                
                """
                y_hat_tirage = pd.DataFrame({"tirage": [y_hat_tirage]})
                # Put the predictions made from all individuals feature values in
                # subset S (shuffled values) in a DataFrame / Column name "tirage"

                y_hat_tirage = pd.DataFrame(np.repeat(y_hat_tirage.values, N, axis=0), columns=["tirage"])
                # Create a DataFrame where each row contains all of the predictions
                # from all individuals feature values in subset S (shuffled values)
                # Necessary to use the method apply on a DataFrame to implement a function
                # on each row. Specifically, we want to compute the number of concordant
                # pairs for each individual in the database.

                ####

                y_temp = pd.DataFrame({"y": [y]})  # DataFrame with target values / column name "y"

                y_temp = pd.DataFrame(np.repeat(y_temp.values, N, axis=0), columns=["y"])
                # As for the object "y_hat_tirage" we create a DataFrame where each
                # row contains all of the target values. Necessary to use the method
                # apply on a DataFrame to implement a function on each row.
                # Specifically, we want to compute the number of concordant
                # pairs for each individual in the database.

                df_temp = pd.concat([y_hat_proba_i, y_hat_tirage, y_temp], axis=1)

                delta_n1 = df_temp.apply(
                    lambda row: np.mean((1 - row["y"]) * (row["proba"] > row["tirage"])), axis=1)
                # Compute delta_n1 for individual i / scalar value

                delta_n2 = df_temp.apply(
                    lambda row: np.mean(row["y"] * (1 - (row["proba"] > row["tirage"]))), axis=1)
                # Compute delta_n2 for individual i / scalar value
                """

                G = (y[j] * delta_n1 + (1 - y[j]) * delta_n2) / globals()['delta_1']  # delta_1 created with globals()['delta_{}'.format(d)] // different from delta_n1
                # Compute the individual i contribution to the performance metric
                # for the subset S and without knowing the feature value of interest of
                # individual i

            elif Eval_Metric == ["BS"]:
                y_pred_i = model_predict(X_tirage_i, model)
                G = -(y[j] - y_pred_i) ** 2

            elif Eval_Metric == ["Balanced_accuracy"]:
                y_hat_pred_i = model_predict(X_tirage_i, model)
                G = 0.5 * ((y_hat_pred_i * y[j]) / globals()['delta_1'] + ((1 - y_hat_pred_i) * (1 - y[j]) / globals()[
                    'delta_2']))

            elif Eval_Metric == ["Accuracy"]:
                y_hat_pred_i = model_predict(X_tirage_i, model)
                G = (y_hat_pred_i * y[j]) + ((1 - y_hat_pred_i) * (1 - y[j]))

            elif Eval_Metric == ["Sensitivity"]:
                y_hat_pred_i = model_predict(X_tirage_i, model)
                G = (y_hat_pred_i * y[j]) / globals()['delta_1']

            elif Eval_Metric == ["Specificity"]:
                y_hat_pred_i = model_predict(X_tirage_i, model)
                G = ((1 - y_hat_pred_i) * (1 - y[j])) / globals()['delta_1']

            elif Eval_Metric == ["Precision"]:
                X_shuffle_combination = X_shuffle.copy()
                X_shuffle_combination[:, S] = X[:, S].copy()  # Change the value of the feature
                # values shuffled in column S to the feature values not shuffled

                y_hat_tirage = model_predict(X_shuffle_combination, model)
                # N predictions / for all individuals
                
                delta_n1 = np.mean(y_hat_tirage)
                
                y_hat_pred_i = model_predict(X_tirage_i, model)
                G = (y_hat_pred_i * y[j]) / delta_n1

            elif Eval_Metric == ["MC"]:
                y_hat_pred_i = model_predict(X_tirage_i, model)
                G = -(CFP * ((y_hat_pred_i == 0) * (y[j] == 1)) + CFN * ((y_hat_pred_i == 1) * (y[j] == 0)))

            elif Eval_Metric == ["MSE"]:
                y_pred_i = model_predict(X_tirage_i, model)
                G = -(y[j] - y_pred_i) ** 2

            elif Eval_Metric == ["MAE"]:
                y_pred_i = model_predict(X_tirage_i, model)
                G = -abs(y[j] - y_pred_i)

            elif Eval_Metric == ["R2"]:
                y_pred_i = model_predict(X_tirage_i, model)
                G = 1 - ((y[j] - y_pred_i) ** 2) / globals()['delta_1']

            if kernel != True:
                # =============================================================================
                # Code specific to the exact computation because we compute the XPER
                # value for each variable one by one whereas when using Kernel XPER we
                # compute them all at once without computing the performance metric with
                # the variable of interest in each coalition.
                # =============================================================================

                # =====================================================================
                #      Variable of interest known / feature values of the variable of
                #                interest is known for individual i
                # =====================================================================

                X_tirage_i_vinteret = X.copy()  # Feature values
                X_tirage_i_vinteret[:, S] = X[j, S].copy()
                # Change the values of feature S for all individuals
                # to the corresponding value of individual j

                X_tirage_i_vinteret[:, var_interet] = X[j, var_interet].copy()
                # Change the value of the variable of interest, for all individuals,
                # to the one of individual i

                if Eval_Metric == ["AUC"]:
                    X_shuffle_combination_vinteret = X_shuffle_combination.copy()
                    X_shuffle_combination_vinteret[:, var_interet] = X[:, var_interet].copy()
                    # Change the value of the feature of interest in the shuffled database
                    # to the one of individual i

                    y_hat_tirage_vinteret = model_predict(X_shuffle_combination_vinteret, model)
                    # N predictions / for all individuals

                    y_hat_proba_i_vinteret = model_predict(X_tirage_i_vinteret, model)  # N predictions
                    # N predictions / Individual i feature values in subset S

                    y_hat_proba_i_vinteret = pd.DataFrame(y_hat_proba_i_vinteret, columns=["proba"])
                    # Put the predictions made from Individual i feature values in subset S in
                    # a DataFrame / Column name "proba"

                    ####

                    y_hat_tirage_vinteret = pd.DataFrame({"tirage": [y_hat_tirage_vinteret]})
                    # Put the predictions made from all individuals feature values in
                    # subset S (shuffled values) in a DataFrame / Column name "tirage"

                    y_hat_tirage_vinteret = pd.DataFrame(
                        np.repeat(y_hat_tirage_vinteret.values, N, axis=0), columns=["tirage"])
                    # Create a DataFrame where each row contains all of the predictions
                    # from all individuals feature values in subset S (shuffled values)
                    # Necessary to use the method apply on a DataFrame to implement a function
                    # on each row. Specifically, we want to compute the number of concordant
                    # pairs for each individual in the database.

                    ####

                    y_temp_vinteret = pd.DataFrame({"y": [y]})  # DataFrame with target values / column name "y"

                    y_temp_vinteret = pd.DataFrame(np.repeat(y_temp_vinteret.values, N, axis=0), columns=["y"])
                    # As for the object "y_hat_tirage" we create a DataFrame where each
                    # row contrains all of the target values. Necessary to use the method
                    # apply on a DataFrame to implement a function on each row.
                    # Specifically, we want to compute the number of concordant
                    # pairs for each individual in the database.

                    df_temp_vinteret = pd.concat([y_hat_proba_i_vinteret, y_hat_tirage_vinteret, y_temp_vinteret],
                                                 axis=1)

                    delta_n1_vinteret = df_temp_vinteret.apply(
                        lambda row: np.mean((1 - row["y"]) * (row["proba"] > row["tirage"])), axis=1)
                    # Compute delta_n1 for individual i / scalar value

                    delta_n2_vinteret = df_temp_vinteret.apply(
                        lambda row: np.mean(row["y"] * (1 - (row["proba"] > row["tirage"]))), axis=1)
                    # Compute delta_n2 for individual i / scalar value

                    G_vinteret = (y[j] * delta_n1_vinteret + (1 - y[j]) * delta_n2_vinteret) / globals()[
                        'delta_1']  # delta_1 created with globals()['delta_{}'.format(d)] // different from delta_n1
                    # Compute the individual i contribution to the performance metric
                    # for the subset S while knowing the feature value of interest of
                    # individual i

                elif Eval_Metric == ["BS"]:
                    y_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = -(y[j] - y_pred_i_vinteret) ** 2

                elif Eval_Metric == ["Balanced_accuracy"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = 0.5 * ((y_hat_pred_i_vinteret * y[j]) / globals()['delta_1'] + (
                                (1 - y_hat_pred_i_vinteret) * (1 - y[j]) / globals()['delta_2']))

                elif Eval_Metric == ["Accuracy"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = (y_hat_pred_i_vinteret * y[j]) + ((1 - y_hat_pred_i_vinteret) * (1 - y[j]))

                elif Eval_Metric == ["Sensitivity"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = (y_hat_pred_i_vinteret * y[j]) / globals()['delta_1']

                elif Eval_Metric == ["Specificity"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = ((1 - y_hat_pred_i_vinteret) * (1 - y[j])) / globals()['delta_1']

                elif Eval_Metric == ["Precision"]:
                    X_shuffle_combination_vinteret = X_shuffle_combination.copy()
                    X_shuffle_combination_vinteret[:, var_interet] = X[:, var_interet].copy()
                    # Change the value of the feature of interest in the shuffled database
                    # to the one of individual i

                    y_hat_tirage_vinteret = model_predict(X_shuffle_combination_vinteret, model)
                    # N predictions / for all individuals

                    delta_n1 = np.mean(y_hat_tirage_vinteret)
                    
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = (y_hat_pred_i_vinteret * y[j]) / delta_n1

                elif Eval_Metric == ["MC"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = -(CFP * ((y_hat_pred_i_vinteret == 0) * (y[j] == 1)) + CFN * (
                                (y_hat_pred_i_vinteret == 1) * (y[j] == 0)))

                elif Eval_Metric == ["MSE"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = -(y[j] - y_hat_pred_i_vinteret) ** 2

                elif Eval_Metric == ["MAE"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = -abs(y[j] - y_hat_pred_i_vinteret)

                elif Eval_Metric == ["R2"]:
                    y_hat_pred_i_vinteret = model_predict(X_tirage_i_vinteret, model)
                    G_vinteret = 1 - ((y[j] - y_hat_pred_i_vinteret) ** 2) / globals()['delta_1']

            ###
            G_i_j.append(G)

            if kernel != True:
                G_i_j_vinteret.append(G_vinteret)

            Metric_ind_j[:, j] = np.sum(G) / N

            if kernel != True:
                Metric_ind_j_vinteret[:, j] = np.sum(G_vinteret) / N

            time_elapsed_AUC = datetime.now() - start_time_AUC

            # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed_AUC))

        Metric_s = np.sum(G_i_j) / ((N) ** 2)  # np.mean(Metric_ind_j)

        if kernel != True:
            Metric_s_vinteret = np.sum(G_i_j_vinteret) / ((N) ** 2)  # np.mean(Metric_ind_j_vinteret)

            # =============================================================================
            # Code specific to the exact computation because in this setting we have the
            # empty and the "complete" coalition whereas in the Kernel case we do not
            # take them into account to avoid giving an infinite weight to the coalition
            # =============================================================================

            if S == []:  # If empty coalitions (feature values unknown)
                phi_0 = Metric_s  # Benchmark of the AUC
                phi_i_0 = Metric_ind_j  # "Individual" benchmarks of the AUC
            else:
                phi_0 = np.nan
                phi_i_0 = np.zeros(shape=(1, len(y)))
                phi_i_0[:] = np.nan

            if len(S) == p - 1:
                E_XY = Metric_s_vinteret  # Observed AUC
                E_XY_i = Metric_ind_j_vinteret  # Observed AUC_i
            else:
                E_XY = np.nan
                E_XY_i = np.zeros(shape=(1, len(y)))
                E_XY_i[:] = np.nan

            return s, weight_s, Metric_s, Metric_ind_j, Metric_s_vinteret, Metric_ind_j_vinteret, phi_0, E_XY, phi_i_0, E_XY_i

        else:

            return s, weight_s, Metric_s, Metric_ind_j
