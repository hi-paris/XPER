import os
from os.path import join
import pandas as pd
import numpy as np
import sklearn.datasets

def boston():
    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "BostonHousing.csv")
    df = pd.read_csv(DATA_PATH)
    return df

def loan_status():
    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "Loan_Status.csv")
    df = pd.read_csv(DATA_PATH)
    return df

def iris():
    data = sklearn.datasets.load_iris()
    return data

def adult():
    data =  pd.read_csv("adult.data", sep=",")
    return data
