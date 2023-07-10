import os
from os.path import join
import pandas as pd
import numpy as np
import sklearn.datasets

def boston():
    url = "https://github.com/hi-paris/XPER/blob/main/XPER/datasets/boston/BostonHousing.csv?raw=true"
    df = pd.read_csv(url)
    return df

def loan_status():
    url = "https://github.com/hi-paris/XPER/blob/main/XPER/datasets/loan/Loan_Status.csv?raw=true"
    df = pd.read_csv(url)
    return df

def iris():
    data = sklearn.datasets.load_iris()
    return data

def adult():
    data =  pd.read_csv("adult.data", sep=",")
    return data
