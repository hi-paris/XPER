import os
from os.path import join
import pandas as pd
import numpy as np
import sklearn.datasets

def boston():
    df = pd.read_csv("BostonHousing.csv")
    return df

def iris():
    data = sklearn.datasets.load_iris()
    return data

def adult():
    data =  pd.read_csv("adult.data", sep=",")
    return data
