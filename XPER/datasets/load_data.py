import os
from os.path import join
import pandas as pd
import numpy as np

def boston():
    df = pd.read_csv("BostonHousing.csv")
    return df

