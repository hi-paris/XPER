import os
from os.path import join
import pandas as pd
import numpy as np

def boston():
    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "boston", "BostonHousing.csv")
    df = pd.read_csv(DATA_PATH)
    return df