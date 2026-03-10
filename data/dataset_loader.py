import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "Fraud.csv")

def load_data(nrows=None):
    return pd.read_csv(DATA_PATH, nrows=nrows)
