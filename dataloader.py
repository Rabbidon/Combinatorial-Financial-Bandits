# First job - load the data
# Second job - everything else (but start with first job)

import pandas as pd
import os

def import_hf_data():
    df = pd.read_csv("data/hedge_funds_rets.csv")
    return df