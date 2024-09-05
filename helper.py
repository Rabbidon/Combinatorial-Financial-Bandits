# First job - load the data
# Second job - everything else (but start with first job)

import pandas as pd
import os
import requests

import datetime

from dateutil.relativedelta import relativedelta

def import_hf_data():
    df = pd.read_csv("data/hedge_funds_rets.csv")
    return df

def import_preqin_data():
    df = pd.read_csv("data/preqin.csv")
    return df


def import_cpi_data():
    df = pd.read_csv("data/CPI.csv")
    return df

def import_fama_french():
    df = pd.read_csv("data/fama_french.csv")
    return df

# We want a mechanism to compute inflation given a lookup table

def compute_inflation(cpi_dict,dt):
    time_final = dt.strftime("%#m/%#d/%Y")
    time_initial = (dt - relativedelta(years=1)).strftime("%#m/%#d/%Y")
    cpi_final = cpi_dict[time_final]["CPI"]
    cpi_initial = cpi_dict[time_initial]["CPI"]
    return (cpi_final-cpi_initial)/cpi_initial

def get_fama_french(fama_french,dt):
    
    prior_month = (dt - relativedelta(month=1)).strftime("%Y%m")
    return fama_french.loc[int(prior_month)].values
    # This should return a vetor I guess
# One-time preqin data handling code

'''    
preqin_dir = "data/Preqin"

existing_indices = None

for file in os.listdir(preqin_dir):
    filename = os.fsdecode(file)
    df = pd.read_excel(os.path.join(preqin_dir, filename),index_col=0)
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    existing_data = set()
    for year in df.index.values:
        for month in df.columns.values:
            if not pd.isna(df.loc[year,month]):
                existing_data.add((year,month))
    if existing_indices == None:
        existing_indices = existing_data
    else:
        existing_indices = existing_indices.intersection(existing_data)

# Now we have existing indices we want to meaningfully convert to datetime

# We now want to build the actual data file

# So for each tuple, we'll grab the values and slap them 1 by 1 into the column
# We'll make a big dataframe
# With the rows as the ordered time series
# And the columns as 1 through 20

new_indices = sorted([datetime.datetime.strftime(datetime.datetime.strptime(str(index[0])+","+index[1], '%Y,%b'),"%Y-%m") for index in existing_indices])

blank_df = pd.DataFrame(index=new_indices,columns=range(1,21))

print(blank_df.isnull().values.any())

for i in range(1,21):
    df = pd.read_excel(os.path.join(preqin_dir, str(i)+".xlsx"),index_col=0)
    for index in existing_indices:
        blank_df.loc[datetime.datetime.strftime(datetime.datetime.strptime(str(index[0])+","+index[1], '%Y,%b'),"%Y-%m"),i] = df.loc[index[0],index[1]]/100
        
print(blank_df.isnull().values.any())
# Save new dataframe as csv
 
blank_df.to_csv("data/preqin.csv")
# Should be pretty easy to also just parse as datetry:
'''