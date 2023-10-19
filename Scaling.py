#%%
# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

# Load the data
df = sys.argv[1]

#%% Function to do the mean centering scale using pandas
def mean_centering_scale(df):
    df_scaled = df - df.mean()
    return df_scaled

# Function to do the autoscaling using pandas
def autoscaling(df):
    df_scaled = (df - df.mean()) / df.std()
    return df_scaled

# Function to do the pareto scaling using pandas
def pareto_scaling(df):
    df_scaled = df / np.sqrt(df.std())
    return df_scaled

#%% Perform the csaling methods
mean_cent_scal_df = mean_centering_scale(df)

autoscaling_df = autoscaling(df)

pareto_scal_df = pareto_scaling(df)
