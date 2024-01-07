# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 05:46:46 2024

@author: ntcrw
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Show all columns.
pd.set_option('display.max_columns', None)


# Read in only the columns that are required for the assessment.
# Description of these columns can be found in the written document that 
# accompanies this file.

feature_cols = {'Lat','Lng','Population','Children', 'Age','Income', 'VitD_levels', 'Doc_visits'
                ,'Full_meals_eaten', 'vitD_supp', 'Initial_days'
                , 'TotalCharge', 'Additional_charges'} 

medical_df = pd.read_csv('medical_clean.csv', usecols=feature_cols)


# Print some information about  the data that is in the dataframe
#print(medical_df.head())
#print(medical_df.info())
med_cols = medical_df.columns
#print(med_cols)


# ############################################################################

# Step C2. Standardize the data and then output to the standarized file to a CSV file
# Create the scaler object.
std_scaler = StandardScaler()


medical_scaled = std_scaler.fit_transform(medical_df)
print(medical_scaled)

# Create a new dataframe that has the scaled data.
# pass in the names of the columns to be associated with the data.

medical_scaled_df = pd.DataFrame(medical_scaled
                                 , columns=med_cols)


# Check to see if the dataframe was created and it contains the right data.
# Transpose the data for a different look.
#print(medical_scaled_df.head().T) 
#print(medical_scaled_df.info())
#print(medical_scaled_df.shape)



# ############################################################################
# Section D Performing the PCA Analysis of the Features
# Section D1 Create the PCA object to perform the analysis.
# Two arguments will be passed the number of components (n_component) 
# and the random_state to allow for reproducibility.

# Create a list with the columns

col_list = []
count = 1

for cols in medical_scaled_df.columns:
    col_list.append("PC" + str(count))
    count = count + 1
    
# Check to see if the list is created
#print(col_list)


# set the initial value of n_components
n_comps = medical_scaled_df.shape[1]
#print(n_comps)

# Create and instantiate the PCA object.
pca = PCA(n_components=medical_scaled_df.shape[1], random_state=247)
#print(pca)

# Fit the data
pca.fit(medical_scaled)

medical_pca = pd.DataFrame(pca.transform(medical_scaled))

print(medical_pca)


# Create the loadings Matrix.

loadings_df = pd.DataFrame()






