# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:20:50 2023

@author: mehei
"""

import missingno as msno
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

my_df = pd.read_csv('medical_raw_data.csv')
print(my_df.head(10))
print("\n\n\nNulls in frame BEFORE: ", my_df.isna().sum())
"""
sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(8, 8), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=my_df['Population'], ax = ax1).set(title="Population")
sns.stripplot(data=my_df['Population'], ax=ax2, color='green').set(title="Population")
plt.show()
"""


print(my_df['Population'].describe())


# Calculate the bound.
lower_bound = my_df['Population'].quantile(0.25)
upper_bound = my_df['Population'].quantile(0.75)
IQR = upper_bound - lower_bound


# Identify the outliers in the dataframe
threshold = 1.5
outliers = my_df[(my_df['Population']  < lower_bound - threshold * IQR ) 
                 |(my_df['Population']  > upper_bound + threshold * IQR )] 

#print(outliers.index)
#print(outliers.shape)
#print(type(outliers))

#Drop the outliers
my_df = my_df.drop(outliers.index)
#print(my_df.shape)
#print("\n\n\nNulls in frame BEFORE: ", my_df.isna().sum())
#print("\n\n",my_df['Population'].describe())


# Children********************************************************************


#print("\n\n\nNulls in Children frame BEFORE: ", my_df["Children"].isna().sum())
#print("\n\n Children: \n",my_df['Children'].describe())
# Calculate the bound.
lower_bound = my_df['Children'].quantile(0.25)
upper_bound = my_df['Children'].quantile(0.75)
IQR = upper_bound - lower_bound


# Identify the outliers in the dataframe
threshold = 1.5
outliers = my_df[(my_df['Children']  < lower_bound - threshold * IQR ) 
                 |(my_df['Children']  > upper_bound + threshold * IQR )] 

#print(outliers.index)
print(outliers.shape)
#print(type(outliers))


my_df = my_df.drop(outliers.index)
#print("\n\n\nNulls in Children frame After: ", my_df["Children"].isna().sum())
print(my_df.shape)
print("\n\n Children: \n",my_df['Children'].describe())


# Age ************************************************************************#

print("\n\n\nNulls in Age frame BEFORE: ", my_df["Age"].isna().sum())
print("\n\n Age: \n",my_df['Age'].describe())

# Calculate the bound.
lower_bound = my_df['Age'].quantile(0.25)
upper_bound = my_df['Age'].quantile(0.75)
IQR = upper_bound - lower_bound


# Identify the outliers in the dataframe
threshold = 1.5
outliers = my_df[(my_df['Age']  < lower_bound - threshold * IQR ) 
                 |(my_df['Age']  > upper_bound + threshold * IQR )] 

print("Age outliers:\n",outliers.index)
print("Age shape before drop",outliers.shape)


my_df = my_df.drop(outliers.index)
print("\n\n\nNulls in Age frame After: ", my_df["Age"].isna().sum())
print(my_df.shape)
print("\n\n Age: \n",my_df['Age'].describe())






msno.matrix(my_df, labels=True)

msno.matrix(my_df, labels=True)


