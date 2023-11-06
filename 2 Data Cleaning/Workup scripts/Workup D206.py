# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 07:02:06 2023

@author: mehei
"""

import missingno as msno
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

my_df = pd.read_csv('medical_raw_data.csv')
print(my_df.head())
print("\n\n\nNulls in frame BEFORE: ", my_df.isna().sum())

"""
sns.set(style='darkgrid')
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 12), ncols=3, sharex= True,
                                    sharey=False)
sns.boxplot(data=my_df['Age'], ax = ax1).set(title="Age")
sns.boxplot(data=my_df['Children'], ax = ax2).set(title="Children")
sns.boxplot(data=my_df['Income'], ax = ax3).set(title="Income")
plt.show()
"""


""" Work up for finding outliers.
"""
print(my_df)
new_df = my_df[ ~my_df['Children'].isnull()]
null_frame_df = my_df[my_df['Children'].isnull()]
print("\n\nNot Null: \n", new_df)
print("\n\n Is NULL: \n", null_frame_df)

print("\n\n Indices: ", null_frame_df.index)
null_indices = null_frame_df.index

print(type(null_indices))
print(null_indices)

# drop nulls experiment
# Drop the nulls
my_df.drop(index=null_indices, inplace=True)
print(my_df)
print("Nulls in frame: ", my_df.isna().sum())
print("Lenght of dataframe: ", len(my_df))


