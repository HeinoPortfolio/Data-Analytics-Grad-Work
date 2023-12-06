# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:03:18 2023

@author: Matthew Heino
"""


import matplotlib.pyplot as plt
import missingno as msno
import numpy as np

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor



""" 
    Pre-assessment tasks:
        
        1) Read in the data from the CSV.
        2) Get a feel for what the data contains. Print the the first five 
        rows of the dataframe.


"""
# Show all columns (Marques, 2022).
pd.set_option('display.max_columns', 16)

# Read in the CSV file into a pandas dataframe.
# Read in only specific columns into the dataframe (GeeksforGeeks, 2020).
lin_cols = ['Children','Age', 'Income', 'Gender','VitD_levels','Doc_visits'
            ,'Initial_admin','Complication_risk','Arthritis','Diabetes'
            ,'BackPain','TotalCharge','Initial_days'
            ]

medical_df = pd.read_csv('medical_clean.csv', usecols=lin_cols)

print("Medical dataframe information: \n" , medical_df.info())

# Print the first five rows of the data frame
#print(medical_df.head(5))

# ****************************************************************************
# Check for duplicates in the dataset.
# Step 1
#medical_dups = medical_df[medical_df.duplicated()]
#print("Duplicated rows: \n",medical_dups)

#*****************************************************************************
#Step 2
# Count the numbero fmissing values for the dataframe.
#check if there are missing values in dataset
#print("\nAre there any missing values: ",medical_df.isnull().values.any())
#print("\nTotal missing values: ", medical_df.isnull().sum())

# Create a missing matrix using Missingno library.
#print(msno.matrix(medical_df))

#******************************************************************************
# Step 3  Outliers.
#fig, (ax1, ax2) = plt.subplots(figsize=(8, 8), ncols=2, sharex= True,
#                                    sharey=False)
#sns.boxplot(data=medical_df['Children'], ax = ax1).set(title="Children")
#sns.stripplot(data=medical_df['Children'], ax=ax2, color='red').set(title="Children")
#plt.show()


# See if any of the data lies outside a normal range.

# Calculate the bound.
"""
lower_bound = medical_df['Children'].quantile(0.25)
upper_bound = medical_df['Children'].quantile(0.75)
IQR = upper_bound - lower_bound


# Identify the outliers in the dataframe
threshold = 1.5
outliers = medical_df[(medical_df['Children']  < lower_bound - threshold * IQR ) 
                 |(medical_df['Children']  > upper_bound + threshold * IQR )] 

print("Lower bound: ", lower_bound)
print("Upper bound: ", upper_bound)
print("\nOutliers shape: ", outliers.shape)

medical_df = medical_df.drop(outliers.index)

print("Shape after dropping Children outliers: ",medical_df.shape)
print('Desription of Children: \n',medical_df['Children'].describe())


print(medical_df['Income'].head(10))
"""

#Section C2. *****************************************************************
"""medical_cont_cols = medical_df.select_dtypes(include='number').columns

print(medical_cont_cols)

for col_name in medical_cont_cols:
    print("\nThe summary descriptives for ", col_name)
    print(medical_df[col_name].describe())
    print("\n\n")
  

# For categorical variables
medical_cont_cols = medical_df.select_dtypes(include='object').columns

print(medical_cont_cols)

for col_name in medical_cont_cols:
    print("\nThe summary descriptives for ", col_name)
    print(medical_df[col_name].describe())
    #print(medical_df[col_name].count())
    print(medical_df.groupby([col_name]).size())
    print("\n\n")
    
"""
#Section C3 Data Visualizations

# Citations (Seaborn.Histplot — Seaborn 0.13.0 Documentation, n.d.) 
# and (Creating Multiple Subplots Using Plt.Subplots — Matplotlib 3.8.2 Documentation, n.d.)


#medical_cont_cols = medical_df.select_dtypes(include='number').columns

#print(medical_cont_cols)
"""
fig, axs = plt.subplots(figsize=(15,15),nrows=4, ncols=2, sharex=False, sharey=False)
fig.tight_layout(pad=5.0)
sns.histplot(data=medical_df['Initial_days'],ax=axs[0,0]).set(title='Initial Days Count (Target)')
sns.histplot(data=medical_df['Children'], discrete=True, ax=axs[0,1]).set(title='Children Count')
sns.histplot(data=medical_df['Age'], ax=axs[1,0]).set(title='Age Count')
sns.histplot(data=medical_df['Income'], ax=axs[1,1]).set(title='Income Count')
sns.histplot(data=medical_df['VitD_levels'], ax=axs[2,0]).set(title='Vitamin D Levels Count')
sns.histplot(data=medical_df['Doc_visits'], discrete=True, ax=axs[2,1]).set(title='Doctor Visit Count')
sns.histplot(data=medical_df['TotalCharge'], discrete=False, ax=axs[3,0]).set(title='Total Charge Count')
fig.delaxes(axs[3,1])

plt.show()

#medical_cont_cols = medical_df.select_dtypes(include='object').columns
# print(medical_cont_cols)


fig, axs = plt.subplots(figsize=(15,15),nrows=3, ncols=2, sharex=False
                        , sharey=False)
fig.tight_layout(pad=5.0)

sns.histplot(data=medical_df['Gender']
             ,ax=axs[0,0]).set(title='Gender (Male, Female, Nonbinary) Count')
sns.histplot(data=medical_df['Initial_admin']
             ,ax=axs[0,1]).set(title='Initial Admission Count')
sns.histplot(data=medical_df['Complication_risk']
             ,ax=axs[1,0]).set(title='Complication Risk Count')
sns.histplot(data=medical_df['Arthritis']
             ,ax=axs[1,1]).set(title='Arthritis Count')
sns.histplot(data=medical_df['Diabetes']
             ,ax=axs[2,0]).set(title='Diabetes Count')
sns.histplot(data=medical_df['BackPain']
             ,ax=axs[2,1]).set(title='Back Pain Count')
plt.show()
 
"""

# Bivariate

#Continous 
"""
fig, axs = plt.subplots(figsize=(15,15),nrows=3, ncols=2, sharex=False
                        , sharey=False)
fig.tight_layout(pad=5.0)

sns.scatterplot(x='Children', y='Initial_days',data=medical_df
                , ax=axs[0,0]).set(title='Initial Days Count (Target) vs Children')
sns.scatterplot(x='Age', y='Initial_days',data=medical_df
                , ax=axs[0,1]).set(title='Initial Days Count (Target) vs Age')
sns.scatterplot(x='Income', y='Initial_days',data=medical_df
                , ax=axs[1,0]).set(title='Initial Days Count (Target) vs Income')
sns.scatterplot(x='VitD_levels', y='Initial_days',data=medical_df
                , ax=axs[1,1]).set(title='Initial Days Count (Target) vs Vitamin D Levels')
sns.scatterplot(x='Doc_visits', y='Initial_days',data=medical_df
                , ax=axs[2,0]).set(title='Initial Days Count (Target) vs Doctor Visits')
sns.scatterplot(x='TotalCharge', y='Initial_days',data=medical_df
                , ax=axs[2,1]).set(title='Initial Days Count (Target) vs TotalCharge')





# Categorical variables
# Citation (Seaborn.Violinplot() Method, n.d.)

fig, axs = plt.subplots(figsize=(15,15),nrows=3, ncols=2, sharex=False
                        , sharey=False)
fig.tight_layout(pad=5.0)

sns.set_style("darkgrid")
sns.violinplot(data=medical_df, x="Gender", y="Initial_days", ax=axs[0,0]
            , dodge=False).set(title='Initial Days Count (Target) vs Gender')


sns.violinplot(data=medical_df, x="Initial_admin", y="Initial_days", ax=axs[0,1]
            , dodge=True).set(title='Initial Days Count (Target) vs Initial Admission')

sns.violinplot(data=medical_df, x="Complication_risk", y="Initial_days", ax=axs[1,0]
            , dodge=True).set(title='Initial Days Count (Target) vs Complication Risk')

sns.violinplot(data=medical_df, x="Arthritis", y="Initial_days", ax=axs[1,1]
            , dodge=True).set(title='Initial Days Count (Target) vs Arthritis')

sns.violinplot(data=medical_df, x="Diabetes", y="Initial_days", ax=axs[2,0]
            , dodge=True).set(title='Initial Days Count (Target) vs Diabetes')

sns.violinplot(data=medical_df, x="BackPain", y="Initial_days", ax=axs[2,1]
            , dodge=True).set(title='Initial Days Count (Target) vs Back Pain')


plt.show()

"""




#Step 4 Transform
medical_cats_cols = medical_df.select_dtypes(include='object')

print(medical_cats_cols)

print(medical_cats_cols.columns[0] + "\n")

for name in medical_cats_cols.columns:
    if len(medical_df[name].unique()) > 2:
       #print("Has three or more values:", name)
       # print("Length: ", str(len(medical_df[name].unique())))
        medical_df =   pd.get_dummies(medical_df
               ,columns = [name]
               ,drop_first = True,
               prefix = name
               )
    else:
        #print("Has two values:", name)
        # Yes = 1 and No = 0.
        medical_df[name].replace(['Yes','No'],[1,0] ,inplace=True)
        
#print(medical_df[['Diabetes','Arthritis','BackPain']])

medical_df.rename(columns={"Initial_admin_Emergency Admission": "Initial_admin_Emergency_Admission"
                           , "Initial_admin_Observation Admission": "Initial_admin_Observation_Admission"}, inplace=True)

print("After transform \n\n")
print(medical_df.head())
print("\n\n\n Medical Info: ",medical_df.info())    

print(medical_df["Initial_admin_Observation_Admission"].head(5))


# Check for multicollinearity. (GeeksforGeeks, 2023)

X = medical_df[['Children','Age','Income','VitD_levels'
                ,'Doc_visits','Arthritis', 'Diabetes','BackPain'
                ,'TotalCharge','Gender_Male','Gender_Nonbinary'
                ,'Initial_admin_Emergency_Admission',
                'Initial_admin_Observation_Admission','Complication_risk_Low'
                ,'Complication_risk_Medium']]


print("Columns: ", X.columns)

vif_df = pd.DataFrame()
vif_df["Features"] = X.columns 

print("\n\n Dataframe: \n", vif_df)

vif_df["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 

print("\n\n", vif_df)


X = medical_df[['Children','Age','Income'
                ,'Doc_visits','Arthritis', 'Diabetes','BackPain'
                ,'TotalCharge','Gender_Male','Gender_Nonbinary'
                ,'Initial_admin_Emergency_Admission',
                'Initial_admin_Observation_Admission','Complication_risk_Low'
                ,'Complication_risk_Medium']]


print("Columns: ", X.columns)

vif_df = pd.DataFrame()
vif_df["Features"] = X.columns 

print("\n\n Dataframe: \n", vif_df)

vif_df["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 

print("\n\n Vit D Levels removed: \n", vif_df)

X = medical_df[['Children','Age','Income'
                ,'Arthritis', 'Diabetes','BackPain'
                ,'TotalCharge','Gender_Male','Gender_Nonbinary'
                ,'Initial_admin_Emergency_Admission',
                'Initial_admin_Observation_Admission','Complication_risk_Low'
                ,'Complication_risk_Medium']]


print("Columns: ", X.columns)

vif_df = pd.DataFrame()
vif_df["Features"] = X.columns 

#print("\n\n Dataframe: \n", vif_df)

vif_df["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 

#print("\n\n Doc Visits removed: \n", vif_df)


# Reduced Dateset 
print("\nValues:    ", vif_df['Features'].values )
reduced_vif_df = medical_df[vif_df["Features"].values]
reduced_vif_df['Initial_days'] = medical_df['Initial_days']

print("VIF", reduced_vif_df['Initial_days'].head(5))
print("MED", medical_df['Initial_days'].head(5))


#print(reduced_vif_df.head())
#print("Frame TYPE_____:",type(reduced_vif_df))


# Save the reduced dataframe to a CSV file.
reduced_vif_df.to_csv('Heino_reduced_medical.csv', index = False, header = True)












