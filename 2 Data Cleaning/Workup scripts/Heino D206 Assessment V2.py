# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:55:42 2023

@author: Matthew Heino

Course:         D206 Data Cleaning
Instructor:     Dr. K. Middleton
Purpose:        Script file for answering the code based requirements of the 
                assessment.
            

Note:           This script assumes that the data file (medical_raw_data.csv) 
                corruently stored in the local directory 

"""

import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

from fancyimpute import KNN

medical_df = pd.read_csv('medical_raw_data.csv') 
#print(medical_df.head(5))
#print(medical_df.dtypes) 
#print(medical_df.shape)


"""" 
******************************************************************************

Step 1: Detect duplicate elements in the dataframe
    
    Actions:
         
        1) Look for duplicates in the dataframe.

    Note: This file assumes that the data file currently stored in the local 
    directory. This sections corresponds to section C of the document.
    
****************************************************************************** 
"""

# Citation:  (How to Count Duplicates in Pandas Dataframe?, n.d.)
#print("Number of duplicted rows: ", medical_df.duplicated().sum())


""" **************************************************************************

Step 2:     Fix formatting and inconsistencies in the data.  
            
    Note:   This corresponds to section C & D of the accompanying document. 

    Actions:
        
        1) Look for unique categories in the each of the categorical columns 
        of the dataframe. Check the data for errors in 
        
        2) Reduce the number of categories where appropriate.
    
        3) Change the data type of columns to a more appropriate type.
    

************************************************************************** """

# Check for uniqueness among the columns. Will check all columns with type 
# object as they may be categorical and if catergorical see if there are too 
# many categories.

""" **************************************************************************
    
    ZIP code checked for right length.
    This will show an error since this is currently stored as an integer
    This error will be noted in the assessment document in greater detail.
    Note this corresponds to Section D2 of the included document.

******************************************************************************
"""

#print("\nMinimum ZIP value: ", medical_df['Zip'].min())
# Note this 
# Convert Zipcode from int64 to a String to get back leading zero in Zipcode.
medical_df = medical_df.astype({'Zip': str})

# Check data types of the columns.
#print(medical_df.Zip.dtypes)

# Check the Zip code to appropriate length.
# Does not return appropriate length.
#print(medical_df.Zip.str.len().min())

# Add appropriate amount of zeroes to the left of the Zip string.

medical_df['Zip'] = medical_df['Zip'].str.zfill(5)
#print("Min Zip: ",medical_df.Zip.str.len().min())
#print("Max Zip: ",medical_df.Zip.str.len().max())
#print(medical_df['Zip'].str.startswith('0'))
#print("Zip Type: ", medical_df.Zip.dtypes)
#print("\nDatatype of the Zip column: ",medical_df['Zip'].dtypes)    
     
#Check Area for uniqueness

#print("\nUnique values of Area: ",np.sort(medical_df.Area.unique()))     
# All values are acceptable      
      
""" **************************************************************************
    
    Tmezone checked for the amount of timezone categories.
    Found to have 26 will be reduced to 7 timezones.  
    Note this corresponds to Section D2 of the included document.

******************************************************************************
"""
#print(np.sort(medical_df.Timezone.unique())) 
#print("\nUnique Time Zones Before change :",medical_df.Timezone.unique().size)  

timezone_replace_dict = {
            'America/Adak': 'Hawaii-Aleutian', 
            'America/Anchorage' : 'Alaska','America/Boise' : 'Mountain',
            'America/Chicago' :'Central','America/Denver' :  'Mountain',
            'America/Detroit' :'Eastern', 'America/Indiana/Indianapolis' : 'Eastern',
            'America/Indiana/Knox' : 'Central', 'America/Indiana/Marengo' : 'Eastern',
            'America/Indiana/Tell_City' : 'Central','America/Indiana/Vevay' : 'Eastern',
            'America/Indiana/Vincennes' : 'Eastern','America/Indiana/Winamac' : 'Eastern',
            'America/Kentucky/Louisville' : 'Eastern', 'America/Los_Angeles' : 'Pacific',
            'America/Menominee' : 'Central','America/New_York': 'Eastern',
            'America/Nome' : 'Alaska','America/North_Dakota/Beulah' : 'Central',
            'America/North_Dakota/New_Salem' : 'Central','America/Phoenix' : 'Mountain',
            'America/Puerto_Rico' : 'Atlantic','America/Sitka' : 'Alaska',
            'America/Toronto' : 'Eastern', 'America/Yakutat' : 'Alaska', 
            'Pacific/Honolulu' : 'Hawaii-Aleutian'
    }

medical_df['Timezone'].replace(timezone_replace_dict, inplace=True)    
#print("Unique Time Zones after change :",np.sort(medical_df.Timezone.unique())) 

# Conversion of Children and Age column to a data type that is more appropriate.
#print(medical_df[['Children','Age']].dtypes)


# Convert because ages and children are store for whole number only and may 
# cause problems for imputation e.g. fractional children.
#convert_dict = {'Children': "Int64",
 #               'Age': "Int64"
#    }

#medical_df = medical_df.astype(convert_dict)

#medical_df = medical_df.astype(convert_dict)
#print("\n\nChildren data type (Current): ", medical_df.Children.dtypes)
#print("Age data type (Current): ", medical_df.Age.dtypes)

# Check for the range of values to make sure they are acceptable
#print("Children minimum value (Current): ", medical_df.Children.min())
#print("Children maximum value (Current): ", medical_df.Children.max())
#print("Age minimum value (Current): ", medical_df.Age.min())
#print("Age maximum value (Current): ", medical_df.Age.max())


# Check the uniqueness and values that are stored in Education column
#print("\nUnique Education categories :",medical_df.Education.unique())
 
# Check the uniqueness and values that are stored in Employment column
#print("\nUnique Employment categories :",medical_df.Employment.unique())


# check the uniqueness and values that are stored in Income column
# Note: missing salaries
#print("\nUnique Employment categories :",medical_df.Income.nunique())
#print("\nMinimum income :", medical_df.Income.min())
#print("Maximum income :", medical_df.Income.max())


# check the uniqueness and values that are stored in Marital column
#print("\nUnique Marital categories :",medical_df.Marital.unique())

# check the uniqueness and values that are stored in Gender column
#print("\nUnique Gender categories :",medical_df.Gender.unique())
gen_repl = {'Prefer not to answer' : 'nonbinary'}
medical_df['Gender'].replace(gen_repl, inplace=True)
#print("\nUnique Gender categories :",medical_df.Gender.unique())
gender_df = medical_df[medical_df['Gender'] == 'nonbinary']
#print(gender_df['Gender'])


# check the uniqueness and values that are stored in ReAdmis column
#print("\nUnique ReAdmis categories :",medical_df.ReAdmis.unique())

# Check of the doctor vistits
#print("\nVisits minimum value: ", medical_df.Doc_visits.min())
#print("Visits maximum value: ", medical_df.Doc_visits.max())

# Check of the full meals
#print("\n\nFull meals data type (Current): ", medical_df.Full_meals_eaten.dtypes)
#print("\nFull meals minimum value: ", medical_df.Full_meals_eaten.min())
#print("Full meals value: ", medical_df.Full_meals_eaten.max())

# Check of the Vitamin supplements count
#print("\n\nVit D supplements data type (Current): ", medical_df.VitD_supp.dtypes)
#print("Vit D minimum value: ", medical_df.VitD_supp.min())
#print("Vit D maximum value: ", medical_df.VitD_supp.max())

# check the uniqueness and values that are stored in Soft drink column
# note there are missing values nan within this column.
#print("\nUnique Soft drinks categories :",medical_df.Soft_drink.unique())

# check the uniqueness and values that are stored in initial admission column
# note there are missing values nan within this column.
#print("\n\nUnique Soft Initial admission categories : ", medical_df.Initial_admin.unique())

# check the uniqueness and values that are stored in high blood pressure column
#print("\n\nUnique High Blood pressure categories : ", medical_df.HighBlood.unique())

# check the uniqueness and values that are stored in Stoke column
#print("\n\nUnique Stroke categories : ", medical_df.Stroke.unique())

# check the uniqueness and values that are stored in Complication column
#print("\n\nUnique Complication risk categories : ", medical_df.Complication_risk.unique())

# check the uniqueness and values that are stored in Overweight column
#print("\n\nUnique Overweight categories : ", medical_df.Overweight.unique())

""" 
******************************************************************************
    
    Correct propblems with this column not being in the same format as the 
    other columns

*******************************************************************************
"""

repl_dict = {'0.0' : 'No', '1.0' : 'Yes',np.NaN : np.NaN }
#print(medical_df['Overweight'].isnull().sum())

medical_df['Overweight'] = medical_df['Overweight'].astype(str)
medical_df['Overweight'].replace(repl_dict, inplace=True)

#print("\n\nUnique Overweight categories : ", medical_df.Overweight.unique())
#print("Overweight data type (Current): ", medical_df.Overweight.dtypes)

medical_df['Overweight'].replace('nan',np.NaN, inplace=True)

#print("\n\nUnique Overweight categories : ", medical_df.Overweight.unique())

# check the uniqueness and values that are stored in Arthritis columns column
#print("\n\nUnique Arthritist categories : ", medical_df.Arthritis.unique())

# check the uniqueness and values that are stored in Diabetes columns column
#print("\nUnique Diabetes categories : ", medical_df.Diabetes.unique())

# check the uniqueness and values that are stored in Hyperlipidemia columns column
#print("\nUnique Hyperlipidemia categories : ", medical_df.Hyperlipidemia.unique())

# check the uniqueness and values that are stored in BackPain columns column
#print("\nUnique BackPain categories : ", medical_df.BackPain.unique())

repl_anx_dict = {'0.0' : 'No', '1.0' : 'Yes',np.NaN : np.NaN }

# To make sure the NULL values are retained for handling in Step 4
#print("Sum of NULL values in Anxiety (Before): ",medical_df['Anxiety'].isnull().sum())

medical_df['Anxiety'] = medical_df['Anxiety'].astype(str)
medical_df['Anxiety'].replace(repl_anx_dict, inplace=True)

#print("Unique Anxiety categories : ", medical_df.Anxiety.unique())
#print("Overweight data type (Current): ", medical_df.Anxiety.dtypes)

medical_df['Anxiety'].replace('nan',np.NaN, inplace=True)
#print("Sum of NULL values in Anxiety (After): ",medical_df['Anxiety'].isnull().sum())

# Check the uniqueness and values that are stored in Allergic rhinitis columns column
#print("\nUnique Allergic rhinitis categories : ", medical_df.Allergic_rhinitis.unique())

# Check the uniqueness and values that are stored 
# in Reflux esophagitis  column
#print("\nUnique Allergic rhinitis categories : ", medical_df.Allergic_rhinitis.unique())

# Check the uniqueness and values that are stored Asthma column
#print("\nUnique Asthma categories: ", medical_df.Asthma.unique())

# Check the uniqueness and values that are stored Services column
#print("\nUnique Services categories: ", medical_df.Services.unique())

# Check the uniqueness and values that are stored Initial days column
#print("\nUnique Initial_days categories: ", medical_df.Initial_days.unique())
#print("Sum of NULL values in Initial Days: ",medical_df['Initial_days'].isnull().sum())


# Check the uniqueness and values that are stored Total_Charge column
#print("\nUnique Total Charge categories: ", medical_df.TotalCharge.unique())
#print("Sum of NULL values in Total Charge: ",medical_df['TotalCharge'].isnull().sum())

# Check the uniqueness and values that are stored Additional_charges column
#print("\nUnique Additional Charges categories: ", medical_df.Additional_charges.unique())
#print("Sum of NULL values in Additional Charges: ",medical_df['Additional_charges'].isnull().sum())


# Give the items more appropriate names for columns.
column_names = {'Item1':'admission', 'Item2':'treatment', 'Item3':'timely_visits'
                ,'Item4':'reliability', 'Item5':'options', 'Item6':'hours_of_treament'
                , 'Item7':'courteous', 'Item8':'active_listening'}

medical_df =  medical_df.rename(columns= column_names)  
# Print the column names.
#for col in medical_df.columns:
#    print(col)

# check to see if you can access by the new column name.
#print(medical_df['admission'].values)
#print(medical_df.admission.dtypes)

""" Look for outliers using a boxplot
"""
"""

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 12), ncols=3, sharex= True,
                                    sharey=False)
sns.boxplot(data=my_df['Age'], ax = ax1).set(title="Age")
sns.boxplot(data=my_df['Children'], ax = ax2).set(title="Children")
sns.boxplot(data=my_df['Income'], ax = ax3).set(title="Income")
plt.show()
"""

# Check to see if therea are any NULL values
# Visualize missing date.
#print(msno.matrix(medical_df, labels=True))

# Population *****************************************************************
#print(medical_df['Population'].describe())
"""sns.set(style='darkgrid')
sns.boxplot(data=medical_df['Population']).set(title="Population")
sns.stripplot(data=medical_df['Population'])
"""
"""
sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Population'], ax = ax1).set(title="Population")
sns.stripplot(data=medical_df['Population'], ax=ax2, color='green').set(title="Population")
plt.show()
"""

# Children *****************************************************************
"""
print(medical_df['Children'].describe())
print(medical_df.dtypes)

sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Children'], ax = ax1).set(title="Children")
sns.stripplot(data=medical_df['Children'], ax=ax2, color='red').set(title="Children")
plt.show()
"""

#Age *****************************************************************
"""
print(medical_df['Age'].describe())
print(medical_df.dtypes)

sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Age'], ax = ax1).set(title="Age")
sns.stripplot(data=medical_df['Age'], ax=ax2, color='indigo').set(title="Age")
plt.show()
"""
#Income *****************************************************************
"""
print(medical_df['Income'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Income'], ax = ax1).set(title="Income")
sns.stripplot(data=medical_df['Income'], ax=ax2, color='indigo').set(title="Income")
plt.show()
"""
#Vit_d *****************************************************************
"""print(medical_df['VitD_levels'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['VitD_levels'], ax = ax1).set(title="Vit D Levels")
sns.stripplot(data=medical_df['VitD_levels'], ax=ax2, color='pink').set(title="Vit D Levels")
plt.show()
"""

#Doc Visits *****************************************************************
"""
print(medical_df['Doc_visits'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Doc_visits'], ax = ax1).set(title="Doctor Visits")
sns.stripplot(data=medical_df['Doc_visits'], ax=ax2, color='orange').set(title="Doctor Visits")
plt.show()
"""


#Full meals *****************************************************************
"""
print(medical_df['Full_meals_eaten'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Full_meals_eaten'], ax = ax1).set(title="FUll Meals")
sns.stripplot(data=medical_df['Full_meals_eaten'], ax=ax2, color='orange').set(title="Full Meals")
plt.show()

"""

#Vit D Supps *****************************************************************
"""
print(medical_df['VitD_supp'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['VitD_supp'], ax = ax1).set(title="Vit D Supps")
sns.stripplot(data=medical_df['VitD_supp'], ax=ax2, color='blue').set(title="Vit D supps")
plt.show()
"""

# Initial Days ****************************************************************
"""
print(medical_df['Initial_days'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Initial_days'], ax = ax1).set(title="Initial Days")
sns.stripplot(data=medical_df['Initial_days'], ax=ax2, color='blue').set(title="Initial Days")
plt.show()
"""

# Total Charge **************************************************************
"""
print(medical_df['TotalCharge'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['TotalCharge'], ax = ax1).set(title="Total Charge")
sns.stripplot(data=medical_df['TotalCharge'], ax=ax2, color='blue').set(title="Total Charge")
plt.show()
"""

#Additional Charges***********************************************************
"""print(medical_df['Additional_charges'].describe())


sns.set(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(figsize=(12, 12), ncols=2, sharex= True,
                                    sharey=False)
sns.boxplot(data=medical_df['Additional_charges'], ax = ax1).set(title="Additional Charges")
sns.stripplot(data=medical_df['Additional_charges'], ax=ax2, color='blue').set(title="Additional Charges")
plt.show()
"""

#REmove Population Outliers

# Calculate the bound.
lower_bound = medical_df['Population'].quantile(0.25)
upper_bound = medical_df['Population'].quantile(0.75)
IQR = upper_bound - lower_bound


# Identify the outliers in the dataframe
threshold = 1.5
outliers = medical_df[(medical_df['Population']  < lower_bound - threshold * IQR ) 
                 |(medical_df['Population']  > upper_bound + threshold * IQR )] 

#print(outliers.index)
#print(outliers.shape)
#print(type(outliers))

#Drop the outliers
medical_df = medical_df.drop(outliers.index)
print(medical_df.shape)

# Children *******************************************************************
print("\n\n\nNulls in Children frame BEFORE: ", medical_df["Children"].isna().sum())

# Calculate the bound.
lower_bound = medical_df['Children'].quantile(0.25)
upper_bound = medical_df['Children'].quantile(0.75)
IQR = upper_bound - lower_bound


# Identify the outliers in the dataframe
threshold = 1.5
outliers = medical_df[(medical_df['Children']  < lower_bound - threshold * IQR ) 
                 |(medical_df['Children']  > upper_bound + threshold * IQR )] 

print(outliers.index)
print(outliers.shape)
print(type(outliers))


medical_df = medical_df.drop(outliers.index)
print("\n\n\nNulls in Children frame After: ", medical_df["Children"].isna().sum())
print(medical_df.shape)
print(medical_df.shape)
print("\n\n\nNulls in frame BEFORE: ", medical_df['Children'].isna().sum())


# Remove Children Outliers


msno.matrix(medical_df, labels=True)




