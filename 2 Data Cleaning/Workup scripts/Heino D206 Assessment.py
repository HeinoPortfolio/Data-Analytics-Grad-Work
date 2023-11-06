# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:42:28 2023

@author: Matthew E. Heino

Course:         D206 Data Cleaning
Instructor:     Dr. K. Middleton
Purpose:        Script file for answering the code based requirements of the 
                assessment.
            

Note:           This script assumes that the data file (medical_raw_data.csv) 
                corruently stored in the local directory 

 
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:19:34 2023

@author: Matthew Heino

"""
#import fancyimpute as fi
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

from fancyimpute import KNN



# Import the Medical data set.
medical_df = pd.read_csv('medical_raw_data.csv') 
#print(medical_df.head(5))
#print(medical_df.dtypes) 
#print(medical_df.shape)

""""
Step 1: Remove Duplicate or unneeded elements from the dataframe
    
    Actions:
        
        1) Drop the first row of the datset.  Information is redundant and 
        not needed.
        
        2) Look for duplicates in each column of the dataframe.

    Note: This file assumes that the data file corruently stored in the local 
    directoy
"""

medical_df.drop(medical_df.columns[0], axis=1, inplace = True)

# Check to see that the columns was successfully dropped.
#print(medical_df.head(5)) 
#print(medical_df.shape)


""" **************************************************************************

Step 2:     Fix formatting and inconsistencies in the data.  
            
    Note:   This corresponds to section D of the accompanying document. 

    Actions:
        
        1) Look for unique categories in the each of the categorical columns 
        of the dataframe.
        
        2) Reduce  the number of categories where appropriate.
    
        3) Change the data type of columns to a more appropriate type.
    

************************************************************************** """

# Check for uniqueness among the columns. Will check all columns with type 
# object as they may be categorical and if catergorical see if there are too 
# many categories.

# Check unique Customer IDs)
#print("\nUnique Customer IDs: ", medical_df.Customer_id.unique().size)
# Only unique customer IDs.


# Check unique Interactions.
#print("\nUnique Interactions: ", medical_df.Interaction.unique().size)
# Only unique Interactions.

# Check unique UIDs.
#print("\nUnique UIDs: ", medical_df.UID.unique().size)
# Only unique UIDs.


# City will be skipped 
# Diffcult to for unique values 
# Check City for uniqueness
#print("\nUnique cities",np.sort(medical_df.City.unique()))
#print("Unique Cities size: ", medical_df.City.unique().size)

 
# Check state for unique values 
#print("\nUnique states",np.sort(medical_df.State.unique()))
#print("Unique states size: ", medical_df.State.unique().size)
# only 52 unique


# County will be skipped 
# Diffcult to for unique values 
# Check County for uniqueness and valid entries

#print("\nUnique Counties",np.sort(medical_df.County.unique()))
#print("Unique Counties size: ", medical_df.County.unique().size)

""" **************************************************************************
    
    ZIP code checked for right length.
    This will show an error since this is currently stored as an integer
    This error will be noted in the assessment document in greater detail.
    Note this corresponds to Section D2 of the included document.

******************************************************************************
"""

print("\nMinimum ZIP value: ", medical_df['Zip'].min())
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
#print("\nDatatype of the data set ",medical_df.dtypes)    
     
#Check Area for uniqueness

#print("\nUnique values of Area: ",np.sort(medical_df.Area.unique()))     
# All values are acceptable      
      
# Check Timezone
# Will undergo category collapsing.


""" **************************************************************************
    
    Tmezone checked for the amount of timezone categories.
    Found to have 26 will be reduced to 7 timezones.  
    Note this corresponds to Section D2 of the included document.

******************************************************************************
"""
#print(np.sort(medical_df.Timezone.unique())) 
#print("\nUique Time Zones Before change :",medical_df.Timezone.unique().size)  


timezone_replace_dict = {'America/Adak': 'Hawaii-Aleutian', 
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
#print("Uique Time Zones after change :",np.sort(medical_df.Timezone.unique())) 
 
#Check Job
#print("\n Jobs: ",np.sort(medical_df.Job.unique()))
#print("Unique Job Titles: ",medical_df.Job.unique().size)


"""
#  Insert   CODE HERE ???????????????????????????????????????????????????????
#       ???????????????????????????????????????
"""

# Conversion of Children and Age column to a data type that is more apprpriate.
print(medical_df[['Children','Age']].dtypes)
convert_dict = {'Children': "Int64",
                'Age': "Int64"
    }

medical_df = medical_df.astype(convert_dict)
#print("\n\nChildren data type (Current): ", medical_df.Children.dtypes)
#print("Age data type (Current): ", medical_df.Age.dtypes)

# Check for the range of values to make sure 
#print("Children minimum value (Current): ", medical_df.Children.min())
#print("Age maximum value (Current): ", medical_df.Age.max())

# check the uniqueness and values that are stored in Education column
#print("\nUnique Education categories :",medical_df.Education.unique())
 
# check the uniqueness and values that are stored in Employment column
#print("\nUnique Employment categories :",medical_df.Employment.unique())

# check the uniqueness and values that are stored in Income column
#print("\nMinimum income :", medical_df.Income.min())
#print("Maximum income :", medical_df.Income.max())

# check the uniqueness and values that are stored in Marital column
#print("\nUnique Marital categories :",medical_df.Marital.unique())

# check the uniqueness and values that are stored in Gender column
#print("\nUnique Gender categories :",medical_df.Gender.unique())

# check the uniqueness and values that are stored in ReAdmis column
#print("\nUnique ReAdmis categories :",medical_df.ReAdmis.unique())

# Check of the doctor vistits
#print("\nVisits minimum value: ", medical_df.Doc_visits.min())
#print("Visits maximum value: ", medical_df.Doc_visits.max())

# Check of the full meals
#print("\nFull meals minimum value: ", medical_df.Full_meals_eaten.min())
#print("Full meals value: ", medical_df.Full_meals_eaten.max())

# Check of the Vitamin supplements count
#print("\nVit D minimum value: ", medical_df.VitD_supp.min())
#print("Vit D maximum value: ", medical_df.VitD_supp.max())

# check the uniqueness and values that are stored in Soft drink column
# note there are missing values nan within this column.
#print("\nUnique Soft drinks categories :",medical_df.Soft_drink.unique())

# check the uniqueness and values that are stored in initial admission column
# note there are missing values nan within this column.
#print("\n\nUnique Soft Initial admission categories : ", medical_df.Initial_admin.unique())

# check the uniqueness and values that are stored in initial admission column
#print("\n\nUnique High Blood pressure categories : ", medical_df.HighBlood.unique())

# check the uniqueness and values that are stored in initial admission column
#print("\n\nUnique Stroke categories : ", medical_df.Stroke.unique())

# check the uniqueness and values that are stored in initial admission column
#print("\n\nUnique Complication risk categories : ", medical_df.Complication_risk.unique())

# check the uniqueness and values that are stored in initial admission column
#print("\n\nUnique Overweight categories : ", medical_df.Overweight.unique())



print('\n\nTotal nulls before change: ', medical_df.isnull().sum())

repl_dict = {'0.0' : 'No', '1.0' : 'Yes',np.NaN : np.NaN }
#print(medical_df['Overweight'].isnull().sum())
medical_df['Overweight'] = medical_df['Overweight'].astype(str)



medical_df['Overweight'].replace(repl_dict, inplace=True)
#print("\n\nUnique Overweight categories : ", medical_df.Overweight.unique())
#print("Overweight data type (Current): ", medical_df.Overweight.dtypes)
medical_df['Overweight'].replace('nan',np.NaN, inplace=True)


# Convert some of the columns to categorical values
data_dict = {'Education': 'category','Employment' :'category',
             'Marital' :'category', 'Gender': 'category', 'ReAdmis' :'category'
             ,'Soft_drink' : 'category', 'Initial_admin' : 'category',
             'HighBlood' : 'category', 'Stroke' : 'category',
             'Complication_risk' :'category', 'Overweight' :'category'}

medical_df = medical_df.astype(data_dict)
medical_df.info()



print(medical_df['Overweight'].head())
print("\n\nUnique Overweight categories : ", medical_df.Overweight.unique())
print(medical_df.isnull().sum())



#new_df = medical_df[ medical_df['Overweight'].isnull()]

#print(new_df)

# FINISH ??????????????????????????????????????????????????????????????????



# Section D5 of the document *************************************************
# write the dataframe to a file
#medical_df.to_csv('cleaned_csv.csv') 
      
      
      
