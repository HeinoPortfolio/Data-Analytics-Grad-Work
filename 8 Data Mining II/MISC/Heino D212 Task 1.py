# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:07:02 2024

@author: Matthew Heino

Task 1: Hierarchical Clustering 


Note: This file makes use of the medical_clean.csv. Not all columns will be 
used to create this model or be used in the assessment.

"""

import pandas as pd


# Show all columns.
pd.set_option('display.max_columns', None)

# Read in the data from the CSV file.
# Read on only the columns that will be used to create the clustering model.


# Read in only the columns that are required for the assessment.
feature_cols = {'Item1', 'Item2','Item3', 'Item4', 'Item5'
                ,'Item6', 'Item7', 'Item8' } 

medical_df = pd.read_csv('medical_clean.csv', usecols=feature_cols)

print(medical_df.info())
#print(medical_df.head())
#print(medical_df.head())

# Change the column names to more informative names
col_names = {'Item1' : 'timely_admis_surv', 'Item2' :'timely_treatment_surv' 
             ,'Item3' :'timely_visits_surv' , 'Item4': 'reliability_surv' 
             , 'Item5' : 'options_surv','Item6' : 'hours_of_treatment_surv'
             , 'Item7' : 'courteous_staff_surv'
             , 'Item8' : 'active_listening_surv' } 

# Rename the columns using the dictionary.
# Renaming the columns will happen inplace. (Awan, 2022)
medical_df.rename(columns=col_names, inplace=True)

# Print the column names to check.
#print(medical_df.columns)
#print(medical_df.head())
#print(medical_df.head())

#print ("\n\n\n")


# Change the scale to make it reversed so it ranks the surveys in the a 
# more intuitive manner.
# These will be remapped in the following manner.
# (pandas.Series.map — Pandas 2.1.4 Documentation, n.d.)
survey_map ={1 : 8, 2 : 7, 3 : 6, 4 : 5, 5 : 4, 6 : 3, 7 : 2, 8 : 1}

# Map the columns to the columns new values
medical_df['timely_admis_surv'] = medical_df['timely_admis_surv'].map(survey_map)
medical_df['timely_treatment_surv'] = medical_df['timely_treatment_surv'].map(survey_map)
medical_df['timely_visits_surv'] = medical_df['timely_visits_surv'].map(survey_map)
medical_df['reliability_surv'] = medical_df['reliability_surv'].map(survey_map)
medical_df['options_surv'] = medical_df['options_surv'].map(survey_map)
medical_df['hours_of_treatment_surv'] = medical_df['hours_of_treatment_surv'].map(survey_map)
medical_df['courteous_staff_surv'] = medical_df['courteous_staff_surv'].map(survey_map)
medical_df['active_listening_surv'] = medical_df['active_listening_surv'].map(survey_map)


# Check to see that the columns have been reordered/remapped. 
print(medical_df.head())


# Change the datatype to be able to use the data in the clustering model.
# Cast all the columns as float64. (GeeksforGeeks, 2023)

#medical_df = medical_df.astype('float64')
#print(medical_df.dtypes)
#print(medical_df.info())
#print(medical_df.head())
#print(medical_df.tail())


# C4. The Prepared dataset.
# The outputting the prepared dataset. 

# Save the cleaned dataframe to a CSV file.
medical_df.to_csv('Heino_cleaned_medical_task1.csv', index = False, header = True)

























