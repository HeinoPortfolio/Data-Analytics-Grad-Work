# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:07:12 2023

@author: Matthew Heino
"""
import pandas as pd
#import numpy as np

pd.set_option('display.max_columns', 11)
# Read the file into the pandas dataframe

col_names = ['SEQN','RIAGENDR','RIDAGEYR','DMDHHSZA', 'DMDHHSZB']
ren_names ={'RIAGENDR': 'Gender', 'RIDAGEYR'  : 'Age'}

demo_df  =  pd.read_csv('demographic.csv', usecols=col_names)

# Rename the columns
# To make them more descriptive nad more friendly.
demo_df.rename(columns=ren_names, inplace=True)

#Combine the two columns for children.
demo_df['ChildrenCDC'] = demo_df['DMDHHSZA'] + demo_df['DMDHHSZB']

#Drop the columns that are note needed.
drop_cols = ['DMDHHSZA','DMDHHSZB']
demo_df.drop(drop_cols, axis=1, inplace = True)


# Map Gender to prepalce the 1 and 2 values that were 
# used to encode the data.
gender_map = {1: 'Male', 2: 'Female'}
demo_df['Gender'] = demo_df['Gender'].map(gender_map)

#print(demo_df)
#print(demo_df.info())


# Read in the questionaire File
""" 
    
    BPQ080 -- Hyperlipidemia
    MCQ035 -- Asthma
    MCQ160A -- Arthritis
    DIQ010 -- Diabetes
    BPQ020 -- High Blood Pressure
    MCQ080 -- Overweight
    MCQ160F -- Stroke
    
"""
ques_col_names = ['SEQN', 'BPQ080', 'MCQ035','DIQ010','MCQ160A','BPQ020'
                  ,'MCQ080', 'MCQ160F']
ques_df = pd.read_csv('questionnaire.csv', usecols=ques_col_names)


#Rename  the columns
ren_ques_names = {'BPQ080' : 'Hyperlipidemia', 'MCQ035': 'Asthma'
                  ,'DIQ010' : 'Arthritis', 'MCQ160A': 'Diabetes'
                  ,'BPQ020' : 'High Blood Pressure', 'MCQ080' : 'Overweight'
                  , 'MCQ160F' : 'Stroke'}

# Rename the columns
# To make them more descriptive nad more friendly.
ques_df.rename(columns=ren_ques_names, inplace=True)

# Fill all the NaN vlaues with a false value.

ques_df.fillna(value= 2.0, inplace=True)

#Map new values to the numerics
yes_no_map = {1.0: 'Yes', 2.0: 'No'}

ques_df['Hyperlipidemia'] = ques_df['Hyperlipidemia'].map(yes_no_map)
ques_df['High Blood Pressure'] = ques_df['High Blood Pressure'].map(yes_no_map)
ques_df['Arthritis'] = ques_df['Arthritis'].map(yes_no_map)
ques_df['Asthma'] = ques_df['Asthma'].map(yes_no_map)
ques_df['Overweight'] = ques_df['Overweight'].map(yes_no_map)
ques_df['Diabetes'] = ques_df['Diabetes'].map(yes_no_map)
ques_df['Stroke'] = ques_df['Stroke'].map(yes_no_map)

#Fill the NaN values
ques_df.fillna("No", inplace=True)

# Check the number of unique values

uni_cols = ['Hyperlipidemia', 'Asthma','Arthritis', 'Diabetes'
            ,'High Blood Pressure', 'Overweight', 'Stroke']

for col in uni_cols:
    print("\n UNIQUE", ques_df[col].unique())


#print(ques_df.info())
#print(ques_df)


# Join the two data frames on the SEQN number.
new_data_df = demo_df.merge(ques_df, on="SEQN")

print("\nThe frame: \n", new_data_df.head())
print("\n\nInfo:\n", new_data_df.info())



# Write the prepared file to a CSV file for use.
new_data_df.to_csv('Cleaned_CDC_File_Task_1.csv', index = False, header = True)





















































