# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:19:34 2023

@author: Matthew Heino

"""
#import fancyimpute as fi
import missingno as msno
import numpy as np
import pandas as pd

from fancyimpute import KNN



# Import the Medical data set.
medical_df = pd.read_csv('medical_raw_data.csv') 
print(medical_df.head(5))
print(medical_df.dtypes) 
print(medical_df.shape)

"""
# Look for duplicates within the table.
duplicated_rows = medical_df[medical_df.duplicated()]
print("Number of duplicated rows: ", duplicated_rows.size)


# Check data types of the columns.
print(medical_df.dtypes)


"""
"""
Check for uniqueness among the columns. Will check all columns with type 
object as they may be categorical and if catergorical see if there are too 
many categories.

"""
#print(np.sort(medical_df.State.unique()))


#Check Area for uniqueness
#print(np.sort(medical_df.Area.unique()))


#Check Timezone
# will undergo category collapsing.
#print(np.sort(medical_df.Timezone.unique()))

#Check  Job
# Show a few example JOb collapsing for categories.
#print(np.sort(medical_df.Job.unique()))

#Check on education
#print(np.sort(medical_df.Education.unique()))

#check on employment
#print(np.sort(medical_df.Employment.unique()))

#check on marital
#print(np.sort(medical_df.Marital.unique()))

#Check on gender
#print(np.sort(medical_df.Gender.unique()))

#Check on Services
#print(np.sort(medical_df.Services.unique()))


"""
Convert Columns to more appropriate data type.
Children, Age will converted to integer and Zipcode will changed to a 
string and fixed to deal with missing zero
"""
"""
#Convert Zipcode from int64 to a String to get back leading zero in Zipcode.
medical_df = medical_df.astype({'Zip': str})
# Check data types of the columns.
print(medical_df.Zip.dtypes)

#Check the Zip code to appropriate length.
#Does not return appropriate length.
print(medical_df.Zip.str.len().min())

#add appropraite amount of zeroes to the left of the Zip string.

medical_df['Zip'] = medical_df['Zip'].str.zfill(5)
print("Min Zip: ",medical_df.Zip.str.len().min())
print("Max Zip: ",medical_df.Zip.str.len().max())
print(medical_df['Zip'].str.startswith('0'))
print("Zip Type: ",medical_df.Zip.dtypes)



# Give the items more appropriate names for columns.

column_names = {'Item1':'admission', 'Item2':'treatment', 'Item3':'timely_visits'
                ,'Item4':'reliability', 'Item5':'options', 'Item6':'hours_of_treament'
                , 'Item7':'courteous', 'Item8':'active_listening'}

medical_df =  medical_df.rename(columns= column_names)  
# Print the column names.
for col in medical_df.columns:
    print(col)

# check to see if you can access by the new column name.
print(medical_df['admission'].values)
print(medical_df.admission.dtypes)



"""
"""
# Will need to fill in values for Children and Age
medical_df = medical_df.astype(convert_dict)   
# Check data types of the columns.
print(medical_df.dtypes)

"""
"""
print(medical_df.dtypes)
convert_dict = {'Children': "Int64",
                'Age': "Int64"}

medical_df = medical_df.astype(convert_dict)
print("Children data type: ", medical_df.Children.dtypes)
print("Age data type: ", medical_df.Age.dtypes)
print("Children values:", medical_df.Children.values)
print("Age values: ", medical_df.Age.values)

print("Number of NULLS: \n", medical_df.isna().sum())




# Visualize missing date.
print(msno.matrix(medical_df, labels=True))

print(msno.dendrogram(medical_df))


#  Fill in the empty values
# Children.

medical_df['Children'] =  medical_df['Children'].fillna(medical_df['Children'].median())
print("Number of NULLS: \n", medical_df.isna().sum())

#Age
medical_df['Age'] =  medical_df['Age'].fillna(medical_df['Age'].median())
print("\n\nNumber of NULLS: \n\n", medical_df.isna().sum())

#Income
medical_df['Income'] =  medical_df['Income'].fillna(medical_df['Income'].median())
print("\n\nNumber of NULLS: \n", medical_df.isna().sum())


#Initial days
medical_df['Initial_days'] =  medical_df['Initial_days'].fillna(medical_df['Initial_days'].median())
print("\n\nNumber of NULLS: \n", medical_df.isna().sum())

# Visualize missing date.
print(msno.matrix(medical_df, labels=True))




#Soft_drink
print(medical_df['Soft_drink'].head(5))


medical_df['Soft_drink'] =  medical_df['Soft_drink'].fillna('No')
print("\n\nNumber of NULLS: \n", medical_df.isna().sum())


# Visualize missing date.
print(msno.matrix(medical_df, labels=True))


# 
#Overweight
print("\n\n\n ***Overwight:", medical_df['Overweight'].head(5))
medical_df['Overweight'] = medical_df['Overweight'].astype(str)
print("Overweight data type: ", medical_df.Overweight.dtypes)

medical_df['Overweight'] = medical_df['Overweight'].replace(to_replace="1.0"
                                                            , value ='Yes')
medical_df['Overweight'] = medical_df['Overweight'].replace(to_replace="0.0"
                                                            , value ='No')
medical_df['Overweight'] = medical_df['Overweight'].fillna('No')




medical_df['Anxiety'] = medical_df['Anxiety'].replace(to_replace="1"
                                                      , value ='Yes')
medical_df['Anxiety'] = medical_df['Anxiety'].replace(to_replace="0"
                                                      , value ='No')
medical_df['Anxiety'] = medical_df['Anxiety'].fillna('No')

print("\n\nNumber of NULLS-FIANL****: \n", medical_df.isna().sum())

# Visualize missing date.
print(msno.matrix(medical_df, labels=True))

"""
"""
Ancillary Sites:
    
    
https://www.geeksforgeeks.org/change-data-type-for-one-or-more-columns-in-pandas-dataframe/  

https://pandas.pydata.org/docs/reference/api/pandas.Series.str.zfill.html

https://pandas.pydata.org/docs/reference/api/pandas.Series.str.startswith.html

https://bobbyhadz.com/blog/pandas-find-percentage-of-missing-values-in-each-column

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
"""
