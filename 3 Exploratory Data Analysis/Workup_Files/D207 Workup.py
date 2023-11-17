# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 23:31:37 2023

@author: Matthew Heino

Purpose:    Workup for the D207 assessment.  Code will be used in a Jupyter 
            Notebook that will be submitted in with the assessment document
            
"""

# Packages that will be needed for the assignment.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency


"""
    Pre-assessment Tasks:
        
        1) Read in the CSV file from the fime: medical_clean.csv.
        
        2) Explore the data in the file to see what it contains.
    
        3) Get some statistics about the dataframe.
        
        4) Display the first five rows of the dataframe.  To show what 
        the  contains.

"""

# Read in the CSV file into a pandas dataframe.

medical_df = pd.read_csv('medical_clean.csv')



# Get some info about the medical_df dataframe.

#print(medical_df.info())

# No nulls were found.

# Show the contents of the first five rows.
#print(medical_df.head(5))


"""
Section B:
    
    Section B tasks:
        
        1) Perfroming test
        
        
        2) Show the  output of the results.
    
"""
# https://pythonfordatascienceorg.wordpress.com/chi-square-python/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

# Using crosstab and the counts ReAdmis and Overweight

cross_table = pd.crosstab(medical_df.ReAdmis,medical_df.Overweight)

print("\nThe Contents of the cross table: \n ", cross_table)

# Conduct the chi-squared using chi2_cintingency.

alpha = 0.05

result = chi2_contingency(cross_table)
print("\nResults: ", result)

print("\nThe p-value  is the following:", result[1])
if result[1] > alpha:
    print("The null hypothesis is accepted!")
else:
    print("The null hypothesis is rejected!")



"""

Section C1:
    
    Section C1 tasks:
        
        1) Univariate distributions for two continuous variables.  
        Graph the variables using the appropirate graph.
        
        - Income
        - Total Charge
        
        2)  Univariate distributions for two continuous variables.  
        Graph the variables using the appropirate graph.
        
        - Gender
        - Complication_risk

"""

"""
# Continuous variable distribution. Using Income and TotalCharge columns

sns.set(style='darkgrid')

fig, (ax1, ax2) = plt.subplots(figsize =(12, 12), ncols=2, 
                               sharex=False, sharey=False)

# Added Kernal dsitribution ((Holtz, n.d.))
sns.histplot(data=medical_df, x='Income', ax=ax1, kde=True).set(title='Distribution of Patient Income'
                                                      , xlabel='Income of the Patients'
                                                      , ylabel='Count of Patients In Each Income')

sns.histplot(data=medical_df, x='TotalCharge', ax=ax2, kde=True).set(title='Distribution of Total Charge (Avg)'
                                                      , xlabel='Total Charge of the Patients'
                                                      , ylabel='Count of Patients Total Charge')

plt.suptitle("Univariate Continuous Variable Distribution")
plt.show()



# Categorical Variables dstiribution

sns.set(style='darkgrid')

fig, (ax1, ax2) = plt.subplots(figsize =(12, 12), ncols=2, 
                               sharex=False, sharey=False)


sns.histplot(data=medical_df, x='Gender', ax=ax1, hue=medical_df['Gender'], kde=False).set(title='Distribution of Gender'
                                                      , xlabel='Gender of the Patients'
                                                      , ylabel='Count of Patients In Gender')

sns.histplot(data=medical_df, x='Complication_risk', ax=ax2,hue=medical_df['Complication_risk'], kde=False).set(title='Distribution of Complication Risk'
                                                      , xlabel='Complication Risk of the Patients'
                                                      , ylabel='Count of Patients Complication Risk')


plt.suptitle("Univariate Categorical Variable Distribution")
plt.show()


fig, (ax1) = plt.subplots(figsize =(12, 12), ncols=1, 
                               sharex=False, sharey=False)

sns.scatterplot(data=medical_df, x="Age", y="Initial_days",hue='Gender', ax=ax1).set(title='Age vs. Initial Day'
                                                      , xlabel='Age of the Patients'
                                                      , ylabel='Initial Days')

plt.suptitle("Bivariate Continuous Variable Distribution")
plt.show()

fig, (ax1) = plt.subplots(figsize =(12, 12), ncols=1, 
                               sharex=False, sharey=False)

sns.scatterplot(data=medical_df, x="Age", y="Initial_days",hue='Gender', ax=ax1).set(title='Age vs. Initial Day'
                                                      , xlabel='Age of the Patients'
                                                      , ylabel='Initial Days')

plt.suptitle("Bivariate Continuous Variable Distribution")
plt.show()


fig, (ax1) = plt.subplots(figsize =(12, 12), ncols=1, 
                               sharex=False, sharey=False)

sns.scatterplot(data=medical_df, x="Age", y="Initial_days",hue='Gender', ax=ax1).set(title='Age vs. Initial Day'
                                                      , xlabel='Age of the Patients'
                                                      , ylabel='Initial Days')

plt.suptitle("Bivariate Continuous Variable Distribution")
plt.show()


#sns.set(style='darkgrid', {'grid.color' : 'black'})

sns.set_style("darkgrid", {'grid.color':'black' ,'grid.linestyle': '--'})

fig, (ax1) = plt.subplots(figsize =(12, 12), ncols=1, 
                               sharex=False, sharey=False)

bi_cat = sns.barplot(data=medical_df, x="Item1", y="Item2").set(title="Timely Admission vs Time Treatment", xlabel='Timely Admission'
                                                     , ylabel='Timely Treatment')



plt.suptitle("Bivariate Categorical Variable Distribution")
plt.legend(loc='upper left')
plt.show()


# Using Stroke and HIghBlood pressure
# Cross tabulation of HIgh Blood Pressure and Stroke
# (Hashmi & Hashmi, n.d.)
tabResult = pd.crosstab(index=medical_df['HighBlood']
                        , columns=medical_df['Stroke'])

# check the values.
print(tabResult)

print(tabResult.plot.bar(figsize=(12,12), rot = 0,
                         xlabel='High Blood Pressure',
                         ylabel='Count of the Categoricals'))



"""





