# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 23:49:52 2024

@author: Matthew Heino

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

feature_cols = {'Lat','Lng','Children', 'Age','Income', 'VitD_levels', 'Doc_visits'
                ,'Full_meals_eaten', 'vitD_supp', 'Initial_days'
                , 'TotalCharge', 'Additional_charges'} 

medical_df = pd.read_csv('medical_clean.csv', usecols=feature_cols)

# Print some information about  the data that is in the dataframe
#print(medical_df.head())
#print(medical_df.info())
med_cols = medical_df.columns
#print(med_cols)



# *****************************************************************************
# Step C2. Standardize the data and then output to the standarized file to a CSV file
# Create the scaler object.
std_scaler = StandardScaler()

# Fit the data using the scaler.
std_scaler.fit(medical_df)

# Create a new dataframe that has the scaled data.
# pass in the names of the columns to be associated with the data.

medical_scaled_df = pd.DataFrame(std_scaler.transform(medical_df)
                                 , columns=med_cols)

# Check to see if the dataframe was created and it contains the right data.
# Transpose the data for a different look.
#print(medical_scaled_df.head().T) 
#print(medical_scaled_df.info())
#print(medical_scaled_df.shape)


# Output or export the standardized data to a CSV file.
#medical_scaled_df.to_csv('Heino_D212_Task2_Standardized_Data.csv'
 #                        , index=False, header=True) 



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
#n_comps = medical_scaled_df.shape[1]
#print(n_comps)


# Create and instantiate the PCA object.
pca = PCA(n_components=medical_scaled_df.shape[1], random_state=247)

# Fit the PCA to the standardized data (medical_scaled_df)
med_pca = pca.fit_transform(medical_scaled_df)

# Create the matrix of the PCA components.
pca_matrix = pd.DataFrame(pca.components_.T,columns=col_list
                          , index=med_cols)

print(pca_matrix)

#print(pca_matrix.shape[1])
print("\n\n")

"""
# Create a heatmap for the loadings of the
plt.figure(figsize=(20,20))
sns.heatmap(pca_matrix , cmap='mako',annot=True, fmt='.3g')
plt.title('Principal Component Matrix')
plt.show() 



#*****************************************************************************
# D2. Create the Elbow plot.
# Retrieve the total varaince explained by the given list of principle 
# components.  This can be found in the explained_varaince_ratio found in 
# the PCA object.

"""
total_var = sum(pca.explained_variance_ratio_*100).round()

print("Total variance explained by all {} of the principal components is: {}%"
      .format(pca_matrix.shape[1], total_var))


# Create a dataframe with the percent of the variance captured.
#print(pca.explained_variance_ratio_ * 100)
#print(col_list)



var_perc = (pca.explained_variance_ratio_ * 100).round(2)

#print(var_perc)
variance_df = pd.DataFrame(var_perc,columns=["Captured Variance by PC"]
                           , index=col_list)


#print(variance_df)


#Retrieve the eigenvalues and add to the frame.
eigenvalues = pca.explained_variance_

#print(eigenvalues)



# Add to the frame.
variance_df["Eigenvalues by PC"] = eigenvalues

# print the dataframe.
#print(variance_df)



# Calculate the cumulative sum of the variances 
pc_sum = np.cumsum(pca.explained_variance_ratio_*100)

#print(pc_sum)



# Create a dataframe to make displaying more pleasing.
pc_sum_df = pd.DataFrame(pc_sum, columns=["Sum of the Cumulaitve Varainces by PC"]
                         ,index=col_list)



#print(pc_sum_df)


# Add to the cumaltive sum of the variances to the dataframe.
variance_df["Sum of the Cumulative Variances by PC"] = pc_sum

#print(variance_df)

# Reindex the columns to make it easier to read.
variance_df = variance_df.iloc[ : ,[0,2,1]]

print(variance_df)


# Create the required Scree plot.
fig, ax = plt.subplots(figsize=(15, 15))
plt.plot(pc_sum)
plt.title("The Scree Plot of all the PCs")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Experience (%)")
ax.set_xticks(range(0, medical_scaled_df.shape[1]))
ax.set_xticklabels(col_list)

# Add some percent lines to the graph
plt.axhline(y = 50, color='r', linestyle='--', label='50% Variance')
plt.axhline(y = 60, color='y', linestyle='--', label='60% Variance')
plt.axhline(y = 75, color='c', linestyle='--', label='75% Variance')
plt.axhline(y = 85, color='m', linestyle='--', label='85% Variance')
plt.axhline(y = 100, color='g', linestyle='--', label='100% Variance')

# Add a legend to the plot.
plt.legend(loc='lower right')

plt.show()


# Create a heatmap for the loadings of the
plt.figure(figsize=(20,20))
sns.heatmap(pca_matrix , cmap='mako',annot=True, fmt='.3g')
plt.title('Principal Component Matrix')
plt.show() 



# code adapted from "PCA Explained Variance Concepts with Python Example"
# (Kumar, 2023)

# Graph of the cumulative variances by PC number.
var_pca = pca.explained_variance_ratio_ * 100

var_pca_rnd = var_pca.round(2)

cumualitve_eigenvalues = np.cumsum(var_pca)

labels = [f"{i}" for i in var_pca_rnd]


# Set up the graph.
fig, ax = plt.subplots(figsize=(15,20))
plt.bar(range(0, len(var_pca)), var_pca, alpha=0.75, align='center'
        , tick_label=col_list, label="Individual Explained Variance")

plt.step(range(0, len(cumualitve_eigenvalues)), cumualitve_eigenvalues
         , where='mid', label='Cumulative Explained Variance')   

plt.title('Cumulative Variance Graph')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Varaince (%)')

# Add some variance percent lines.
plt.axhline(y = 60, color='y', linestyle='--', label='60% Variance')
plt.axhline(y = 75, color='c', linestyle='--', label='75% Variance')
plt.axhline(y = 85, color='m', linestyle='--', label='85% Variance')
plt.axhline(y = 100, color='g', linestyle='--', label='100% Variance')

# add some labels
rectangles = ax.patches

labels2 = variance_df['Sum of the Cumulative Variances by PC'].tolist()

# Create the labels
for rect, label in zip(rectangles, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2 , height / 2, str(label) + ' %', 
        ha='center', va='bottom', color='white'
        
        )



# Create lables for the step plot.
for rect2, label2 in zip(rectangles, labels2):
  
   ax.text(rect2.get_x(), label2 + 2 , str(round(label2, 2)) + ' %'
           , ha='center', color='red'
   
   )
   

# Add a legend on the graph.
plt.legend(loc='best')

plt.show()










































