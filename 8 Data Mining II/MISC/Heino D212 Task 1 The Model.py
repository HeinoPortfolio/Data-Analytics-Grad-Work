# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 00:44:05 2024

@author: Matthew Heino


"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score


# Show all columns.
pd.set_option('display.max_columns', None)


# Read in the cleaned file
medical_df = pd.read_csv('Heino_cleaned_medical_task1.csv')

# Display what is in the dataframe.
print(medical_df.head())
print(medical_df.info())


# Create the clustering model. This model will be created using the linkage 
# function is located in the scipy.cluster.heirarch library.  
# Create the linkage which returns an array.

ward_matrix_arr = linkage(medical_df, method='ward', metric='euclidean')

"""
# Visualize the model using a dendrogram. (tree visual).
plt.figure(figsize=[15, 5])
ward_dendro = dendrogram(ward_matrix_arr)
 
# Set up the plot labels.
plt.xlabel("Patient Survey Responses")
plt.ylabel("Distance Between the Clusters")

# Show the dendrogram graph.
print(plt.show())

plt.show()
"""

medical_df["Cluster Labels"] = fcluster(ward_matrix_arr, 2, criterion='maxclust')
#print(medical_df['Cluster Labels'].value_counts().sort_index())



"""
# Visualizing the distribution of the answers to the answers to the questions.

fig, axs = plt.subplots(figsize=(20,20),nrows=4, ncols=2, sharex=False
                        , sharey=False)

fig.tight_layout(pad=5.0)

# Question 1
sns.countplot(data=medical_df, x="timely_admis_surv", ax=axs[0,0]
              , hue="Cluster Labels").set(xlabel="Timely Admission Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 1 Distribution of Scores by Cluster Label")
axs[0,0].bar_label(axs[0,0].containers[0])
axs[0,0].bar_label(axs[0,0].containers[1])

# Question 2
sns.countplot(data=medical_df, x="timely_treatment_surv", ax=axs[0,1]
              , hue="Cluster Labels").set(xlabel="Timely Treatment Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 2 Distribution of Scores by Cluster Label")
axs[0,1].bar_label(axs[0,1].containers[0])
axs[0,1].bar_label(axs[0,1].containers[1])

# Question 3
sns.countplot(data=medical_df, x="timely_visits_surv", ax=axs[1,0]
              , hue="Cluster Labels").set(xlabel="Timely Visits Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 3 Distribution of Scores by Cluster Label") 
axs[1,0].bar_label(axs[1,0].containers[0])
axs[1,0].bar_label(axs[1,0].containers[1])           

# Question 4
sns.countplot(data=medical_df, x="reliability_surv", ax=axs[1,1]
              , hue="Cluster Labels").set(xlabel="Reliability Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 4 Distribution of Scores by Cluster Label") 
axs[1,1].bar_label(axs[1,1].containers[0])
axs[1,1].bar_label(axs[1,1].containers[1])
                                          
# Question 5
sns.countplot(data=medical_df, x="options_surv", ax=axs[2,0]
              , hue="Cluster Labels").set(xlabel="Options Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 5 Distribution of Scores by Cluster Label") 
axs[2,0].bar_label(axs[2,0].containers[0])
axs[2,0].bar_label(axs[2,0].containers[1])

# Question 6
sns.countplot(data=medical_df, x="hours_of_treatment_surv", ax=axs[2,1]
              , hue="Cluster Labels").set(xlabel="Hours of Treatment Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 6 Distribution of Scores by Cluster Label")  
axs[2,1].bar_label(axs[2,1].containers[0])
axs[2,1].bar_label(axs[2,1].containers[1])                                         

# Question 7
sns.countplot(data=medical_df, x="courteous_staff_surv", ax=axs[3,0]
              , hue="Cluster Labels").set(xlabel="Courteous Staff Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 7 Distribution of Scores by Cluster Label")  
axs[3,0].bar_label(axs[3,0].containers[0])
axs[3,0].bar_label(axs[3,0].containers[1])

# Question 8
sns.countplot(data=medical_df, x="active_listening_surv", ax=axs[3,1]
              , hue="Cluster Labels").set(xlabel="Active Listening Importance"
                                          , ylabel="Number of Patients"
                                          , title="Question 8 Distribution of Scores by Cluster Label")  
axs[3,1].bar_label(axs[3,1].containers[0])
axs[3,1].bar_label(axs[3,1].containers[1])                                          


plt.show()
"""

# Section E

frame_cols = ['timely_admis_surv', 'timely_treatment_surv', 'timely_visits_surv'
              ,'reliability_surv','options_surv','hours_of_treatment_surv'
              , 'courteous_staff_surv','active_listening_surv' ]

accuracy_df = medical_df[frame_cols]

# Series of cluster groups 
cluster_sers = medical_df['Cluster Labels']


# Print Accuracy frame contents
#print(accuracy_df.info())
#print(accuracy_df.shape)
#print(type(cluster_sers))
#print(len(cluster_sers))

# Generate the silhouette score for the clusters that were created earlier.
cluster_model_score = silhouette_score(accuracy_df, cluster_sers
                                       , metric='euclidean')

# Print the silhouette score.
print("The silhouette score for the clustering: ", cluster_model_score)
print("The silhouette score for the clusterig is: {:.5f} ".format(cluster_model_score))


















