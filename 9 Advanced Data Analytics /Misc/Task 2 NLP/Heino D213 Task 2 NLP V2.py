# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:36:38 2024

@author: ntcrw
"""
import matplotlib.pyplot as plt
#import tensorflow as tf
import nltk
import pandas as pd
import re 
#import seaborn as sns

pd.set_option('display.max_columns', 500)

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer



# Read in the data from the three files. **************************************
# Note: there are no headers on these files, so the header argument will set 
# to None


# Read in the data from the three files. **************************************


# Note: there are no headers on these files, so the header argument will set 
# to None
col_names = ['text','label']

# Amazon. **********************************************************************
amazon_df =  pd.read_csv('amazon_cells_labelled.txt', sep='\t', names=col_names
                         , header=None)

print(amazon_df.info())

# No null values.


# IMDB. ***********************************************************************
imdb_df =  pd.read_csv('imdb_labelled.txt', sep='\t', names=col_names
                         , header=None)

print(imdb_df.info())
"""
nan_values = imdb_df[imdb_df.isnull().any(axis=1)]

print(nan_values)

print(imdb_df.iloc[117])

imdb_df.drop(117, inplace=True)

print(imdb_df.info())


nan_values = imdb_df[imdb_df.isnull().any(axis=1)]

print(nan_values)

# Set the value of the label for the missing value.
imdb_df.at[716, 'label'] = 1

nan_values = imdb_df[imdb_df.isnull().any(axis=1)]

print(nan_values)


print(imdb_df.info())

"""



