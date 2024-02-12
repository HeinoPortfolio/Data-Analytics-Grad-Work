# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 04:32:33 2024

@author: Matthew Heino
"""

import matplotlib.pyplot as plt
import nltk
import pandas as pd
#import re 
import seaborn as sns
import tensorflow as tf
import warnings

#from datetime import datetime
from keras import models, layers

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences as ps
from keras.callbacks import EarlyStopping
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.models import Sequential as tsf
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)

#nltk.download('stopwords')

#nltk.download('punkt')

#nltk.download('wordnet')



###############################################################################
# Functions Start Here. #######################################################
###############################################################################

def create_wordclud(plot_title: str, plot_data, color='blue' ):
    print("INCLOUD")
    

    # Create a new word cloud.
    word_cloud = WordCloud(stopwords=STOPWORDS, background_color=color
                           , width=1500
                           , height=1500).generate(plot_data)
    
    # Plot the wordCloud figure.
    plt.figure(1, figsize=(20, 20))
    
    # show the wordcloud.
    plt.imshow(word_cloud)
    
    #Turn the axis off.
    plt.axis('off')
    
    # show the plot.
    plt.show()

###############################################################################

def find_common_words(num_common: int, col_name : str) -> pd.Series : 
    
   # print("IN common!")
    
    all_words = " ". join([word for word in master_df[col_name]])
    
    # Tokenize the words. 
    words = nltk.word_tokenize(all_words)
    
    freq_dist = FreqDist(words)
    
    top_common_words = freq_dist.most_common(num_common)
    
    
    # Create a dictionary with the word as a key and the count as the value.
    dist = pd.Series(dict(top_common_words))
    
    
    #print(" \n\nDIST type: ",type(dist))
    
    return dist      #None

def get_sequential_model(vocab_size: int, opt_dim : int, max_len : int
                         , drop : float ):
    
    seq_model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=vocab_size
                                    , output_dim=opt_dim, input_length=max_len)
                                     ,layers.Dropout(drop)
                                     ,tf.keras.layers.Flatten()
                                     ,tf.keras.layers.Dense(1, activation='sigmoid') ])
    

    return seq_model

##############################################################################
###############################################################################

def look_at_connotation( review_df : pd.DataFrame(), word_list : list) -> None:
    
    """ Method to look at the connotation of  list of words that could have 
        both a good and basd connotartion based on usage.
   
   Parameters:
    
       rev_df (Datafarme):     Dataframe with the review text.
       word_list(list):        List with the words that can have good and bad 
                               connotation.
    
   Returns:
      None
       
   """

    for wrd in word_list:
        
        print("\nGood Connotation: ", wrd)
        print(master_df[(master_df['text'].str.contains(wrd) >= 1) 
                        & (master_df['label'] == 1)].head(2))

        print("\nBad Connotation: ", wrd)
        print(master_df[(master_df['text'].str.contains(wrd) >= 1) 
                & (master_df['label'] == 0)].head(2))
        print("\n ######################################")

    print("\n\n")

###############################################################################

def remove_punctuation(rev_text: str) -> str:
    
    punct_removed = "".join(rem for rem in rev_text if rem not in("?",".",";"
                            ,":", "!",'"',",",))
    
    return punct_removed

###############################################################################





###############################################################################
###############################################################################



###############################################################################
# Read in the data from the three files. **************************************
# Note: there are no headers on these files, so the header argument will set 
# to None
col_names = ['text','label']

# Amazon. **********************************************************************
amazon_df =  pd.read_csv('amazon_cells_labelled.txt', sep='\t', names=col_names
                         , header=None)

#print(amazon_df.info())


# IMDB. ***********************************************************************
imdb_df =  pd.read_csv('imdb_labelled.txt', sep='\t', names=col_names
                         , header=None)

#print(imdb_df.info())

# Yelp. ***********************************************************************

# IMDB. ***********************************************************************
yelp_df =  pd.read_csv('yelp_labelled.txt', sep='\t', names=col_names
                         , header=None)

#print(yelp_df.info())

# Concatenate the Dataframe into one data frame and set the index.*************
master_df = pd.concat([amazon_df, imdb_df, yelp_df])

# Reset the index to make it easier to reference in the future, otherwise 
# the indexes from the previous files will still be in place.
  
"""
#print(master_df.head())
print("\n\n")
print(master_df.tail())
print("\n\n")
print(master_df.info())
"""

master_df = master_df.reset_index(drop=True)


# Create a data visual to show the distribution of the labels.*****************

#print(master_df['label'].value_counts())

#val_counts = master_df['label'].value_counts()
#print(val_counts[0])


"""
plt.title("Test")
plt.xlabel("Sentiment Lables")
pd.value_counts(master_df['label']).plot.bar(color=['b', 'g']
                                             , xlabel='Sentiment Label'
                                             , ylabel='Count Per Label')
"""




###############################################################################
# Look at words that can carry both a negative and postive connotation.  
# For example: bizarre, good, bad, great, , cheap, smell, aroma,
# Look at some words that have both 
###############################################################################

word_list = ['bizarre','good','bad','great','firm','cheap','smell','aroma'
             ,'thrifty']
# print(master_df[(master_df['text'].str.contains('bad') >= 1) 
#                & (master_df['label'] == 1)]).head(3)

#look_at_connotation(master_df, word_list)

###############################################################################


###############################################################################
# Clean the data. 
#
# 1) Remove punctuation.
# 2) Change to lowercase.
# 3) Create first tokenization of the data.

###############################################################################


# Remove the punctuaion for the text reviews.##################################
# Remove the punctuation from the review text and replace the current 
# string in the "text" column with it. 

#print("BEFORE Punctuation being removed: {}".format(master_df['text'].loc[0]))

# Apply the function across all rows of the 'text' column.
master_df['text'] = master_df['text'].apply(remove_punctuation)


#print("AFTER Punctuation removed: {}".format(master_df['text'].loc[0]))

###############################################################################

###############################################################################
# Change the case to lowercase. ###############################################

#print("\nBEFORE being changed to lowercase: {} \n".format(master_df['text'].loc[0]))

# Apply the lowercase function across all rows of the 'text' column.
master_df['text'] = master_df['text'].astype(str).str.lower() 

#print("AFTER being changed to lowercase: {}\n".format(master_df['text'].loc[0]))

###############################################################################



###############################################################################
# Begin to tokenize the words in the text column of the dataframe. ############


reg_exp = RegexpTokenizer('\w+')

#print("BEFORE: {}".format(master_df['text'].loc[0]))

master_df['tokenized_text'] = master_df['text'].apply(reg_exp.tokenize)

#print("AFTER : {}".format(master_df['token_text'].loc[0]))

###############################################################################


###############################################################################
# Remove the common stopwords from the dataframe. #############################

# This will retrieve the common stopwords.
stop_words = nltk.corpus.stopwords.words("english")


#print("BEFORE revoing the stopwords: {}".format(master_df['text'].loc[0]))

master_df['tokenized_text'] = master_df['tokenized_text'].apply(
    lambda sw : [word for word in sw if word not in stop_words])

#print("AFTER removing the stopwords: {}".format(master_df['token_text'].loc[0]))

###############################################################################

###############################################################################
# Remove the infrequent words. ################################################


print("REVIEW WORD STRING: {}".format(master_df['tokenized_text'].loc[0]))

master_df['review_word_str'] =master_df['tokenized_text'].apply(lambda rws : 
            ' '.join([wrd for wrd in rws if len(wrd) > 2]))

print("REVIEW WORD STRING: {}".format(master_df['review_word_str'].loc[0]))



# Create a list with all the words.  This is to be used to show the 
# frequency of the words that are in the data. ################################

all_words_str = ' '. join([token for token in master_df['review_word_str']])


tokenized_words = nltk.tokenize.word_tokenize(all_words_str)


freq_dist = FreqDist(tokenized_words)


minimum = 1
master_df['freq_dist_str'] = master_df['tokenized_text'].apply(
    lambda fd: ' '.join([word for word in fd if freq_dist[word] >= minimum]))

#print("REVIEW WORD STRING (After): {}".format(master_df['review_word_str'].loc[0]))

#print("REVIEW WORD STRING [freq_dist_str] (After): {}".format(master_df['freq_dist_str'].loc[0]))


###############################################################################
#
# Lemmatize the words to get similar word roots.
# Using the WordNetLemmatizer to get to the "root" of the words that are 
# in the review.
# 
###############################################################################

# Create a WordNetLemmitizer object.
wnl_lemmatizer = WordNetLemmatizer()

print("BEFORE being lemmatized: {}".format(master_df['text'].loc[0]))

master_df['lemmatized_text'] = master_df['freq_dist_str'].apply(wnl_lemmatizer.lemmatize)

print("AFTER being lemmatized : {}".format(master_df['lemmatized_text'].loc[0]))


# Call the find common words function.
freq_dist = find_common_words(10,'lemmatized_text')

print(freq_dist)


# Create a graph of the common words. #########################################
###############################################################################
"""
plt.figure(figsize=(20, 20))

sns.set_style("whitegrid")
ax = sns.barplot(x=freq_dist.index, y=freq_dist.values
            , color='green').set(title="Ten Most Common Words in the Reviews"
                                 , xlabel='Word', ylabel='Count of the Word')
sns.set(font_scale=10)
                    
plt.show()
"""
###############################################################################

###############################################################################
# Create a Wordcloud to show the frequncy of the words graphically. 
###############################################################################
"""
all_words = " ". join([word for word in master_df['lemmatized_text']])

#plt.clf()
create_wordclud(plot_title="The Word Cloud", plot_data=all_words)

"""
###############################################################################



###############################################################################
# Split the data into train and test sets. 
###############################################################################

X = master_df['lemmatized_text']
y = master_df['label']

# Test split size.
test_split = 0.05           # Change to 0.20 **********************************


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split
                                                    , random_state=42)

# Print a sample of the data from the test and training sets. 
print(X_train[0:3])

print('\nX_train shape-type: {}-{}'.format(X_train.shape, type(X_train)))
print('X_test shape: {}'.format(X_test.shape))
print('y_train shape-type: {} {}'.format(y_train.shape, type(y_train)))
print('y_test shape: {}'.format(y_test.shape))


###############################################################################

##############################################################################
# Create the first Sequential model. 

# All the code below will use the following citations:
# (tf.keras.preprocessing.text.Tokenizer, n.d.)
# (What Does Keras Tokenizer Method Exactly Do?, n.d.)
###############################################################################

number_of_token_words = 4000

max_length = 64

keras_tokenizer = Tokenizer(num_words=number_of_token_words)


# Fit the data using the training dataset. ************************************
# Used to update to the internal vocabulary.  See citation given at the 
# beginning of this section. **************************************************
keras_tokenizer.fit_on_texts(X_train)


# Use texts_to_sequences to transform each sequence into a list of numerics. 
# See citation above.**********************************************************

X_train = keras_tokenizer.texts_to_sequences(X_train)
X_test = keras_tokenizer.texts_to_sequences(X_test)

vocabulary_size = len(keras_tokenizer.word_index) +1 



##############################################################################
# Add Padding to the numeric rerpresentations.
###############################################################################

X_train = ps(X_train, padding='post', maxlen=max_length)
X_test = ps(X_test, padding='post',maxlen=max_length)


# print and example.
print(X_test[0])

###############################################################################


###############################################################################
# Define the model.
###############################################################################

# Create the compiled model.**************************************************

output_dimension = 4000
seq_model = get_sequential_model(vocabulary_size, output_dimension
                                 , max_length, drop=0.3)


print(seq_model.summary())

##############################################################################


# Compile the model. ##########################################################

seq_model.compile(optimizer='adam', loss='binary_crossentropy'
                  , metrics=['accuracy'])


###############################################################################

###############################################################################
# Fit the model. ##############################################################
#
# This section uses the concept of Early stopping to make usre the model does 
# not overfit the data.  
# The code from this section uses the following citation. (Brownlee, 2020)
#
###############################################################################


batch_sze = 48
number_of_epochs = 25

early_stop = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1)
"""
model_history = seq_model.fit( X_train, y_train
                             , batch_size=batch_sze, epochs=number_of_epochs
                             , verbose=1, validation_data=(X_test, y_test))
"""

model_history = seq_model.fit( X_train, y_train
                             , batch_size=batch_sze, epochs=number_of_epochs
                             , verbose=1, validation_data=(X_test, y_test)
                             , callbacks=[early_stop])




###############################################################################

###############################################################################
# Plot the summary for the training history.******************************
# Accuracy. *******************************************

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy','Validation Accuracy'])
plt.show()

# Plot the loss for the model. ***********************************************
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss','Validation Loss'])
plt.show()

###############################################################################

###############################################################################
# Look at the scroe from evaluate. 
##############################################################################

score= seq_model.evaluate(X_test, y_test, verbose=1)

print("\n\nTest loss: ", score[0])
print("Test accuracy: ", score[1])






###############################################################################

###############################################################################
# Add save model code using SaveModel. 
###############################################################################

# Retrieve the current time. **************************************************
#current_time = datetime.now()

#date_time = current_time.strftime("_%y%m%d_%H%M")

# save the model with the time and date.***************************************
#seq_model.save('Models/' + date_time)

###############################################################################



#print(vocabulary_size)
#print(X_test)
#print(type(X_test))
#print(type(X_train))

print(master_df.info())





























