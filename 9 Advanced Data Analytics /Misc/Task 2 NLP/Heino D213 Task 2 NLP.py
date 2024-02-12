# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:53:53 2024

@author: Matthew Heino
"""

import matplotlib.pyplot as plt
#import tensorflow as tf
import nltk
import pandas as pd
import re 
import seaborn as sns
import warnings

from datetime import datetime
from keras import models, layers
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences as ps
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)

#nltk.download('stopwords')

#nltk.download('punkt')

#nltk.download('wordnet')

# Functions start here. #######################################################

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

def remove_extra_space(rev_text: str) -> str:
    """ Method to remove extra spaces from the text reviews. 
   
   Parameters:
   ----------
       rev_text (str):     String with the text review.
      
    
   Returns:
      None
       
   """
    
   # print("IN extra space:")
    
    space_removed = re.sub(' +', ' ', rev_text)
    
    return space_removed
    
    
    
###############################################################################    

def remove_numbers(rev_text: str) -> str:
                   
   # print("in remove numbers")
    
    number_removed = "".join(num for num in rev_text if not num.isdigit())
    
    return number_removed
    


##############################################################################

def remove_punctuation(rev_text: str) -> str:
    
    #print("In Removed:", rev_text)
    
    
    punct_removed = "".join(rem for rem in rev_text if rem not in('?','.',';'
                        ,':', '!', '"', ',','/','#','%','(',')','*','+','-'
                        , '<', '>', '@', '[',']', "\\",'~', '`', '_','{'
                        , '}', "|", '-' ))
    
    #print("Punctuation removed: ", punct_removed)
    return punct_removed

###############################################################################
# Funbctions End Here. ########################################################


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

"""
print(master_df.head())
print("\n\n")
print(master_df.tail())
print("\n\n")
print(master_df.info())
print(master_df.iloc[2477])



#print(master_df.info())

# Print a sampling of the data from the newly created dataframe. 
# Set the random_state to make reproducible.

#print(master_df.sample(10, random_state=2477))

#print(master_df['label'].value_counts())

#val_counts = master_df['label'].value_counts()




# Create a data visual to show the distribution of the labels.*****************

#print(val_counts[0])
"""


"""
plt.title("Test")
plt.xlabel("Sentiment Lables")
pd.value_counts(master_df['label']).plot.bar(color=['b', 'g']
                                             , xlabel='Sentiment Label'
                                             , ylabel='Count Per Label')
"""

# Look at words that can carry both a negative and postive connotation.  
# For example: bizarre, good, bad, great, , cheap, smell, aroma,
# Look at some words that have both 


word_list = ['bizarre','good','bad','great','firm','cheap','smell','aroma','thrifty']
# print(master_df[(master_df['text'].str.contains('bad') >= 1) 
#                & (master_df['label'] == 1)]).head(3)

#look_at_connotation(master_df, word_list)



# Clean the data. *************************************************************
###############################################################################


# Remove the punctuation from the review text and replace the current 
# string in the "text" column with it. 

#print("BEFORE: {}".format(master_df['text'].loc[0]))
# Apply the function across all rows of the 'text' column.
master_df['text'] = master_df['text'].apply(remove_punctuation)
#print("AFTER: {}".format(master_df['text'].loc[0]))


# Remove the numbers from the text column. **************************************

#print("BEFORE: {}".format(master_df['text'].loc[57]))
master_df['text'] = master_df['text'].apply(remove_numbers)
#print("AFTER: {}".format(master_df['text'].loc[57]))




# Remove extra space in the text columns.**************************************

#print("BEFORE: {}".format(master_df['text'].loc[30]))
master_df['text'] = master_df['text'].apply(remove_extra_space)
#print("AFTER : {}".format(master_df['text'].loc[30]))


# Change the case to lowercase. **********************************************

#print("BEFORE: {}".format(master_df['text'].loc[57]))
master_df['text'] = master_df['text'].astype(str).str.lower() 
#print("AFTER : {}".format(master_df['text'].loc[57]))


#print(master_df['text'].sample(20, random_state=247))



# Begin to tokenize the words in the text column of the dataframe.************
reg_exp = RegexpTokenizer('\w+')

#print("BEFORE: {}".format(master_df['text'].loc[0]))

master_df['token_text'] = master_df['text'].apply(reg_exp.tokenize)

#print("AFTER : {}".format(master_df['token_text'].loc[0]))

#print(master_df.info())

# Remove the stopwords. *******************************************************
stop_words = nltk.corpus.stopwords.words("english")

#print(stop_words[0:30])

#print("BEFORE: {}".format(master_df['text'].loc[136]))

master_df['token_text'] = master_df['token_text'].apply(
    lambda sw : [word for word in sw if word not in stop_words])

#print("AFTER : {}".format(master_df['token_text'].loc[136]))





# Remove the infrequent words. ******************************

#print(master_df.info())

#print("REVIEW WORD STRING: {}".format(master_df['token_text'].loc[57]))

master_df['review_word_str'] =master_df['token_text'].apply(lambda rws : 
            ' '.join([wrd for wrd in rws if len(wrd) > 2]))

#print("REVIEW WORD STRING: {}".format(master_df['review_word_str'].loc[57]))



# Create a list with all the words.  This is to be used to show the 
# frequency of the words that are in the data. 

all_words_str = ' '. join([token for token in master_df['review_word_str']])

#print(len(all_words_str))

tokenized_words = nltk.tokenize.word_tokenize(all_words_str)

#print(len(tokenized_words))

freq_dist = FreqDist(tokenized_words)

#print(freq_dist) 

minimum = 1

master_df['freq_dist_str'] = master_df['token_text'].apply(
    lambda fd: ' '.join([word for word in fd if freq_dist[word] >= minimum])  )

#print("REVIEW WORD STRING (After): {}".format(master_df['review_word_str'].loc[0]))

#print("REVIEW WORD STRING [freq_dist_str] (After): {}".format(master_df['freq_dist_str'].loc[0]))




###############################################################################
# Lemmatize the words to get similar word roots. ******************************
# Using the WordNetLemmatizer to get to the "root" of the words that are 
# in the review. 
###############################################################################

# Create a WordNetLemmitizer object.
wnl_lemmatizer = WordNetLemmatizer()

#print("BEFORE: {}".format(master_df['text'].loc[0]))

master_df['text_lemma'] = master_df['freq_dist_str'].apply(wnl_lemmatizer.lemmatize)

#print("AFTER : {}".format(master_df['text_lemma'].loc[0]))


# Find the most common words for the  dataset. 
# This is to see what are the common words that appear in the reviews.
# Find the top ten words in  the reviews. 

most_common = 10

all_words2 = ' '. join([word for word in master_df['text_lemma']])

new_token_words = nltk.word_tokenize(all_words2)

freq_dist2 = FreqDist(new_token_words)

top_words = freq_dist2.most_common(most_common)

dist_freq = pd.Series(dict(top_words))


#print(freq_dist2.most_common(most_common))

# Create a nice plot of the words.

"""
plt.figure(figsize=(30, 20))

sns.set_style("whitegrid")
ax = sns.barplot(x=dist_freq.index, y=dist_freq.values
            , color='green').set(title="Ten Most Common Words in the Reviews"
                                 , xlabel='Word', ylabel='Count of the Word')
sns.set(font_scale=5)
"""                               


# Display a Word cloud forthe words that are in the reviews.###################
###############################################################################

"""
word_cloud = WordCloud(width=1000, height=1000, max_font_size = 150
                       , background_color='grey'
                       , random_state=247).generate(all_words2)

plt.figure(figsize=(10,10))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')

"""
# Export the data to the CSV file. *******************************************
#master_df.to_csv("Heino D213 Task 2 Cleaned.csv", index=True, header=True)





###############################################################################
# Create the train and test data for the model. 
###############################################################################

# Extract the lemmatized text reviews for the model.***************************
X = master_df['text_lemma'] 

y = master_df['label'] 

test_split = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split
                                                    , random_state=42)


# Print a sample of the data from the test and training sets. 
print(X_train[0:3])

print('\nX_train shape-type: {}-{}'.format(X_train.shape, type(X_train)))
print('X_test shape: {}'.format(X_test.shape))
print('y_train shape-type: {} {}'.format(y_train.shape, type(y_train)))
print('y_test shape: {}'.format(y_test.shape))



#print("\n\n FROM THE END: \n",master_df.info())


###############################################################################
# Create the first model.  This will be the Sequential keras model.
# Will make use of the Tokenizer function from the keras library.
#
# All the code below will use the following citation:
# (tf.keras.preprocessing.text.Tokenizer, n.d.)
#
###############################################################################

# The number of words to keep from the reviews. 
num_words_token = 4425


# Create a Tokenizer to convert the text to numerics. 
keras_tokenizer = Tokenizer(num_words=num_words_token)

# Fit the data using the training dataset. ************************************
# Used to update to the internal vocabulary.  See citation given at the 
# beginning of this section. **************************************************

keras_tokenizer.fit_on_texts(X_train)

# Use texts_to_sequences to transform each sequence into a list of texts. 
# See citation above.**********************************************************

X_train = keras_tokenizer.texts_to_sequences(X_train)
X_test = keras_tokenizer.texts_to_sequences(X_test)





#print(X_train[0:3])
#print(len(keras_tokenizer.word_index))

vocabulary_size = len(keras_tokenizer.word_index) + 1

max_length = 64 # 64

# Pad the values to fit.
X_train = ps(X_train, padding='post', maxlen=max_length)
X_test = ps(X_test, padding='post', maxlen=max_length)


#print('Vocabulary size: {}'.format(vocabulary_size))
#print('Maximum length: {}'.format(max_length))


# Print a sample padded of the padding sequence. ******************************
#print(X_test[3])


# Define the model layers along with the dropout value. **********************

drop_out = 0.

output_dimension = 2000 # 2000

seq_model = models.Sequential()

# ADd to the model. ***********************************************************
seq_model.add(layers.Embedding(input_dim=vocabulary_size
                               , output_dim=output_dimension
                               , input_length=max_length ))
              
if(drop_out > 0):
    seq_model.add(layers.Dropout(drop_out))
    
# Add the Flatten layer. ******************************************************
seq_model.add(layers.Flatten())


# Add a Dense layer. **********************************************************
seq_model.add(layers.Dense(1, activation='sigmoid'))

# Show the model summary using summary(). ************************************
#print(seq_model.summary())




# Compile the model with the given parameters and layers. ********************
seq_model.compile(optimizer='adam', loss='binary_crossentropy'
                  , metrics=['accuracy'])

"""

# Add save model code using SaveModel. ***************************************

# Retrieve the current time. **************************************************
#current_time = datetime.now()

#date_time = current_time.strftime("_%y%m%d_%H%M")

# save the model with the time and date.***************************************
#seq_model.save('Models/' + date_time)


"""

# 
# Split the data again.

"""
value_split = .2

val_split = int(value_split * len(X_train))

x_val = X_train[ : val_split]
X_train_partial = X_train[val_split : ]

y_val = y_train[ : val_split]
y_train_partial = y_train[val_split : ]
"""


# Fit the model wtih the partial sets. ****************************************

"""
model_history = seq_model.fit( X_train_partial, y_train_partial
                             , batch_size=batch_sze, epochs=number_of_epochs
                             , verbose=1, validation_data=(x_val, y_val))


"""
batch_sze = 32
number_of_epochs = 50

"""
model_history = seq_model.fit( X_train, y_train
                             , batch_size=batch_sze, epochs=number_of_epochs
                             , verbose=1, validation_data=(X_test, y_test))
"""
model_history = seq_model.fit( X_train, y_train
                             , batch_size=batch_sze, epochs=number_of_epochs
                             , verbose=1, validation_split=0.2)


#score = seq_model.evaluate(X_test, y_test, verbose=1)  

                           

print(model_history.history.keys())

# Plot the summary for the training history.******************************
# Accuracy. *******************************************

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.show()

# Plot the loss for the model. ***********************************************
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Test'])
plt.show()








"""
#print(master_df[master_df['label'].isna()])

# Find the duplicate rows.
duplicateRows = master_df['label'].duplicated()

print(duplicateRows)
print(master_df.head(30))


print("Test loss: ",score[0])
print("\n Test Accuracy: ", score[1])
"""


