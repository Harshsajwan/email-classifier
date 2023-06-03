import numpy as np
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string

df = pd.read_csv('emails.csv')
print(df.head(5))

#Print the shape (Get the number of rows and cols)
print(df.shape)

#Get the column names
print(df.columns)

#Checking for duplicates and removing them
df.drop_duplicates(inplace = True)

#Show the new shape (number of rows & columns)
print(df.shape)

#Show the number of missing (NAN, NaN, na) data for each column
print(df.isnull().sum())

#Need to download stopwords
print(nltk.download('stopwords'))


#Tokenization (a list of tokens), will be used as the analyzer
#1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
#2.Stop words in natural language processing, are useless words (data).
def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words


#Show the Tokenization (a list of tokens )
print(df['text'].head().apply(process_text))

print(100)
