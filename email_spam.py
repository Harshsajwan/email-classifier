#Import libraries
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


# Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

## Converts strings to integer counts
#vectorizer = CountVectorizer(analyzer=process_text)

## Learn a vocabulary dictionary of all tokens in the raw documents.
#bow_transformer = vectorizer.fit(df['text'])    

## Transform documents to document-term matrix.
#messages_bow = bow_transformer.transform(df['text'])

#Convert string to integer counts, learn the vocabulary dictionary and return term-document matrix
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])


#Split data into 80% training & 20% testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)


#Get the shape of messages_bow
messages_bow.shape

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Print the predictions
print(classifier.predict(X_train))
#Print the actual values
print(y_train.values)


#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))



#Print the predictions
print('Predicted value: ',classifier.predict(X_test))
#Print Actual Label
print('Actual value: ',y_test.values)



#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))
print(100)
