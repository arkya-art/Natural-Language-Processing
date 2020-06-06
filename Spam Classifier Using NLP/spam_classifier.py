# -*- coding: utf-8 -*-
import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep = '\t',names = ['label','message'])
#here in the given file the label is separated using tabs key and not space so separator=\t given for separating into 2 columns


#datacleaning and preprocessing
import re
import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

clean_msgs = []

for i in range(0,len(messages)):
    
    # we are selecting the messages in the dataframe and applying various operations in it
    review = re.sub('[^a-zA-Z]'," ", messages['message'][i])
    review = review.lower()
    
    # we are splitting the messages into word list on the basis of spaces
    review = review.split()
    
    # if word is not stopword then we stem it and excludes all the stopwords 
    review = [stemmer.stem(word)  for word in review if word not in set(stopwords.words('english')) ]
    
    review = " ".join(review)
    
    clean_msgs.append(review)


#creating a bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
# when no val. passed to CountVectorizer() it gives 6292 col. so we wanted only frequent words that may be present so passing random max_features and selecting top 5000 features appered

# we are taking our features in X
X = cv.fit_transform(clean_msgs).toarray()
    
# we are taking our response in Y
Y = pd.get_dummies(messages['label'])
Y = Y.iloc[:,1].values
#get_dummies method creates numeric representation of binary classification for ham:10 and spam:01
# but we try to reduce col. so reduce the prev representation as ham:1 and spam:0

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
building the machine learning model

""" 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)

#training model using Naive-Bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train,y_train)

y_pred = spam_detect_model.predict(x_test)

#checking the accuracy of the model and printing it's confusion matrix
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy value:",accuracy*100,'%')





