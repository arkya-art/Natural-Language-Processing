# -*- coding: utf-8 -*-
"""
-both in TF-IDF and BOW semantic info. is not stored.TF-IDF gives importance to uncommon words.

- there exist chance of overfitting


to avoid these instead of using 0,1 for particular word we can use word2vec

----------------------------------------------------------------------------------------------
                                              word2vec

definition

- in this model ,each word is represented as a vector of 32 or more dimension instead of single 
  no. in form of array
  
   word1 = [dim1,dim2,......dim32..]
   word2 = [dim1,dim2,......dim32..]

- In these semantic info and relation b/w diff. words is also preserved
  (words with similar meaning are places close to each other while diff. meaning are apart)
  
steps

- Tokenization of sentences
- Create histograms
- Take most fequent words
- Create a matrix with all unique words,it also represent the occurence relation b/w words
 
"""
import nltk


from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""
               
#preprocessing the data and only removing the stop words not stemming/lemmatization
sentences = nltk.sent_tokenize(paragraph)

cleaned_sentences = []

for i in range(len(sentences)):
    
    review = re.sub('[^a-zA-Z]'," ",sentences[i])   #replaces all the characters except a-z and A-Z letters with spaces
    review = review.lower()                         # lowering uppercase characters
    review = review.split()                         # splitting on the basis of spaces creates list of words
    
    review = [word for word in review if word not in set(stopwords.words('english'))]
    
    cleaned_sentences.append(review)  

          
#training the word2vec model
model = Word2Vec(cleaned_sentences, min_count = 1) 

#if the word is present at-least 2 times then use this, which converts each word in sentence to vector 

#for finding the vocabulary in these word2vec model we use wv.vocab
words = model.wv.vocab
#the above line createswords with 100 dim. of vector---eg-> word1 = [dim1,dim2,...dim100]


#finding word vectors
vector = model.wv['war']

#most similar words
similar = model.wv.most_similar('war')





              