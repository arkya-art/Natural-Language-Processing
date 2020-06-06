# -*- coding: utf-8 -*-

import nltk

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

#cleaning the text
import re                                            #for regularization
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

sentences = nltk.sent_tokenize(paragraph)

cleaned_sentences = []

for i in range(len(sentences)):
    
    review = re.sub('[^a-zA-Z]'," ",sentences[i])   #replaces all the characters except a-z and A-Z letters with spaces
    review = review.lower()                         # lowering uppercase characters
    review = review.split()                         # splitting on the basis of spaces creates list of words
    
    review = [lemma.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    
    
    review = " ".join(review)
    cleaned_sentences.append(review) 
    
"""
In TF_IDF method we convert word to vectors/numerics with special focus on imp. words, that implies some important 
info then we input it into machine learning model,In these particular case we have reduced each and every sentence
 of paragraph by using stopwords and lemmatization into cleaned_sentence list now we proceed as:- 

let the cleaned_list be:   sent1 - good boy
                           sent2 - good girl
                           sent3 - boy girl good                         

1) prepare histogram of each and every words that occur in  cleaned_sentence list=[sent1,sent2,...], eg->
    
    good - 3
    boy - 2
    girl -2
    
2) We compute TF(term-frequency) and IDF(inverse doc. frequency) and multiply it

  TF = (no. of repition of word in a sent.) /(no. of words in sent.)
  
                good            boy              girl
    
    sent1        1/2            1/2               0
    
    sent2        1/2            0                1/2
    
    sent3        1/3           1/3               1/3
    
 IDF = log(no.of sent/no. of sent. containg word)   
 
                good           boy              girl
    
    IDF     log(3/3)=0       log(3/2)           log(3/2)
    
    
3) we prepare the dataset as follows:-
    
              good            boy                girl              output

 sent1       1/2*0=0       1/2*log(3/2)       0*log(3/2)=0                                 
 
 sent2       1/2*0=0       0*log(3/2)=0       1/2*log(3/2)                                 

 sent3       1/3*0 =0      1/3*log(3/2)       1/3*log(3/2)                                

all the above can simply be done using sklearn module feature_extraction with class TfidVectorizer

""" 

from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer()

X = cv.fit_transform(cleaned_sentences).toarray()
    
    
    
    
    
    
    
