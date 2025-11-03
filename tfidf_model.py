# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 22:06:51 2025

@author: DHIVANUJAN
"""

import nltk

nltk.download('wordnet')

paragraph = """Steve Jobs, in his inspiring message to young people, shares three lessons from his life. First, he explains that we can only connect the dots by looking backward, so we should trust our curiosity and instincts even when the path is unclear. Second, he talks about facing setbacks, including being fired from Apple, and emphasizes that loving what you do will help you persevere—if you haven’t found your passion yet, keep searching and never settle. Finally, he reflects on the shortness of life, urging us not to waste time living someone else’s dream or being held back by fear. His advice is simple but powerful: follow your heart, be true to yourself, and always “stay hungry, stay foolish.”"""

#cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# creating the TF-IDF model  
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()