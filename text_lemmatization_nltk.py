# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 21:06:41 2025

@author: DHIVANUJAN
"""

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

paragraph = """Steve Jobs, in his inspiring message to young people, shares three lessons from his life. First, he explains that we can only connect the dots by looking backward, so we should trust our curiosity and instincts even when the path is unclear. Second, he talks about facing setbacks, including being fired from Apple, and emphasizes that loving what you do will help you persevere—if you haven’t found your passion yet, keep searching and never settle. Finally, he reflects on the shortness of life, urging us not to waste time living someone else’s dream or being held back by fear. His advice is simple but powerful: follow your heart, be true to yourself, and always “stay hungry, stay foolish.”"""

# Sentence tokenization
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)