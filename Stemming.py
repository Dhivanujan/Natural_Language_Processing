import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """Steve Jobs, in his inspiring message to young people, shares three lessons from his life. First, he explains that we can only connect the dots by looking backward, so we should trust our curiosity and instincts even when the path is unclear. Second, he talks about facing setbacks, including being fired from Apple, and emphasizes that loving what you do will help you persevere—if you haven’t found your passion yet, keep searching and never settle. Finally, he reflects on the shortness of life, urging us not to waste time living someone else’s dream or being held back by fear. His advice is simple but powerful: follow your heart, be true to yourself, and always “stay hungry, stay foolish.”"""

sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    sentences[i] = ' '.join(stemmed_words)
