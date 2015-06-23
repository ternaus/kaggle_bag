from __future__ import division
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import graphlab as gl
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


__author__ = 'Vladimir Iglovikov'

'''
In this approach I will use only labeled train data and Naive bag of words
implementation.
'''
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer("english")
wordnet_lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", ' ', review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if not w in stops] #clean
    # meaningful_words = [porter_stemmer.stem(w) for w in words if not w in stops] #porter_stemmer
    # meaningful_words = [lancaster_stemmer.stem(w) for w in words if not w in stops] #lancaster_stemmer_stemmer
    # meaningful_words = [snowball_stemmer.stem(w) for w in words if not w in stops] #snowball stemmer_stemmer_stemmer
    meaningful_words = [wordnet_lemmatizer.lemmatize(w) for w in words if not w in stops] #worldnet lemmatizer


    return ' '.join(meaningful_words)


train = gl.SFrame.read_csv('../data/labeledTrainData.tsv', header=True, delimiter='\t', quote_char='"')
test = gl.SFrame.read_csv('../data/testData.tsv', header=True, delimiter='\t', quote_char='"')

print 'clean train'
train['review_cleaned'] = train['review'].apply(review_to_words)

print 'clean test'
test['review_cleaned'] = test['review'].apply(review_to_words)

del train['review']
del test['review']

print 'saving train'
train.save('../data/train_cleaned2.csv', format='csv')

print 'saving test'
test.save('../data/test_cleaned2.csv', format='csv')
