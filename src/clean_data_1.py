from __future__ import division
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import graphlab as gl

__author__ = 'Vladimir Iglovikov'

'''
In this approach I will use only labeled train data and Naive bag of words
implementation.
'''

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", ' ', review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
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
train.save('../data/train_cleaned1.csv', format='csv')

print 'saving test'
test.save('../data/test_cleaned1.csv', format='csv')
