from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from nolearn import lasagne
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid
from nolearn.lasagne import NeuralNet
import numpy as np
import theano
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
import os
import time

__author__ = 'Vladimir Iglovikov'

import pandas as pd
'''
Script is used to estimate model preformance using cross validatio and to
create submission
'''

def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


ind = False


train = pd.read_csv(os.path.join('..', 'data', 'train_cleaned1.csv'))

vectorizer = CountVectorizer(analyzer='word',
                            tokenizer = None,
                            preprocessor=None,
                            stop_words=None,
                            max_features=50)

train_data_features = vectorizer.fit_transform(train['review_cleaned'].values)

print 'scaling'
scaler = preprocessing.StandardScaler()
encoder = LabelEncoder()

X_train = scaler.fit_transform(train_data_features.toarray()).astype(np.float32)
y = train['sentiment']

y = encoder.fit_transform(y).astype(np.int32)
# pca = decomposition.PCA(n_components=n_components)

# X_new = pca.fit_transform(X).astype(np.float32)



layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout2', DropoutLayer),
           # ('dense2', DenseLayer),
           ('output', DenseLayer),
           ]
num_units = 100

num_classes = 2
num_features = X_train.shape[1]

clf = NeuralNet(layers=layers0,

                 input_shape=(None, num_features),
                 dense0_num_units=num_units,
                 dropout1_p=0.5,
                 dense1_num_units=num_units,
                 dropout2_p=0.5,
                 # dense2_num_units=num_units,
                 output_num_units=num_classes,
                # output_num_units=1,
                 output_nonlinearity=softmax,
                # output_nonlinearity=sigmoid,

                 update=nesterov_momentum,
                 # update_learning_rate=0.001,
                 # update_momentum=0.9,
                 update_momentum=theano.shared(float32(0.9)),
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=1000,
                 update_learning_rate=theano.shared(float32(0.03)),
                 # objective_loss_function= binary_crossentropy,
                 on_epoch_finished=[
                    AdaptiveVariable('update_learning_rate', start=0.001, stop=0.00001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    EarlyStopping(patience=100),
                ])


if not ind:
  skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=42)

  scores = cross_validation.cross_val_score(clf, X_train, y, cv=skf, scoring='roc_auc')
  print scores
  print np.mean(scores), np.std(scores)

if ind:
  test = pd.read_csv(os.path.join('..', 'data', 'test_cleaned1.csv'))
  test_data_features = vectorizer.fit_transform(train['review_cleaned'].values)
  X_test = scaler.transform(test_data_features.toarray()).astype(np.float32)
  clf.fit(X_test)
  submission = pd.DataFrame()
  submission['sentiment'] = clf.predict_proba()[:, 1]
  submission['id'] = test['id']
  submission.to_csv('predictions/{timestamp}.csv'.format(timestamp=time.time()), index=False)
