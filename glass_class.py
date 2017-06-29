import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('glass.csv')
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

from mlxtend.preprocessing import one_hot
Y = one_hot(Y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build():
    seq = Sequential()
    seq.add(Dense(units = 150, activation = 'relu', kernel_initializer = 'uniform', input_dim = 9))
    seq.add(Dropout(rate = 0.2))
    seq.add(Dense(units = 150, activation = 'relu', kernel_initializer = 'uniform'))
    seq.add(Dropout(rate = 0.2))
    seq.add(Dense(units = 8, activation = 'softmax', kernel_initializer = 'uniform'))
    seq.compile('adam', 'categorical_crossentropy', metrics = ['accuracy'])
    return seq

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

classifier = KerasClassifier(build, batch_size = 32, nb_epoch = 300)
csv = cross_val_score(classifier, X_train, Y_train, cv = 10, n_jobs = -1)

mean = csv.mean()
std = csv.std()

classifier.fit(X_train, Y_train, batch_size = 32 , epochs = 300)
print(classifier.score(X_test,Y_test)*100)

classifier.predict(X_test)
