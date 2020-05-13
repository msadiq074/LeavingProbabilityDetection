# -*- coding: utf-8 -*-
"""
Created on Fri May  8 01:45:13 2020

@author: msadi
"""

#-------------------------------

# Part 1 : Data preprocessing

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,-1].values

# taking care of missing data
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_2 = LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X))
X = X[:,1:]


# Splitting data into training and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#-----------------------------

# Part 2 : Building an ANN

# import libraries again
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initializing ANN
classifier = Sequential()

# adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
#classifier.add(Dropout(p=0.1))

# adding second hidden layer with dropout
#classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
#classifier.add(Dropout(p=0.1))

# adding output layer
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

# compile ANN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN to training set
classifier.fit(X_train, Y_train, batch_size = 25, epochs = 500)

#-------------------------------

# Part 3 : Predictions and Evaluations

test_run=np.array([0,0,600,1,40,3,60000,2,1,1,50000])
test_run=np.array([test_run])
test_run
# predicting the test set results
Y_test
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred
# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print( (cm[0][0] + cm[1][1])/2000)

#-----------------------------

# part 4 evaluating, improving and tuning ann

# evaluating the ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    # initializing ANN
    classifier = Sequential()
    # adding the input layer and the first hidden layer
    classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
    # adding second hidden layer
    classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
    # adding output layer
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    # compile ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier,batch_size=25,epochs=500)

accuracies = cross_val_score(estimator = classifier, X=X_train, y=Y_train, cv=10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving ANN
# dropout regularization to reduce overfitting if needed


# Parameter Tuning
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    # initializing ANN
    classifier = Sequential()
    # adding the input layer and the first hidden layer
    classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
    # adding second hidden layer
    classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
    # adding output layer
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    # compile ANN
    classifier.compile(optimizer = optimizer,loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = { 'batch_size':[25, 32], 'epochs':[100, 500], 'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid=parameters, scoring='accuracy', cv = 10)
grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_parameters
best_accuracy
