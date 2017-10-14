# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 07:23:16 2017

@author: satya
"""
#-------------------------------------------------------------------
#Part 1: Data preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as matplotlib

from keras.utils import np_utils

#Importing Dataset
dataset1=pd.read_csv('crime_data.csv')
Xy=dataset1.iloc[:,4:7].values

X1=Xy[Xy[:,2]==1][0:40000]
X2=Xy[Xy[:,2]==2][0:40000]
X3=Xy[Xy[:,2]==3][0:40000]
X4=Xy[Xy[:,2]==4][0:75]
X5=Xy[Xy[:,2]==5][0:40000]
X6=Xy[Xy[:,2]==6][0:40000]
X7=Xy[Xy[:,2]==7][0:40000]
X8=Xy[Xy[:,2]==8][0:40000]
X9=Xy[Xy[:,2]==9][0:40000]
X10=Xy[Xy[:,2]==10][0:40000]

dataset=np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10),axis=0)

X=dataset1.iloc[0:len(dataset),4:6].values

for i in range (len(dataset)):
    for j in range (2):
        X[i][j]=dataset[i][j]

y=dataset1.iloc[0:len(dataset),6].values

for i in range (len(dataset)):
        y[i]=dataset[i][2]

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,0]=labelencoder_X_1.fit_transform(X[:,0])

labelencoder_X_2=LabelEncoder()
X[:,1]=labelencoder_X_2.fit_transform(X[:,1])

onehotencoder = OneHotEncoder(categorical_features=[0,1])
X=onehotencoder.fit_transform(X).toarray()

X=X[:, 1:]

y = np_utils.to_categorical(y, 11)
y=y[:,1:]




#Splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


####Part 2: Make the ANN

#Importing the Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=11, init='uniform', activation='relu', input_dim=30))

# Adding the second hidden layer
classifier.add(Dense(output_dim=11, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=10, init='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=2)

#Part 3: Making the predictions and evaluating the model

#Predicting the test set results
y_pred=classifier.predict(X_test)
#-------------------------------------------------------------------

#yy_pred=np.zeros(len(y_pred))
#for i in range (len(y_pred)):
#    
#    for j in range (11):
#        max=y_pred[i][]
#    yy[i]=np.amax(y_pred[i], axis=0)


#y_pred1=(y_pred>0.5)

#Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
