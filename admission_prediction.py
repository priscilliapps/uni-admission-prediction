# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:23:31 2020

@author: prisc
"""

#INPUTS
#FEATURES: GRE score (max 340), TOEFL iBt (max 120), Uni Rating (max 5)
#Statement of Purpose, LoR (max 5), Ugrad GPA (max 10), Research exp (0 or 1)

#OUTPUTS: chance of admission (0 to 1)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from jupyterthemes import jtplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

admission_df = pd.read_csv('Admission_Predict.csv')
print (admission_df.head())
admission_df.drop ('Serial No.', axis = 1, inplace = True)
print (admission_df)

#EDA IMPLEMENTATION

#checking null values
print (admission_df.isnull().sum())

#dataframe information
print (admission_df.info())

#statistical summary
print (admission_df.describe())

#grouping data by uni rating 
#and calculate the mean of each feature
df_uni = admission_df.groupby('University Rating').mean()
print (df_uni)

#PERFORM DATA VIS

#histogram
#admission_df.hist(bins = 30, figsize = (20, 20), color = 'g')
#sns.pairplot(admission_df)
corr_matrix = admission_df.corr()
plt.figure(figsize = (6, 6))
sns.heatmap(corr_matrix, annot = True)
plt.show()

#CREATE THE TRAINING & TESTING DATASET
print(admission_df.columns)
x = admission_df.drop(columns = ['Chance of Admit']) #inputs
y = admission_df['Chance of Admit']#outputs chance of admit
print (x.shape)
print (y.shape)

#convert data to numpy array
x = np.array(x)
y = np.array(y)
y = y.reshape(-1, 1)
print (y.shape)

#scaling the data to make it equal 
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_x.fit_transform(y)

#splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15) #0.15 
#means allocating 15% of the data to be testing data

#TRAIN AND EVALUATE LINEAR REGRESSION MODEL

LinReg_model = LinearRegression()
LinReg_model.fit(x_train, y_train)
accuracy_LinReg = LinReg_model.score(x_train, y_train)
print (accuracy_LinReg)

# USING ARTIFICIAL NEURAL NETWORK

ANN_model = keras.Sequential()
ANN_model.add(Dense(50, input_dim = 7))
ANN_model.add(Activation('relu'))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(50))
ANN_model.add(Activation('linear'))
ANN_model.add(Dense(1))

ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()

ANN_model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
epochs_hist = ANN_model.fit(x_train, y_train, epochs = 100, batch_size = 20)
result = ANN_model.evaluate(x_test, y_test)
accuracy_ANN = 1 - result
print ("Accuracy: {}".format(accuracy_ANN))
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])

#USING DECISION TREE & RANDOM FOREST
DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(x_train, y_train)
accuracy_DecisionTree = DecisionTree_model.score(x_test, y_test)
print (accuracy_DecisionTree)

RandomForest_model = RandomForestRegressor(n_estimators = 100, max_depth = 100)
RandomForest_model.fit(x_train, y_train)
accuracy_RandomForest = RandomForest_model.score(x_test, y_test)
print (accuracy_RandomForest)

y_predict = LinReg_model.predict(x_test)
plt.plot(y_test, y_predict, '^', color = 'g')

y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)
plt.plot(y_test_orig, y_predict_orig, '^', color = 'r')

k = x_test.shape[1]
n = len(x_test)
print (n)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)), '))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print ('RMSE = ', RMSE, '\nMSE = ', MSE, '\nMAE = ', MAE, '\nR2 = ', r2, '\nAdjust R2 = ', adj_r2) 