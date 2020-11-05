# -*- coding: utf-8 -*-
"""stock_predictor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qux5RnihRGdwdcf-1j_IO_OJz7qZ5AZo
"""

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense


#Get CSV Dataset into a dataframe
df = pd.read_csv("https://raw.githubusercontent.com/elryan75/Stock-Predictor/main/Dataset/all_stocks_5yr.csv")


#Remove all rows containing Nan values
df = df.dropna()


#Get the data for the company Facebook
fb = (df.loc[df['Name'] == 'FB'])


#Only keep closing price and volume columns
training_set = fb.iloc[:, 4:6].values


#Normalization of the data
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


#Separate into X and y, in the target y we only want the closing price
X = []
y = []

for i in range(len(training_set_scaled)-1):
  X.append([training_set_scaled[i][0],training_set_scaled[i][1]])
  y.append(training_set_scaled[i][0])

X = np.array(X)
y = np.array(y)

#Split data for training and testing
split = int(len(X)*0.8)

Xtrain = X[:split]
Xtest = X[split : len(X)]
ytrain = y[:split]
ytest = y[split : len(y)]

#Now we shift the y array
X_train = []
X_test = []
y_train = []
y_test = []

#We want to use three previous days to predict the next closing price
#So we use Day1, Day2, Day3, to predict Day 4
n=3
for i in range(n, len(Xtrain)):
  X_train.append(Xtrain[i-n : i, : Xtrain.shape[1]])
  y_train.append(ytrain[i])

for i in range(n, len(Xtest)):
  X_test.append(Xtest[i-n : i, : Xtest.shape[1]])
  y_test.append(ytest[i])


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

#Reshape the arrays to use in the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

#Creation of the LSTM model

model = Sequential()
lstm = LSTM(4, input_shape=(X_train.shape[1], X_train.shape[2]))
model.add(lstm)
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer = "adam")

model.fit(X_train, y_train, epochs=200)
model.summary()

#Predict on train and test
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

#The data is still normalized, in order to get the real values, we need to reshape the array to use the inverse_transform() function
trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]

trainPredict = sc.inverse_transform(trainPredict)
trainPredict = trainPredict[:,0]
testPredict = sc.inverse_transform(testPredict)
testPredict = testPredict[:,0]

#Calculate mean squared error 

print("Mean Squared Error")
train_score = mean_squared_error([x[0][0] for x in X_train], trainPredict, squared=False)
print("Training Score: " + str(train_score))

test_score = mean_squared_error([x[0][0] for x in X_test], testPredict, squared=False)
print("Testing Score: " + str(test_score))

#Plot the predicted and actual price of the stock

plt.title("Stock price of Apple")

#Plot the real price of facebook (4 years)
full_prices = training_set[:,0]
days = list(range(0, len(full_prices)))
plt.plot(days, training_set[:,0], label = "Real Price")


#Plot the predicted price using the testing data
predicted_prices = testPredict
prediced_days = days[len(days)-len(testPredict):len(days)]
plt.plot(prediced_days,testPredict, label = "Predicted Price")

plt.xlabel('Day')
plt.ylabel('Stock Price')

plt.legend()
plt.show()