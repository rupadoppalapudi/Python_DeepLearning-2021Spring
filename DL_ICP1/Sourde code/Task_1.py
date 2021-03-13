# 1.	Use the use case in the class:
# a.	Add more Dense layers to the existing code and check how the accuracy changes.

# importing the libraries
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load the dataset - fetching the data from diabetes.csv file
data = pd.read_csv("diabetes.csv", header=None).values

# splitting the data into training and test datav
X_train, X_test, Y_train, Y_test = train_test_split(data[:, 0:8], data[:, 8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
model_seq = Sequential()  # create model
model_seq.add(Dense(20, input_dim=8, activation='relu'))  # hidden layer
# adding more Dense layers
model_seq.add(Dense(30, activation='relu'))  # hidden layer
model_seq.add(Dense(35, activation='relu'))  # hidden layer
model_seq.add(Dense(1, activation='sigmoid'))  # output layer
model_seq.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])
model_seq.fit(X_train, Y_train, epochs=100,
          initial_epoch=0)
print(model_seq.summary())         # get summary

# getting the loss value & metrics values for the model in test mode
print(model_seq.evaluate(X_test, Y_test))
