# Change the data source to Breast Cancer dataset * available in the source code folder and make required changes.
# Report accuracy of the model.

# importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense

# load the dataset - fetching the data from breastcancer.csv file
import pandas as pd
dataset = pd.read_csv("breastcancer.csv")

# extracting features
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values
print(dataset.iloc[:, 1].value_counts())

# encoding the categorical data
lb_enc = LabelEncoder()

# fitting label encoder
y = lb_enc.fit_transform(y)

# splitting the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# implementing the model for breastcancer dataset
model_bc = Sequential()  # create model
model_bc.add(Dense(20, input_dim=30, activation='relu'))  # hidden layer
model_bc.add(Dense(1, activation='sigmoid'))              # output layer
model_bc.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
model_bc.fit(X_train, y_train, epochs=100, verbose=0,
                                     initial_epoch=0)

# getting the string summary of the network
print(model_bc.summary())

# getting the loss value & metrics values for the model in test mode
print(model_bc.evaluate(X_test, y_test))
