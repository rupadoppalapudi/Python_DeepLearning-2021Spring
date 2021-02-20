# Implement Naïve Bayes method using scikit-learn library
# Use train_test_split to create training and testing part
# Evaluate the model on test part

# importing libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pds
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

# fetching the data from glass.csv file
train_df = pds.read_csv('glass.csv')
X_train_drop = train_df.drop("Type", axis=1)
Y_train = train_df["Type"]

# creating training and testing part from train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X_train_drop, Y_train, test_size=0.25)

# calling the naive bayes classifier model and training the model with the train sets using the fit method
NBModel = GaussianNB()
NBModel.fit(X_test, y_test)

# predicting test data using predict function
predict = NBModel.predict(X_test)

# evaluating the model by calculating the accuracy score
accuracy_score(y_test, predict)
print("\nclassification_report :\n", metrics.classification_report(y_test, predict) )