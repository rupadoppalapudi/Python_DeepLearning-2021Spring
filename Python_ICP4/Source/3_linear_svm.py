# Implement linear SVM method using scikit library
# Use train_test_split to create training and testing part
# Evaluate the model on test part

# importing libraries
from sklearn.model_selection import train_test_split
import pandas as pds
from sklearn.svm import SVC
from sklearn import metrics

# fetching the data from glass.csv file
train_df = pds.read_csv('glass.csv')
X_train_drop = train_df.drop("Type", axis=1)
Y_train = train_df["Type"]
X_train, X_test, y_train, y_test = train_test_split(X_train_drop, Y_train, test_size=0.25)

# calling the svm model and training the model with the train sets using the fit method
svm_lin = SVC()
svm_lin.fit(X_train, y_train)

# predicting test data using predict function
predict = svm_lin.predict(X_test)

# evaluating the model by calculating the accuracy score
metrics.accuracy_score(y_test, predict)
print("\nclassification_report :\n", metrics.classification_report(y_test, predict))
