# Create Multiple Regression for the “Restaurant Revenue Prediction” dataset.
# Evaluate the model using RMSE and R2 score.

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# fetching the data from data.csv file
train = pd.read_csv('data.csv')

# working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

# find out the number of null values for the features
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

# handling missing or null value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
# print(sum(data.isnull().sum() != 0))

# building a multiple linear model
y = np.log(data.revenue)             # extracting label (revenue field)
X = data.drop(['revenue'], axis=1)   # extracting features by excluding label i.e., revenue
from sklearn.model_selection import train_test_split

# splitting the data into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
from sklearn import linear_model
lr = linear_model.LinearRegression()

# training the model by using fit method
model = lr.fit(X_train, y_train)

# evaluating the performance of the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# predicting test data
y_actual = model.predict(X_test)
print("\nR^2 is obtained as : ", r2_score(y_test, y_actual))            # R2 score
print("RMSE is obtained as : ", mean_squared_error(y_test, y_actual))   # RMSE