# Create Multiple Regression for the “Restaurant Revenue Prediction” dataset.
# Evaluate the model using RMSE and R2 score.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('data.csv')

#Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()
# print(corr['revenue'])

print(corr['revenue'].sort_values(ascending=False)[:6], '\n')
# print(corr['revenue'].sort_values(ascending=False)[-5:], '\n')
top_5 = ['P2', 'P6', 'P11', 'P21', 'P28']

# Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

##Build a linear model
y = np.log(data.revenue)
X = data.drop(['revenue'], axis=1)
X = X[top_5]
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

##Evaluate the performance and visualize results
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_dash = model.predict(X_test)
print("R^2 is obtained as : ", r2_score(y_test, y_dash))
print("RMSE is obtained as : ", mean_squared_error(y_test, y_dash))