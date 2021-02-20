# Find top 5 most correlated features to the target label(revenue) and then build a model on top of those 5 features.
# Evaluate the model using RMSE and R2 score and then compare the result with the RMSE and R2 you achieved in question 2

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

# finding the correlation with the numeric functions
corr = numeric_features.corr()

# top 5 correlated features with the label revenue
print(corr['revenue'].sort_values(ascending=False)[:6], '\n')
print(corr['revenue'].sort_values(ascending=False)[-5:], '\n')
most_cor = ['P2', 'P6', 'P11', 'P21', 'P28', 'P34', 'P10', 'P8', 'P13', 'P29']

# find out the number of null values for the features
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

# handling missing or null value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
# print(sum(data.isnull().sum() != 0))

# building a multiple linear model
y = np.log(data.revenue)
X = data.drop(['revenue'], axis=1)
X = X[most_cor]

# splitting the data into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
from sklearn import linear_model
lr = linear_model.LinearRegression()

# training the model by using fit method
model = lr.fit(X_train, y_train)

# evaluating the performance of the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_actual = model.predict(X_test)                                       # predicting test data
print("\nR^2 is obtained as : ", r2_score(y_test, y_actual))           # R2 score
print("RMSE is obtained as : ", mean_squared_error(y_test, y_actual))  # RMSE