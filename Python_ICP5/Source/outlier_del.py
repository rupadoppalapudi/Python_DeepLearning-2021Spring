# Delete all the outlier data for the GarageArea field
# for this task you need to plot GaurageArea field and SalePrice in scatter plot, then check which numbers are anomalies.

# importing libraries
import pandas as pd
import matplotlib.pyplot as splt

splt.style.use(style='ggplot')
splt.rcParams['figure.figsize'] = (10, 6)

# fetching the data from train.csv file
train = pd.read_csv('train.csv')

train.SalePrice.describe()

####---------------- scatter plot for GaurageArea field and SalePrice ----------------####

# before deleting the outlier data for GaurageArea field
print(train[['GarageArea']])
splt.scatter(train.GarageArea, train.SalePrice, alpha=.75, color='b')
splt.show()

# after deleting the outlier data for GaurageArea field
filtered = train[(train.GarageArea < 1000) & (train.GarageArea > 200)]
splt.scatter(filtered.GarageArea, filtered.SalePrice, alpha=.75, color='g')
splt.show()