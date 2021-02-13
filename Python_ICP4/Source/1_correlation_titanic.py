# find the correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case in class

# importing pandas library
import pandas as pds

# fetching the preprocessed train data
training = pds.read_csv('train_preprocessed.csv')

# finding the correlation - statistical summary between target column 'survived' and 'sex' column
print("\n", training[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived'))
print("\nCorrelation :", training['Survived'].corr(training['Sex']))