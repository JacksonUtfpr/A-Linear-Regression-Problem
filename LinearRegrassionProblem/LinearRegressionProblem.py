"""
Your neighbor is a real estate agent and wants some help predicting house prices for regions across the US. 
It would be great if you could somehow create a model for it that would allow you to input some features of a house and return an estimate of 
how much the house would sell.
She asked if you could help her with her new data science skills. You say yes and decide that Linear Regression might be a good way to solve this problem.
Your neighbor then gives you some information about a bunch of homes in US regions. everything is contained in the file: USA_Housing.csv.
The data contains the following columns:

* 'Avg. Area Income': Average income of residents where the house is located.
* 'Avg. Area House Age': Average age of houses in the same city.
* 'Avg. Area Number of Rooms': Average number of rooms for houses in the same city.
* 'Avg. Area Number of Bedrooms': Average number of bedrooms for houses in the same city
* 'Area Population': The population of the city where the house is located.
* 'Price': Sale price of the house.
* 'Address': House address;
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to train the model:
from sklearn.model_selection import train_test_split
# to create the model:
from sklearn.linear_model import LinearRegression
# to see the accuracy of the model:
from sklearn import metrics

# storaging the file inside the variable USAhousing
USAhousing = pd.read_csv('USA_Housing.csv')
print(USAhousing.head())

# an information summary of the Data Frame: the index dtype and columns, non-null values and memory usage.
USAhousing.info()

# to generate descriptive statistics, including those that summarize the central tendency, dispersion and shape of a dataset’s distribution:
USAhousing.describe()

USAhousing.columns
print(USAhousing.columns)

#creating plots to check the data:
sns.pairplot(USAhousing)

sns.displot(USAhousing['Price'])

sns.heatmap(USAhousing.corr())

# training the linear regression model:
# First: divide the data in a matrix X with the traiing resources and a matrix Y with the target variable, which is in this case the price column
# Second: Discard the column "Adress" because it only has text information, that the model can't use.
""" 
The output from 'print(USAhousing.columns)' :

Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
      dtype='object')
"""
x = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

# spliting the model into a group of training and a group of testing
# Creating the model using the training group and later the testing group to evaluate the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

# creating the linear model: lm
lm = LinearRegression()

# The 'fit' method trains the algorithm on the training data, after the model is initialized.
# Is essentially the training part of the modeling process. 
# It finds the coefficients for the equation specified via the algorithm being used. 
# It does not classify the data
lm.fit(x_train,y_train)

# evaluating the model: I goes by checking the coefficients and interpreting them
print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
print(coeff_df)

# getting the prediction of the model:
predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)

# a hystogram:
sns.distplot((y_test-predictions),bins=50);

# measuring the accuracy of the model:
# Mean Absolute Error :
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

# Mean Squared Error : 
print('MSE:', metrics.mean_squared_error(y_test, predictions))

# Root Mean Square Error :
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))