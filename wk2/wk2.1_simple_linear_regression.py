######################################
# Week 2.1: Simple Linear Regression #
######################################

# importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

##################
# Importing Data #
##################

# importing library
import wget

# downloading csv file using wget
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
# wget.download(url, 'FuelConsumption.csv')

# opening text file and reading it into a variable called alice_novel
df = pd.read_csv("FuelConsumption.csv")

# checking dataset head
df.head()

#####################
# Exploration Phase #
#####################

# summarizing
df.describe()

# drilling down on certain features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# plotting each of our features in a panel histogram
'''
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()
'''

# emissions vs fuel_consumption, scatter
'''
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
'''

# emissions vs engine_size, scatter
'''
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
'''

############
# Practice #
############

# emissions vs cylinder, scatter
'''
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()
'''

# creating a test - train split in data (80 - 20)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

############################
# Simple Linear Regression #
############################

## plotting training data ~ same scatter plots repeated for training and test datasets

# emissions vs engine_size, scatter
'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
'''

# modeling using sklearn
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

# The coefficients
# print ('Coefficients: ', regr.coef_)
# print ('Intercept: ', regr.intercept_)

# Plot outputs
# plotting fitted line over our data
'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
'''

# preparing to evaluate our model for a good fit
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))








































# in order to display plot within window
# plt.show()
