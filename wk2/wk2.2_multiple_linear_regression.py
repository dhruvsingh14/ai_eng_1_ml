######################################
# Week 2.2: Multiple Linear Regression #
######################################

# importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

##################
# Importing Data #
##################

# opening text file and reading it into a variable called alice_novel
df = pd.read_csv("FuelConsumption.csv")

# checking dataset head
df.head()

#####################
# Exploration Phase #
#####################

# drilling down on certain features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
          'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# overall data: emissions vs engine size, scatter
'''
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
'''

####################
# Test-train split #
####################

# creating a test - train split in data (80 - 20)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# plotting training data distribution
# training data: emissions vs engine_size, scatter
'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
'''

#############################
# Multiple Regression Model #
#############################

# modeling using sklearn
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)

# The coefficients
print ('Coefficients: ', regr.coef_)

############################################
# Prediction: OLS (Ordinary Least Squares) #
############################################

y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - y) ** 2))

# explained variance score: where 1 represents perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

############
# Practice #
############

# declaring a regression object

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_CITY',
                        'FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)

# The coefficients
print ('Coefficients: ', regr.coef_)

# Practice Prediction
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_CITY',
                        'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_CITY',
                        'FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - y) ** 2))

# explained variance score: where 1 represents perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

# this doesn't seem to result in any better variance than before

# it does explain the same amount of variance, that is 85%

# though the coefficients are changed, given the different variables






























































# in order to display plot within window
# plt.show()
