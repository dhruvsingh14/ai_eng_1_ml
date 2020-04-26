###################################
# Week 2.3: Polynomial Regression #
###################################

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

# checking the top of the dataset
df.head()

#####################
# Exploration Phase #
#####################

# drilling down on certain features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB',
        'CO2EMISSIONS']]
cdf.head(9)

# overall data plot: emissions vs engine size, scatter
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

######################################
# Polynomial Regression, iteration 1 #
######################################

# importing sklearn preprocessing package
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
# storing training set dependent / indep vars as arrays
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# storing test set dependent / indep vars as arrays
test_x = np.asanyarray(train[['ENGINESIZE']])
test_y = np.asanyarray(train[['CO2EMISSIONS']])

# setting degree of polynomial function model
poly = PolynomialFeatures(degree=2)

# transforming training+independents using polynomial for fit
train_x_poly = poly.fit_transform(train_x)
train_x_poly

# declaring a regression object
clf  = linear_model.LinearRegression()

#########################################
# Fitting Polynomial Model, iteration 1 #
#########################################
train_y_ = clf.fit(train_x_poly, train_y)

# pulling coefficients
print('Coefficients: ', clf.coef_)
print('Intercepts: ', clf.intercept_)

## plotting our newly fitted model
'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)

# using regression coefficients to plot fitted line
plt.plot(XX, yy, '-r')

# labels
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
'''

###########################
# Evaluation, iteration 1 #
###########################

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_, test_y) )

############
# Practice #
############

######################################
# Polynomial Regression, iteration 2 #
######################################

# setting degree of polynomial function model
poly2 = PolynomialFeatures(degree=3)

# transforming training+independents using polynomial for fit
train_x_poly2 = poly2.fit_transform(train_x)
train_x_poly2

# declaring a regression object
clf2  = linear_model.LinearRegression()

#########################################
# Fitting Polynomial Model, iteration 2 #
#########################################
train_y2_ = clf2.fit(train_x_poly2, train_y)

# pulling coefficients
print('Coefficients: ', clf2.coef_)
print('Intercepts: ', clf2.intercept_)

## plotting our newly fitted model

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX2 = np.arange(0.0, 10.0, 0.1)
yy2 = clf2.intercept_[0]+ clf2.coef_[0][1]*XX2+ clf2.coef_[0][2]*np.power(XX2, 2)+ clf2.coef_[0][3]*np.power(XX2, 3)

# using regression coefficients to plot fitted line
plt.plot(XX2, yy2, '-r')

# labels
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


###########################
# Evaluation, iteration 2 #
###########################

from sklearn.metrics import r2_score

test_x_poly2 = poly2.fit_transform(test_x)
test_y2_ = clf2.predict(test_x_poly2)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y2_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y2_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y2_, test_y) )

# yes, it is a better fit, the R^2 is roughly grt/eq,
# and both mse, and mean abs error are smaller!

# now! keep in mind, that each time you run this model, you'll
# get a slightly different model

# partly b/c of the test train split being randomly done.

# if however you stored the test train vars,

# the estimation is likely to be consistent






















# in order to display plot within window
# plt.show()
