##################################
# Week 2.4: NonLinear Regression #
##################################

# importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Linear
'''
# plotting linear function, degree 1
x = np.arange(-5.0, 5.0, 0.1) # setting up axes cutpoints, and increments

# adjustable slope
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size) # generating some random noise
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata, 'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
'''

# Cubic
'''
# preparing polynomial function, degree 3
x = np.arange(-5.0, 5.0, 0.1)

# adjustable the slope
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
'''

# Quadratic
'''
# Similarly exemplifying what a quadratic, deg=2, will look like
x = np.arange(-5.0, 5.0, 0.1)

# adjustable the slope
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
'''

# Exponential
'''
# setting up an exponential function for plotting
X = np.arange(-5.0, 5.0, 0.1)

# adjustable the slope
Y = np.exp(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
'''

# Logarithmic
'''
# creating a logarithmic function f(x)
X = np.arange(-5.0, 5.0, 0.1)

# adjustable the slope
Y = np.log(X)

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
'''

# Sigmoidal/Logistic
'''
# creating a sigmoidal model
X = np.arange(-5.0, 5.0, 0.1)

# adjustable the slope
Y = 1-4/(1+np.power(3,X-2))

plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
'''

##################################
# Non-Linear Regression: Example #
##################################

# importing libraries
import pandas as pd
import wget

# downloading csv file using wget
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv'
# wget.download(url, 'china_gdp.csv')

##################
# Importing Data #
##################

# opening text file and reading it into a variable called alice_novel
df = pd.read_csv("china_gdp.csv")

# checking dataset head
df.head(10)

#####################
# Plotting Raw Data #
#####################

# pulling columns into x and y variables
x_data, y_data = (df["Year"].values, df["Value"].values)

'''
# setting figure size
plt.figure(figsize=(8,5))

plt.plot(x_data, y_data, 'ro')

plt.ylabel('GDP')
plt.xlabel('Year')

plt.show()
'''

###################################
# Selecting the appropriate model #
###################################

# plotting constructed logistic function
'''
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y)

plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')

plt.show()
'''

######################
# Building our model #
######################

# fitting the model, predicting points using logistic function
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

# applying logistic regression function for prediction
Y_pred = sigmoid(x_data, beta_1, beta_2)

# plotting initial predictions against existing datapoints
'''
plt.plot(x_data, Y_pred*15000000000000.)
plt.show()
'''

# re-running earlier plot for checking
'''
plt.plot(x_data, y_data, 'ro')
plt.show()
'''

# normalizing the data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# using scipy to optimize our model parameters
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)

# printing out actual parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

######################################
# Plotting Logistic Regression Model #
######################################

# setting axis cutpoints
x = np.linspace(1960, 2015, 55)
x = x/max(x)

plt.figure(figsize=(8,5))

# applying sigmoid function
y = sigmoid(x, *popt)

# fitting regression line to normalized data
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')

plt.legend(loc='best')

plt.ylabel('GDP')
plt.xlabel('Year')

plt.show()

############################
# Practice: Model Accuracy #
############################

# honestly i have no effn idea how to evaluate this model.
# I thought i could simply do an mse using y and ydata, but that doesn't work
# copying the provided solution here.

# splitting data into test train set (again 80 - 20 split)
msk = np.random.rand(len(df)) < 0.8

# a new form of test train split
# not sure when to use this rather than using the function
train_x = xdata[msk]
test_x = xdata[~msk]

train_y = ydata[msk]
test_y = ydata[~msk]

# buiding our model using training set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predicting values using testing set
y_hat = sigmoid(test_x, *popt)

# evaluation
from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(y_hat, test_y) )


























# in order to display plot within window
# plt.show()
