#################################
# Week 3.1: K Nearest Neighbors #
#################################

# importing libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing

##################
# Importing Data #
##################

# importing library
import wget

# downloading csv file using wget
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv'
# wget.download(url, 'teleCust1000t.csv')

# opening csv file and reading it into a variable called df
df = pd.read_csv("teleCust1000t.csv")

# checking dataset head
print(df.head())

###################################
# Data Visualization and Analysis #
###################################

# checking frequency tabs for one of our columns
print(df['custcat'].value_counts())

# category 3 has the highest service members
# plus service, that is

# as listed above here are the value labels for each value
# custcat is our depedent var for which we want to predict values for a new entrant
'''
1- Basic Service
2- E-Service
3- Plus Service
4- Total Service
'''

# time to plot a histogram of our customers' income groups
df.hist(column='income', bins=50)
# plt.show()
# ok, it's evident this distribution is skewed right, as always

###############
# Feature Set #
###############

# printing off some of our columns
# to hone in on key features
print(df.columns)

# converting pd df to np array for scikitlearn usage
X = df[['region', 'tenure', 'age', 'marital', 'address',
        'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values # .astype(float)
print(X[0:5])

# subsetting and selecting away my variable of choice
y = df['custcat'].values
print(y[0:5])

########################
# Normalizing our Data #
########################

# literally normalizes - mean 0, var 1
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])


# creating a test train split suited to our problem
from sklearn.model_selection import train_test_split

# test size is 20 percent
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print  ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)


##################
# Classification #
##################

# K nearest neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier

# Training the algorithm, with k = 4
k = 4

# training model and predicting
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh)

# Predicting estimated(y-values) using testset(x-values) as input
yhat = neigh.predict(X_test)
yhat[0:5]

# Evaluating model accuracy using inbuilt sklearn functions
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))




############
# Practice #
############

# building algorithm with k = 6

# Training the algorithm, with k = 4
k = 6

# training model and predicting
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh)

# Predicting estimated(y-values) using testset(x-values) as input
yhat = neigh.predict(X_test)
yhat[0:5]

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# appears that the algorithm is more accurate when k = 4

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1, Ks):

    # Training our Model and Predicting
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

# plotting model-accuracy for different numbers of neighbors

plt.plot(range(1,Ks), mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,
                            mean_acc + 1 * std_acc,
                            alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)






























# in order to display plot within window
# plt.show()
