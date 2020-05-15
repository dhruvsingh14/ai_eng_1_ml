############################
# Week 3.2: Decision Trees #
############################

# importing libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

##################
# Importing Data #
##################

# importing library
import wget

# downloading csv file using wget
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv'
# wget.download(url, 'drug200.csv')

# reading in our csv on pharmaceutical
my_data = pd.read_csv("drug200.csv", delimiter=",")

# checking our dataset, high level
my_data[0:5]

############
# Practice #
############

# checking dataset dimensions
my_data.size
my_data.shape

##################
# Pre-processing #
##################

# subsetting
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

# preprocessing our data
from sklearn import preprocessing

##############labeling################

# creating and applying label for column 1
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

# creating and applying label for column 2
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# creating and applying label for column 3
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

#############target#var###############

# declaring target variable
y = my_data["Drug"]
y[0:5]

################################
# Setting up our Decision Tree #
################################

# importing preprocessing package
from sklearn.model_selection import train_test_split

# creating a test train split using python package
X_trainset, X_testset, y_trainset, y_testset = train_test_split(
                                X, y, test_size=0.3, random_state=3)


# Practice

# 1: display shapes and size of trainsets

# size means num of cells, wheras shape shows dimension
X_trainset.size
X_trainset.shape

y_trainset.size
y_trainset.shape

# 2: display shapes and size of testsets
X_testset.size
X_testset.shape

y_testset.size
y_testset.shape

# naturally, both testsets and both trainsets have the same
# number of rows

# naturally, both x datasets and both y dataset have the same
# number of columns

#####################
# Modeling our Data #
#####################

# creating decision tree object
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree

# fitting decision tree classifications to our training data
drugTree.fit(X_trainset, y_trainset)

###############################
# Prediction on our test data #
###############################

# using decision tree object to predict test data classification
predTree = drugTree.predict(X_testset)

# printing outcome
predTree [0:5]
y_testset [0:5]

# model performs fairly well, predicts all values correctly

###################################
# Evaluation of the Decision Tree #
###################################

# importing scoring mechanism
from sklearn import metrics
import matplotlib.pyplot as plt

# super high accuracy as we can see
# print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#########################
# Visualization Tactics #
#########################

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

# from sklearn.tree import export_graphviz

import sklearn
print(sklearn.__file__)

dot_data = StringIO()
# filename = "drugtree.png"

'''
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()

out = tree.export_graphviz(drugTree, feature_names=featureNames,
                        out_file=dot_data, class_names= np.unique(y_trainset),
                        filled=True, special_characters=True,rotate=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(100, 200))

plt.imshow(img,interpolation='nearest')
'''

# skipping this funky part because graphviz refuses
# to operate in python 3.6
# i'm sure it's just some funk to do with adding
# the path to the environmental variables.

# nonetheless, only the graphing tool is off.
# the decision tree predictor still works a ok :) just fine!

# so technically i skipped the decision tree graphing portion, big deal.
# i retained the crux of the matter.
# also, tried plotting online, on ibm. took wayy too long to load.

# can return if necessary.






















# in order to display plot within window
# plt.show()
