#################################
# Week 3.3: Logistic Regression #
#################################

# importing libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

##################
# Importing Data #
##################

# importing library
import wget

# downloading Customer Churn project data using wget
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv'
# wget.download(url, 'ChurnData.csv')

# opening churn data and reading it into a variable called churn_df
churn_df = pd.read_csv("ChurnData.csv")

# checking dataset head
churn_df.head()

###########################
# Pre-processing our data #
###########################

# subsetting to releveant columns
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed',
                    'employ', 'equip', 'callcard', 'wireless',
                    'churn']]

# changing var type
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

##############
# Practicing #
##############

# checking for dimensions
churn_df.shape

# X is the feature variables set, y is the target output
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income',
                    'ed', 'employ', 'equip']])
X[0:5]

# making churn the target variable
y = np.asarray(churn_df['churn'])
y[0:5]

# also normalizing the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#############################
# Creating Test Train Split #
#############################

from sklearn.model_selection import train_test_split

# 20 - 80 split, 4 folds
X_train, X_test, y_train, y_test = train_test_split( X,
y, test_size=0.2, random_state=4)

# printing test sets, and train sets side by side
# print ('Train set:', X_train.shape, y_train.shape)
# print ('Test set:', X_test.shape, y_test.shape)

###################################
# Modeling: Logit w/ Scikit Learn #
###################################

# importing libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# iteration 1: inverse regularization = .01, solver = liblinear

# fitting regression model to our training dataset
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

# predicting outcome variable of interest
yhat = LR.predict(X_test)
yhat

# this returns probabilities of all binary outcomes yhat
yhat_prob = LR.predict_proba(X_test)
yhat_prob

############################################
# Evaluating our Logistic Regression model #
############################################

# importing scoring metric
from sklearn.metrics import jaccard_similarity_score

#################
# jaccard index #
#################
print(jaccard_similarity_score(y_test, yhat))
# so the model performs pretty well - scoring a .750,
# or 75% similarity levels

###################################
# constructing a confusion matrix #
###################################

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    '''printing and plotting the confusion matrix
    can normalize using option, normalize=True
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normlized confusion matrix")

    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # creating plot features
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # formatting it to our liking
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

confusion_matrix(y_test, yhat, labels=[1,0])

# computing confusion matrix, to predict false positives, and false negatives
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# plotting non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1',
            'churn=0'], normalize= False, title='Confusion matrix')
# plt.show()

# beautiful display of false positives and false negatives
# printing out our vals for comparison
classification_report(y_test, yhat)

# log loss calculations
from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))


############
# practice #
############

# iteration 2: inverse regularization = .05, solver = sag

# fitting new regression model to our training dataset
LR2 = LogisticRegression(C=0.05, solver='sag').fit(X_train,y_train)
LR2

# predicting outcome variable of interest
yhat2 = LR2.predict(X_test)
yhat2

# this returns probabilities of all binary outcomes yhat
yhat_prob2 = LR2.predict_proba(X_test)
yhat_prob2

# evaluating using jacard index
print(jaccard_similarity_score(y_test, yhat2))
# so the model performs fairly well - scoring a .725, though less well than 1st iteration model
# or 72.5% similarity levels

# evaluating using confusion matrix
confusion_matrix(y_test, yhat2, labels=[1,0])

# confusion matrix object, for plotting
cnf_matrix2 = confusion_matrix(y_test, yhat2, labels=[1,0])
np.set_printoptions(precision=2)

# plotting non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1',
            'churn=0'], normalize= False, title='Confusion matrix')
plt.show()

# printing out fp's, and fn's for comparison
classification_report(y_test, yhat2)

# evaluating using log loss
print(log_loss(y_test, yhat_prob2))


# the non-normalized confusion matrixes show
# that the two models perform essentially the same on classification

# though jaccard indices show one is closer to the true values
# while the other isn't

































# in order to display plot within window
# plt.show()
