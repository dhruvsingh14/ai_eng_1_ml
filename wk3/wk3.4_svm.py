####################################
# Week 3.4: Support Vector Machine #
####################################

# importing libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##########################
# Loading Cancer dataset #
##########################

# importing library
import wget

# download, and save
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cell_samples.csv'
# wget.download(url, 'cell_samples.csv')

# read in, print
cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()

# Step 1:
'''
since class (benign, or malignant) is what we want to predict
we set it aside as our target variable
'''

# plotting our data points as an overlayed scatterplot

# plot 1: class vs unifsize
'''
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump',
                y = 'UnifSize', color='DarkBlue', label='malignant');
'''

# plot 2: class vs clump, superimposed
'''
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize',
                color='Yellow', label='benign', ax=ax);
'''
# plt.show()

##############################
# Step 2: Data preprocessing #
##############################

# getting an estimate of data types
cell_df.dtypes

# dropping non numeric rows from barenuc
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
# converting bare nuclei to an integer.
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

# subsetting for predictor vars
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh',
                    'SingEpiSize', 'BareNuc', 'BareNuc', 'BlandChrom',
                    'NormNucl', 'Mit']]
# converting to arrays
X = np.asarray(feature_df)
X[0:5]

# doing the same for outcome variables
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y[0:5]

#############################
# Step 3: Modeling our Data #
#############################

# creating a test and training split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# good way to size up the shapes and sizes of split in 1 line of code.
# print ('Train set:', X_train.shape, y_train.shape)
# print ('Test set:', X_test.shape, y_test.shape)

# using default rbf for modeling in svm: radial basis function
from sklearn import svm

# declaring and fitting object
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# predicting outcome values
yhat = clf.predict(X_test)
yhat [0:5]

# there is no way around choosing different models,
# and comparing results, to then choose the best performing model

##############################
# Step 3: Evaluation metrics #
##############################

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):

    """
    the current function print and plots confusion matrix
    normalization can be applied by setting parameter normalize = true
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]),
                                    range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# computing confusion matrix for current data
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# plotting our non normalized matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign (2)',
                        'Malignant (4)'], normalize= False,
                        title='Confusion matrix')
plt.show()

# using the f1_score for scoring performance
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

# using the jaccard index for scoring performance
from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))


########################################
# Step 4: Practice metrics, iterations #
########################################

# iterating with a different functional type: linear
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)

# predicting outcome values
yhat2 = clf2.predict(X_test)
print(yhat2[0:5])

# computing confusion matrix with new function fitted
cnf_matrix2 = confusion_matrix(y_test, yhat2, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat2))

# plotting our non normalized matrix
plt.figure()
plot_confusion_matrix(cnf_matrix2, classes=['Benign (2)',
                        'Malignant (4)'], normalize= False,
                        title='Confusion matrix 2')
plt.show()

# using the f1_score for scoring linear model performance
print(f1_score(y_test, yhat2, average='weighted'))

# using the jaccard index for scoring linear model performance
print(jaccard_similarity_score(y_test, yhat2))
































# in order to display plot within window
# plt.show()
