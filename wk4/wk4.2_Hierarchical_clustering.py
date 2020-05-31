#####################################
# Week 4.2: Hierarchical Clustering #
#####################################

# importing libraries
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

###################
# Generating Data #
###################

# specifying cluster centers
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1],
                                [1, 1], [10,4]], cluster_std=0.9)

# plotting scatterplot
'''
plt.scatter(X1[:, 0], X1[:, 1], marker='o')
plt.show()
'''

############################
# Agglomerative Clustering #
############################

# aggregation: declaring an agglomerative clustering object
agglom = AgglomerativeClustering(n_clusters = 4,
                                linkage = 'average')

# fitting model to the data
agglom.fit(X1, y1)

# creating a figure frame using dimensions 6 x 4
plt.figure(figsize=(6,4))

# scaling data points down, to fit closer together

# min-max range for X1
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# averaging distance for X1
X1 = (X1 - x_min) / (x_max - x_min)

# looping to display all datapoints
for i in range(X1.shape[0]):
    # replacing pts. w/ cluster value
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
            color=plt.cm.nipy_spectral(agglom.labels_ [i] / 10.),
            fontdict={'weight':'bold', 'size': 9})

# removing x ticks, y ticks, and y axis
plt.xticks([])
plt.yticks([])

# plt.axis('off')

# displauing the plot of original data, then clustering
'''
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()
'''

#########################################
# Agglom. Clustering to Dendrogram Plot #
#########################################

# printing distance matrix between features
dist_matrix = distance_matrix(X1, X1)
dist_matrix

# declaring clustering object
Z = hierarchy.linkage(dist_matrix, 'complete')

# fitting model
'''
dendro = hierarchy.dendrogram(Z)
plt.show()
'''

# declaring clustering object: iteration 2
Z2 = hierarchy.linkage(dist_matrix, 'average')

# fitting model iteration 2
'''
dendro = hierarchy.dendrogram(Z2)
plt.show()
'''

#############################
# Application: Vehicle Data #
#############################

# importing library
import wget

# download, and save
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv'
# wget.download(url, 'cars_clus.csv')

# reading in
filename = 'cars_clus.csv'
pdf = pd.read_csv(filename)

# printing specs
# print("Shape of dataset: ", pdf.shape)
pdf.head(5)

##########################
# Vehicle Data: Cleaning #
##########################

# printing specs
# print("Shape of dataset before cleaning: ", pdf.size)

# conversion type to num
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
    'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt',
    'fuel_cap', 'mpg', 'lnsales']] = pdf[['sales', 'resale',
                                'type', 'price', 'engine_s',
                                'horsepow', 'wheelbas', 'width',
                                'length', 'curb_wgt', 'fuel_cap',
                                'mpg', 'lnsales']].apply(pd.to_numeric,
                                errors='coerce')

# dropping missing values
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
# print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

###################################
# Vehicle Data: Feature Selection #
###################################

featureset = pdf[['engine_s', 'horsepow', 'wheelbas', 'width',
                'length', 'curb_wgt', 'fuel_cap', 'mpg']]

###############################
# Vehicle Data: Normalization #
###############################

from sklearn.preprocessing import MinMaxScaler
x = featureset.values # grabs var list into an array

min_max_scaler = MinMaxScaler()

# preparing to scale for comparability
feature_mtx = min_max_scaler.fit_transform(x)
print(feature_mtx [0:5])

#####################################
# Vehicle Data: Clustering w/ Scipy #
#####################################

import scipy
leng = feature_mtx.shape[0]

# initializing
D = scipy.zeros([leng,leng])

# calculating and storing distances
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i],
                                                feature_mtx[j])

import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

from scipy.cluster.hierarchy import fcluster
max_d = 3

# assigning clusters to vehicle data
clusters = fcluster(Z, max_d, criterion='distance')
clusters

# setting cluster number
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)


# finally, plotting the dendrogram
fig = plt.figure(figsize=(18, 50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id],
                            pdf['model'][id],
                            int(float(pdf['type'][id])))

dendro = hierarchy.dendrogram(Z, leaf_label_func=llf,
                        leaf_rotation=0, leaf_font_size=12,
                        orientation = 'right')


############################################
# Vehicle Data: Clustering w/ Scikit-learn #
############################################

# creating distance matrix
dist_matrix = distance_matrix(feature_mtx, feature_mtx)
print(dist_matrix)

# declaring and fitting clustering object
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_

# assigning labels
pdf['cluster_'] = agglom.labels_

# printing data
pdf.head()

import matplotlib.cm as cm
n_clusters = max(agglom.labels_) + 1

# coloring and labeling clusters
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# creating figure frame
plt.figure(figsize=(16,14))


for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i],
                str(subset['model'][i]),rotation=25)
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10,
                c=color, label='cluster'+str(label), alpha=0.5)
    # plt.scatter(subset.horsepow, subset.mpg)
    plt.legend()
    plt.title('Clusters')
    plt.xlabel('horsepow')
    plt.ylabel('mpg')

plt.show()




























# in order to display plot within window
# plt.show()
