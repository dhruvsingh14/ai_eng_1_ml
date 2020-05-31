################################
# Week 4.1: K-Means Clustering #
################################

## Note: clustering is considered an unsupervised learning method

# importing libraries
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

###################
# Generating Data #
###################

# importing library
np.random.seed(0)

# creating clusters or blobs of 5000 points at a time
# centered at the named coordinates

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1],
                [2, -3], [1, 2]], cluster_std=0.9)

# plotting the data: iteration 1: 4 clusters
plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.show()

# initializing k means feature matrix
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# fitting feature matrix to blobs above
k_means.fit(X)

# labelling each point
k_means_labels = k_means.labels_
k_means_labels

# grabbing the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#######################################
# Creating a data visual: iteration 1 #
#######################################
# initializing dimensions
fig = plt.figure(figsize=(6, 4))

# colors: using map, produces array given number of labels
# k_means_labels - helps produce coloring
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# building a plot
ax = fig.add_subplot(1, 1, 1)

# for loop plotting data points and centroids
# k ranges from 0-3, and matches the cluster number

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])),
                            colors):

    # creating a list of all data points
    # checking for 'is a part of' cluster
    # labelling points in the set as true, else false
    my_members = (k_means_labels == k)

    # defining the centroid, or cluster center, using function calculator
    cluster_center = k_means_cluster_centers[k]

    # plotting datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                            markerfacecolor=col, marker='.')

    # plotting centroids distinctly
    ax.plot(cluster_center[0], cluster_center[1], 'o',
                            markerfacecolor=col, markeredgecolor='k',
                            markersize=6)

# titling
ax.set_title('KMeans')

# Removing x and y axis ticks
ax.set_xticks(())
ax.set_yticks(())

# showing the plot
# plt.show()

#########################
# Practice: iteration 2 #
#########################

# picking initial centroids is somewhat arbitrary
X, y = make_blobs(n_samples=5000, centers=[[-1,3], [2, -1],
                [-2, -2]], cluster_std=0.9)

# plotting the data: iteration 2: 3 clusters
plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.show()

# initializing k means feature matrix
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

# fitting feature matrix to blobs above
k_means.fit(X)

# labelling each point
k_means_labels = k_means.labels_
k_means_labels

# grabbing the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#######################################
# Creating a data visual: iteration 2 #
#######################################
# initializing dimensions
fig = plt.figure(figsize=(6, 4))

# colors: using map, produces array given number of labels
# k_means_labels - helps produce coloring
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# building a plot
ax = fig.add_subplot(1, 1, 1)

# for loop plotting data points and centroids
# k ranges from 0-3, and matches the cluster number

for k, col in zip(range(len([[-1,3], [2, -1], [-2, -2]])),
                            colors):

    # creating a list of all data points
    # checking for 'is a part of' cluster
    # labelling points in the set as true, else false
    my_members = (k_means_labels == k)

    # defining the centroid, or cluster center, using function calculator
    cluster_center = k_means_cluster_centers[k]

    # plotting datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                            markerfacecolor=col, marker='.')

    # plotting centroids distinctly
    ax.plot(cluster_center[0], cluster_center[1], 'o',
                            markerfacecolor=col, markeredgecolor='k',
                            markersize=6)

# titling
ax.set_title('KMeans')

# Removing x and y axis ticks
ax.set_xticks(())
ax.set_yticks(())

# showing the plot
# plt.show()
# cluster pyramid has been established

######################################
# Customer Segmentation: Application #
######################################
# importing library
import wget

# download, and save
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv'
# wget.download(url, 'Cust_Segmentation.csv')

# read in, print
import pandas as pd

cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()


#########################################
# Customer Segmentation: Pre-processing #
#########################################

# dropping unnecessary vars
df = cust_df.drop('Address', axis=1)
print(df.head())

# normalizing/ standardizing data:
from sklearn.preprocessing import StandardScaler

# there are a few different ways to do this
# here, using the standard deviation
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

###################################
# Customer Segmentation: Modeling #
###################################

# setting a global variable
clusterNum = 3

# initializing
k_means = KMeans(init = "k-means++", n_clusters = clusterNum,
                n_init = 12)

# fitting model to real datapoints
k_means.fit(X)

# extracting assigned cluster
labels = k_means.labels_
labels

###################################
# Customer Segmentation: Insights #
###################################

# new column
df["Clus_km"] = labels
df.head(5)

# checking centroid values, w/o resorting to a function
print(df.groupby('Clus_km').mean())
# displays mean value for each var per cluster centroid

#############plot 1: 2d##########################

# plotting little circles for each cluster
area = np.pi * ( X[:, 1])**2

plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
# plt.show()


#############plot 2: 3d##########################

# plotting 3d figure of income, by age, by education
from mpl_toolkits.mplot3d import Axes3D

# initializing dimensions
fig = plt.figure(1, figsize=(8, 6))
plt.clf()

# declaring 3d object
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)

# labeling axes
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
# plt.show()






































# in order to display plot within window
# plt.show()
