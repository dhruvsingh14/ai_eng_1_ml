###############################
# Week 4.3: DBSCAN Clustering #
###############################

# importing libraries
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

##################
# Generating Data #
##################

# function allocates points inputted to centroids, for number of simulations run
def createDataPoints(centroidLocation, numSamples,
                                        clusterDeviation):
    # creating and storing randomly generated data in matrix shell
    # feature matrix: X, target vector: y
    X, y = make_blobs(n_samples=numSamples,
                    centers=centroidLocation,
                    cluster_std=clusterDeviation)

    # standardizing using the mean diff / var method
    X = StandardScaler().fit_transform(X)
    return X, y

# assigning clusters
X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)


################################################
# Modeling our Clusters: Declaring and Fitting #
################################################

epsilon = 0.3
minimumSamples = 7

# fitting dbscan density based clustering to generated data
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)

# checking labels, assignment
labels = db.labels_
labels

#####################
# Identify outliers #
#####################
# creating an array of booleans for testing cluster assignment
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask

# counts number of clusters present
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

# subsetting to only unique values
unique_labels = set(labels)
print(unique_labels)

###############
# Data vizzes #
###############
# creating colors for our clusters
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# plotting points using colors
'''
for k, col in zip(unique_labels, colors):
    if k == -1:
        # black is used for random noise
        col = 'k'

    class_member_mask = (labels == k)

    # plotting datapoints assigned clusters
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o',
                                                alpha=0.5)
plt.show()
'''

######################################
# Practice: contrasting with k-means #
######################################

from sklearn.cluster import KMeans

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

# plotting little circles for each cluster
area = np.pi * ( X[:, 1])**2
'''
plt.scatter(X[:, 0], X[:, 1], s=area, c=labels.astype(np.float), alpha=0.5)
plt.show()
# very sporadic, and not fully capturing the overlay of the cluster
'''
#################################################
# 1: Weather Station Data: Download, Reading In #
#################################################

# download
import csv
import wget

# downloading data
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv'
# wget.download(url, 'weather-stations20140101-20141231.csv')

filename = 'weather-stations20140101-20141231.csv'

# read in, print
pdf = pd.read_csv(filename)
print(pdf.head())

#####################################
# 2: Weather Station Data: Cleaning #
#####################################

# subsetting rows - to remove rows w/ null values in Tm column
pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True) # resetting row index each time we do this
print(pdf.head(5))


##########################################
# 3: Weather Station Data: Visualization #
##########################################

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# from pylab import rcParams # included in matplotlib, pyplot

plt.rcParams['figure.figsize'] = (14,10)

llon=-140
ulon=-50
llat=40
ulat=64

pdf=pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) &
        (pdf['Lat'] > llat) & (pdf['Lat'] < ulat)]


my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, # min longitudes and latitudes
            urcrnrlon=ulon, urcrnrlat=ulat) # max longitudes and latitudes

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
# my_map.shadedrelief() # choosing not to render this fancy layer

# collecting data based on stations
xs, ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))

pdf['xm'] = xs.tolist()
pdf['ym'] = ys.tolist()

'''
# visualization1
for index, row in pdf.iterrows():
#    x, y = my_map(row.Long, row.Lat)
    my_map.plot(row.xm, row.ym, markerfacecolor = ([1,0,0]),
                                marker='o', markersize=5,
                                alpha=0.75)

# plt.text(x,y,stn)
plt.show()
'''
# had to forego some features, but got through the bulk of the training

#######################################
# 4: Weather Station Data: Clustering #
#######################################

from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler

sklearn.utils.check_random_state(1000)

Clus_dataSet = pdf[['xm', 'ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# computing density based clusters
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

# printing few select columns
print(pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5))

# for outliers, cluster label is -1
print(set(labels))

#############################################
# 5: Visualizing Clusters Based on Location #
#############################################

plt.rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, # min longitudes and latitudes
            urcrnrlon=ulon, urcrnrlat=ulat) # max longitudes and latitudes

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
# my_map.shadedrelief() # choosing not to render this fancy layer

# creating a colored map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

# Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4])) if clust_number == -1 else colors[np.int(clust_number)]
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color =c, marker='o',
                    s=20, alpha=0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm)
        ceny=np.mean(clust_set.ym)
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

# plt.text(x,y,stn)
plt.show()

####################################################
# 6: Weather Station Data: Clustering, Iteration 2 #
####################################################

sklearn.utils.check_random_state(1000)

Clus_dataSet = pdf[['xm', 'ym', 'Tx', 'Tm', 'Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# computing density based clusters
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels))

# printing few select columns
print(pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5))

# for outliers, cluster label is -1
print(set(labels))

##########################################################
# 7: Visualizing Clusters Based on Location, Iteration 2 #
##########################################################

plt.rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, # min longitudes and latitudes
            urcrnrlon=ulon, urcrnrlat=ulat) # max longitudes and latitudes

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
# my_map.shadedrelief() # choosing not to render this fancy layer

# creating a colored map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

# Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4])) if clust_number == -1 else colors[np.int(clust_number)]
    clust_set = pdf[pdf.Clus_Db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color =c, marker='o',
                    s=20, alpha=0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm)
        ceny=np.mean(clust_set.ym)
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

# plt.text(x,y,stn)
plt.show()


























# in order to display plot within window
# plt.show()
