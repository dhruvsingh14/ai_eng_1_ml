##############################
# Week 1.1: Data Acquisition #
##############################

# importing libraries
import pandas as pd

#############
# Read Data #
#############
# read url
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)

# print head
#print("first 5 rows of the dataframe")
df.head(5)

##############
# Question 1 #
##############
# bottom ten rows of dataframe
df.tail(10)

###############
# Add Headers #
###############
# create headers list
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
        "drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type",
        "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
        "peak-rpm","city-mpg","highway-mpg","price"]
#print("headers\n", headers)
# print(type(headers))
# using list to assign column names
df.columns = headers
df.head(10)

# dropping missing values
df.dropna(subset=["price"], axis=0)

##############
# Question 2 #
##############
# name of columns of the dataframe
df.columns

# saving dataset
df.to_csv("automobile.csv", index=False)

# prints data types by column
df.dtypes

############
# Describe #
############
# running basic general descriptive statistics
df.describe(include = "all")

##############
# Question 3 #
##############
# describing 2 select columns -
df[['length', 'compression-ratio']].describe()

# printing df info
print(df.info)



























# in order to display plot within window
# plt.show()
