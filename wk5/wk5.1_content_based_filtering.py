#####################################
# Week 5.1: Content-Based Filtering #
#####################################

# importing libraries
import pandas as pd
from math import sqrt
import numpy as np

import matplotlib.pyplot as plt

########################
# Importing Movie Data #
########################

import wget
# import zipfile

# downloading data
# url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip'
# wget.download(url, 'moviedataset.zip')

# Note: Zipfile module seems to be acting up
# may just have to unzip by hand for now.

# reading and storing movie dataset
movies_df = pd.read_csv('ml-latest\movies.csv')

# reading and storing users dataset
ratings_df = pd.read_csv('ml-latest\\ratings.csv')

# printing movies dataset
movies_df.head()


####################
# Parsing out Year #
####################

# extracting date from movies listing
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)

# removing parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)

# removing year from title column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

#  applying strip function to trim whitespaces
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df.head()


####################
# Splitting Genres #
####################

# splitting genres on pipe characters
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()

#########################################
# One-Hot Encoding for Categorical Data #
#########################################

# making a copy of the original dataframe, using copy method for dataframes.
moviesWithGenres_df = movies_df.copy()

# generating columns for each genre - basically converting long to wide
# assigning genre columns binary value of 1
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

# populating empty cells w/ NaNs
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

################################
# Ranking Dataframe: Dataset 2 #
################################

# new, dataframe 2: supplementary
ratings_df.head()

# dropping timestamp column from dataframe.
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()

########################################
# Content-Based Recommendation Systems #
########################################

# super interesting
# starting with a list of movies the user liked and rated
# can extend this list using the same format

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
        ]

# casting user movie ratings to a dataframe
inputMovies = pd.DataFrame(userInput)
inputMovies


##########################################
# Filtering overall data using User data #
##########################################

# searching for users movies in the original dataframe
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

# combining original dataframe with overall ratings one
inputMovies = pd.merge(inputId, inputMovies)

# dropping year and genre from combined dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

# printing dataframe with id, title, and ratings
inputMovies

# additionally, subsetting wide-genres dataframe to rows w/ user's films
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies

###################
# Resetting index #
###################

# resetting index
userMovies = userMovies.reset_index(drop=True)

# only keeping binary genre columns as features
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)


#############################
# User Preferences Analysis #
#############################

# input: user's movies + binary genre columns + ratings
print(inputMovies['rating'])

# using ratings as weights: running dot product
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

# and printing weighted table
print(userProfile)

######################
# Binary Genre Table #
######################

# Grabbing binary genre columns
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

# dropping extraneous columns
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head())

# assessing dimensions
print(genreTable.shape)


############################################
# Weighted Genre Table: Using User Profile #
############################################

# multiplying all movie genres by user weights / ratings, and taking weighted average for each column

# bc there's multiple tags, it's possible to develop a more layered understanding

# and subsequently better recommendations

recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
print(recommendationTable_df.head())

# sorting recs by movie id
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
print(recommendationTable_df.head())

# final recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])
































# in order to display plot within window
# plt.show()
