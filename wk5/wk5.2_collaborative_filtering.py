#####################################
# Week 5.2: Collaborative Filtering #
#####################################

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

########################
# Importing Movie Data #
########################

# reading and storing movie title, and genres dataset
movies_df = pd.read_csv('ml-latest\movies.csv')

# reading and storing user ratings dataset
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

# printing data with year column
movies_df.head()

###################
# Dropping Genres #
###################

# dropping genres column from movie dataframe.
movies_df = movies_df.drop('genres', 1)
movies_df.head()

################################
# Ranking Dataframe: Dataset 2 #
################################

# dataframe 2: user ratings
ratings_df.head()

# dropping timestamp column from dataframe.
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()

###########################
# Collaborative Filtering #
###########################

# starting with a list of movies the user liked and rated

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

# filtering out movies on title: original data vs. user's list
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

# merging original data id w/ user movies, using title by default
inputMovies = pd.merge(inputId, inputMovies)

# dropping year from combined dataframe
inputMovies = inputMovies.drop('year', 1)

# printing dataframe with id, title, and ratings
inputMovies

###############################
# Grouping Ratings by User ID #
###############################

# input movies = user ratings + movie id -> using that to subset
# overall ratings
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()

# grouping rows by user id (contained in ratings dataframe)
userSubsetGroup = userSubset.groupby(['userId'])

# checking one of the users
userSubsetGroup.get_group(1130)

########################################
# Sorting Users' Ratings by Similarity #
########################################

# grouped ratings - not aggregated yet.
userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
userSubsetGroup[0:3]

# selecting in most closely correlated users
userSubsetGroup = userSubsetGroup[0:100]

##############################################################
# Saving the Pearson Coefficient for All users to Input User #
##############################################################

# creating empty shell
pearsonCorrelationDict = {}

# creating and filling correlations
for name, group in userSubsetGroup:

    # sorting input grouped ratings
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')

    # grabbing number of users
    nRatings = len(group)

    # grabbing ratings for in-common movies
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]

    # and storing in a list
    tempRatingList = temp_df['rating'].tolist()

    # setting user group to a list
    tempGroupList = group['rating'].tolist()

    # finally, calculating correlation coefficient b/w users
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(
                            tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(
                            tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList,
            tempGroupList)) - sum(tempRatingList)*sum(
            tempGroupList)/float(nRatings)

    # setting condition, if denom=0, corr=0
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

# printing correlation dictionary
pearsonCorrelationDict.items()

# casting dictionary to a dataframe
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict,
                                    orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print(pearsonDF.head())

#######################
# Capturing Top Users #
#######################

# top 50 similar users
topUsers=pearsonDF.sort_values(by='similarityIndex',
                                ascending=False)[0:50]
print(topUsers.head())

# merging similar users (by input movies) to their rated movies
topUsersRating=topUsers.merge(ratings_df, left_on='userId',
                                right_on='userId', how='inner')
print(topUsersRating.head())

# multiplying similarity index by users overall ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex'
                                    ]*topUsersRating['rating']

print(topUsersRating.head())

# grouping users again, by id, this time aggregating to sum
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[[
                'similarityIndex', 'weightedRating']]
# assigning to columns
tempTopUsersRating.columns = ['sum_similarityIndex',
                                'sum_weightedRating']
print(tempTopUsersRating.head())

# declaring data frame shell
recommendation_df = pd.DataFrame()

# casting weighted average to the dataframe
recommendation_df['weighted average recommendation score'
                    ] = tempTopUsersRating['sum_weightedRating'
                    ]/tempTopUsersRating['sum_similarityIndex']

recommendation_df['movieId'] = tempTopUsersRating.index
print(recommendation_df.head())


# ready to sort and display results
recommendation_df = recommendation_df.sort_values(
    by='weighted average recommendation score', ascending=False)

print(recommendation_df.head(10))

# casting top 10 recommended movies to list based on user ratings & printing
print(movies_df.loc[movies_df['movieId'].isin(
                recommendation_df.head(10)['movieId'].tolist())])















# in order to display plot within window
# plt.show()
