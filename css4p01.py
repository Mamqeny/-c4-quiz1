# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:39:12 2024

@author: Mamqe
"""
import pandas as pd
import numpy as np

#Reading the dataset from the local repo
df = pd.read_csv("movie_dataset.csv")

#identifying null values
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

#Replacing null values with 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

#Finding the highest rated movie
highest_rated_movie = df.sort_values(by='Rating', ascending=False).iloc[0]

print("Highest Rated Movie:")
print(highest_rated_movie[['Title', 'Rating']])

#Renaming colum 'Revenue (Millions)' to Revenue 
df.rename(columns={'Revenue (Millions)': 'Revenue'}, inplace=True)

# Using Revenue as a column name to calculate the average revenue of all the movies
average_revenue = df['Revenue'].mean()

print("Average Revenue of All Movies: ${:,.2f}".format(average_revenue))

#finding Average Revenue of Movies from 2015 to 2017
# Convert 'Release Year' to numeric if it's not already
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Filter movies released between 2015 and 2017
filtered_movies = df[(df['Year'] >= 2015) & (df['Year'] <= 2017)]

# Calculate the average revenue for the filtered movies
average_revenue_2015_to_2017 = filtered_movies['Revenue'].mean()

print("Average Revenue of Movies from 2015 to 2017: ${:,.2f}".format(average_revenue_2015_to_2017))

# Count the number of movies released in 2016
# Convert 'Release Year' to numeric if it's not already
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

movies_2016_count = (df['Year'] == 2016).sum()
print("Number of Movies Released in 2016:", movies_2016_count)

# Count the number of movies directed by Christopher Nolan
nolan_movies_count = (df['Director'] == 'Christopher Nolan').sum()

print("Number of Movies Directed by Christopher Nolan:", nolan_movies_count)

# Count the number of movies with a rating of at least 8.0
high_rating_movies_count = (df['Rating'] >= 8.0).sum()

print("Number of Movies with a Rating of at Least 8.0:", high_rating_movies_count)

# Filter movies directed by Christopher Nolan
nolan_movies = df[df['Director'] == 'Christopher Nolan']

# Calculate the median rating for movies directed by Christopher Nolan
median_rating_nolan_movies = nolan_movies['Rating'].median()

print("Median Rating of Movies Directed by Christopher Nolan:", median_rating_nolan_movies)

# Convert 'Release Year' to numeric if it's not already
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Group by 'Release Year' and calculate the average rating for each year
average_rating_by_year = df.groupby('Year')['Rating'].mean()

# Find the year with the highest average rating
highest_avg_rating_year = average_rating_by_year.idxmax()
highest_avg_rating = average_rating_by_year.max()

print("Year with the Highest Average Rating:", int(highest_avg_rating_year))

#Percentage Increase in Number of Movies from 2006 to 2016
# Convert 'Release Year' to numeric if it's not already
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Count the number of movies for 2006 and 2016
movies_2006_count = (df['Year'] == 2006).sum()
movies_2016_count = (df['Year'] == 2016).sum()

# Calculate the percentage increase
percentage_increase = ((movies_2016_count - movies_2006_count) / movies_2006_count) * 100

print("Number of Movies in 2006:", movies_2006_count)
print("Number of Movies in 2016:", movies_2016_count)
print("Percentage Increase in Number of Movies from 2006 to 2016: {:.2f}%".format(percentage_increase))

# Create a new DataFrame to store individual actors
individual_actors = df['Actors'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).rename('Actor')

# Find the most common actor
most_common_actor = individual_actors.value_counts().idxmax()

print("Most Common Actor in All Movies:", most_common_actor)

# Create a new DataFrame to store individual genres
individual_genres = df['Genre'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True).rename('Genre')

# Count the number of unique genres
unique_genres_count = individual_genres.nunique()

print("Number of Unique Genres in the Dataset:", unique_genres_count)

#Correlation analysis of the numerical values
numerical_features = df.select_dtypes(include=['int64', 'float64'])

import seaborn as sns
import matplotlib.pyplot as plt

# Selecting numerical features for correlation analysis
numerical_features = df.select_dtypes(include=['int64', 'float64'])

# Calculate correlation matrix
correlation_matrix = numerical_features.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
