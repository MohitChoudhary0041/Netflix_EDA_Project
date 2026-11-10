
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load Netflix dataset
df = pd.read_csv("netflix_titles.csv")


# Step 1: Display First 5 Records

print("First 5 Records:")
print(df.head())


# Step 2: Dataset Information

print("\nDataset Information:")
print(df.info())


# Step 3: Dataset Shape

print("\nDataset Shape:")
print(df.shape)


# Step 4: Column Names

print("\nColumn Names:")
print(df.columns.tolist())


# Step 5: Missing Values Analysis

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nMissing Values Percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))


# Step 6: Duplicate Records Analysis

print("\nTotal Duplicate Records:")
print(df.duplicated().sum())


# Step 7: Netflix Content Distribution

print("\nMovies and TV Shows Distribution:")
print(df["type"].value_counts())


# Step 8: Movies vs TV Shows Visualization

content_counts = df["type"].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=content_counts.index, y=content_counts.values)

plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig("output/content_distribution.png")
plt.show()


# Step 9: Movies vs TV Shows Percentage Distribution

plt.figure(figsize=(7, 7))

df["type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage Distribution of Movies and TV Shows")
plt.ylabel("")

plt.tight_layout()
plt.savefig("output/content_percentage.png")
plt.show()


# Step 10: Movies Released Over the Years

movies = df[df["type"] == "Movie"]

movies_per_year = movies["release_year"].value_counts().sort_index()

plt.figure(figsize=(12, 6))

plt.plot(movies_per_year.index, movies_per_year.values)

plt.title("Movies Released Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Number of Movies")
plt.grid(True)

plt.tight_layout()
plt.savefig("output/movies_per_year.png")
plt.show()


# Step 11: TV Shows Released Over the Years

tv_shows = df[df["type"] == "TV Show"]

tv_per_year = tv_shows["release_year"].value_counts().sort_index()

plt.figure(figsize=(12, 6))

plt.bar(tv_per_year.index, tv_per_year.values)

plt.title("TV Shows Released Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Number of TV Shows")

plt.tight_layout()
plt.savefig("output/tv_shows_per_year.png")
plt.show()


# Step 12: Top 10 Countries Producing Netflix Content

country_data = df["country"].dropna()

country_counts = (
    country_data.str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12, 6))

country_counts.plot(kind="bar")

plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("output/top_10_countries.png")
plt.show()


# Step 13: Top 10 Genres on Netflix

genre_data = df["listed_in"].dropna()

genre_counts = (
    genre_data.str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12, 6))

genre_counts.plot(kind="bar", color="orange")

plt.title("Top 10 Genres on Netflix")
plt.xlabel("Genre")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("output/top_10_genres.png")
plt.show()


# Step 14: Netflix Content Ratings Distribution

rating_counts = df["rating"].dropna().value_counts()

plt.figure(figsize=(12, 6))

rating_counts.plot(kind="bar")

plt.title("Netflix Content Ratings Distribution")
plt.xlabel("Content Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("output/content_ratings.png")
plt.show()


# Step 15: Movie Duration Analysis

movies = df[df["type"] == "Movie"].copy()

movies["duration_minutes"] = (
    movies["duration"].str.extract(r"(\d+)").astype(float)
)

plt.figure(figsize=(10, 6))

plt.hist(movies["duration_minutes"].dropna(), bins=30)

plt.title("Distribution of Netflix Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")

plt.tight_layout()
plt.savefig("output/movie_duration.png")
plt.show()


# Step 16: TV Shows Seasons Analysis

tv_shows = df[df["type"] == "TV Show"].copy()

tv_shows["seasons"] = (
    tv_shows["duration"].str.extract(r"(\d+)").astype(float)
)

season_counts = tv_shows["seasons"].value_counts().sort_index()

plt.figure(figsize=(10, 6))

season_counts.plot(kind="bar")

plt.title("Distribution of TV Show Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")

plt.tight_layout()
plt.savefig("output/tv_show_seasons.png")
plt.show()


# Project Completion Message

print("\n===================================")
print("NETFLIX EDA PROJECT COMPLETED!")
print("===================================")
print("Total Records:", len(df))
print("Total Columns:", len(df.columns))
print("Total Visualizations: 9")
print("All charts saved in the output folder.")
