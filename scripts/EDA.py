import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Load raw and cleaned datasets
raw_df = pd.read_csv("C:/User/waghr/Desktop/Movie-Genre-Classification-Sys/data/TMDB_movie_dataset_v11.csv")
clean_df = pd.read_csv("C:/Users/waghr/Desktop/Movie-Genre-Classification-Sys/data/strictly_balanced_top10_cleaned.csv")

# Basic summary
raw_summary = raw_df.describe(include='all')
clean_summary = clean_df.describe(include='all')

# Missing values
raw_missing = raw_df.isnull().sum().sort_values(ascending=False)
clean_missing = clean_df.isnull().sum().sort_values(ascending=False)

# Genre distribution from cleaned data
clean_df['genres'] = clean_df['genres'].apply(ast.literal_eval)
genre_counts = clean_df.explode('genres')['genres'].value_counts()

# Plot genre distribution
plt.figure(figsize=(10,6))
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title("🎬 Genre Distribution (Top 10 Balanced Cleaned Dataset)")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Overview length distribution
clean_df['overview_length'] = clean_df['overview'].astype(str).apply(lambda x: len(x.split()))
raw_df['overview_length'] = raw_df['overview'].astype(str).apply(lambda x: len(x.split()))

# Overview length histogram
plt.figure(figsize=(12,5))
sns.histplot(raw_df['overview_length'], kde=True, bins=50, color="skyblue", label="Raw")
sns.histplot(clean_df['overview_length'], kde=True, bins=50, color="orange", label="Cleaned")
plt.legend()
plt.title("📏 Overview Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

import ace_tools as tools; tools.display_dataframe_to_user(name="Raw Dataset Missing Values", dataframe=raw_missing.to_frame("Missing Count"))
tools.display_dataframe_to_user(name="Cleaned Dataset Missing Values", dataframe=clean_missing.to_frame("Missing Count"))
tools.display_dataframe_to_user(name="Raw Dataset Summary", dataframe=raw_summary)
tools.display_dataframe_to_user(name="Cleaned Dataset Summary", dataframe=clean_summary)
