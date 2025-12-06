## CSPB 3115 Final Project Perez
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

## DATA LOADING
# loading raw music/user listening data from csv files
df_musicinfo = pd.read_csv("data/Music Info.csv")
df_userinfo = pd.read_csv("data/User Listening History.csv")
print(df_musicinfo.columns)
print(df_userinfo.columns)

# removing duplicate songs and unnecessary columns to clean the dataset
songs_df = df_musicinfo.drop_duplicates(subset=['track_id'])
songs_df = songs_df.drop(columns=['spotify_preview_url', 'spotify_id', 'year','key','time_signature'])
songs_df = songs_df.reset_index(drop=True)
# renaming columns to match our naming convention throughout the project
songs_df = songs_df.rename(
    columns={
        'track_id': 'song_id',
        'artist': 'artist_name'
    }
)
print('SONG INFO')
print(songs_df.head())

# creating a clean list of unique users from the listening history
users_df = df_userinfo[['user_id']].drop_duplicates().reset_index(drop=True)
print('USER INFO')
print(users_df.head())

# preparing the interactions dataset showing which users listened to which songs
interactions_df = df_userinfo.rename(columns={'track_id': 'song_id'})
interactions_df = interactions_df.dropna(subset=['user_id', 'song_id']).reset_index(drop=True)
print('INTERACTIONS')
print(interactions_df.head())

# aggregating song data at the artist level to get average audio features per artist
artist_info = songs_df.groupby('artist_name').agg({
    'song_id': 'count',
    'tempo': 'mean',
    'loudness': 'mean',
    'energy': 'mean'
}).reset_index()
artist_info = artist_info.rename(columns={'song_id': 'num_songs'})

# merging interactions with song details to enrich each listening event
interactions_merged = interactions_df.merge(songs_df, on='song_id', how='left')

# creating the final dataset by adding artist-level features to each interaction
final_df = interactions_merged.merge(artist_info, on='artist_name', how='left', suffixes=('', '_artist'))
print('FINAL DATA FRAME')
print(final_df.head())

## DATA CLEANING
# removing rows with missing user or song ids since these are essential
final_df = final_df.dropna(subset=['user_id', 'song_id'])
# filling missing numeric values with column averages to avoid losing data
num_cols = final_df.select_dtypes(include='number').columns
final_df[num_cols] = final_df[num_cols].fillna(final_df[num_cols].mean())
final_df = final_df.reset_index(drop=True)

### final csv files that will be used in this project - taken out because it takes too long to run so ignore
# saving cleaned datasets for future use or sharing with teammates
# songs_df.to_csv("data/songs_clean.csv", index=False)
# users_df.to_csv("data/users_clean.csv", index=False)
# interactions_df.to_csv("data/interactions_clean.csv", index=False)
# artist_info.to_csv("data/artist_info.csv", index=False)
# final_df.to_csv("data/final_dataset.csv", index=False)

## DATA ANALYSIS
# printing basic statistics to understand the dataset size and scope
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"Total interactions: {len(final_df):,}")
print(f"Unique users: {final_df['user_id'].nunique():,}")
print(f"Unique songs: {final_df['song_id'].nunique():,}")
print(f"Unique artists: {final_df['artist_name'].nunique():,}")
print(f"Unique genres: {final_df['genre'].nunique():,}")

## PLAYCOUNT STATISTICS
# examining playcount distribution to understand listening patterns
print("\n" + "="*60)
print("PLAYCOUNT STATISTICS")
print("="*60)
print(final_df["playcount"].describe())
print(f"\nMedian playcount: {final_df['playcount'].median():.2f}")
print(f"95th percentile: {final_df['playcount'].quantile(0.95):.2f}")
print(f"99th percentile: {final_df['playcount'].quantile(0.99):.2f}")

## Basic dataset info
# showing dataset dimensions to verify data loaded correctly
print("\n" + "="*60)
print("DATASET SHAPE")
print("="*60)
print(f"Number of rows: {final_df.shape[0]:,}")
print(f"Number of columns: {final_df.shape[1]}")

print("\n" + "="*60)
print("FIRST FEW ROWS")
print("="*60)
print(final_df.head())

# checking for missing values to ensure data quality
print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
print(final_df.isna().sum())

## IMPROVED PLAYCOUNT VISUALIZATIONS
# creating multiple views of playcount distribution since it has outliers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# showing playcount without extreme outliers for better visibility
q95 = final_df["playcount"].quantile(0.95)
axes[0, 0].hist(final_df[final_df["playcount"] <= q95]["playcount"], bins=50, edgecolor='black')
axes[0, 0].set_title(f"Playcount Distribution (capped at 95th percentile: {q95:.0f})")
axes[0, 0].set_xlabel("Playcount")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].grid(alpha=0.3)

# using log scale on yaxis to see the full range including outliers
axes[0, 1].hist(final_df["playcount"], bins=50, edgecolor='black')
axes[0, 1].set_yscale('log')
axes[0, 1].set_title("Playcount Distribution (Log Scale Y-axis)")
axes[0, 1].set_xlabel("Playcount")
axes[0, 1].set_ylabel("Frequency (log)")
axes[0, 1].grid(alpha=0.3)

# transforming playcount values to reduce skewness and reveal patterns
axes[1, 0].hist(np.log1p(final_df["playcount"]), bins=50, edgecolor='black')
axes[1, 0].set_title("Distribution of Log(Playcount + 1)")
axes[1, 0].set_xlabel("Log(Playcount + 1)")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].grid(alpha=0.3)

# visualizing outliers using a box plot to identify extreme values
axes[1, 1].boxplot(final_df["playcount"])
axes[1, 1].set_title("Playcount Box Plot")
axes[1, 1].set_ylabel("Playcount")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

## TOP GENRES
# identifying the most common music genres in the dataset
plt.figure(figsize=(12, 6))
final_df["genre"].value_counts().head(10).plot(kind="bar", edgecolor='black')
plt.title("Top 10 Genres in the Dataset", fontsize=14, fontweight='bold')
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

## TOP ARTISTS
# finding which artists have the most songs in our dataset
plt.figure(figsize=(12, 6))
songs_df["artist_name"].value_counts().head(10).plot(kind="bar", edgecolor='black')
plt.title("Top 10 Artists by Number of Songs", fontsize=14, fontweight='bold')
plt.xlabel("Artist Name")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

## CORRELATION HEATMAP
# visualizing relationships between numeric features to find patterns
plt.figure(figsize=(12, 8))
corr_matrix = final_df.select_dtypes(include="number").corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap for Numeric Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

## USER ACTIVITY ANALYSIS
# calculating total listening activity for each user
user_activity = final_df.groupby("user_id")["playcount"].sum().reset_index()
user_activity.columns = ["user_id", "total_plays"]

# examining user engagement levels across the platform
print("\n" + "="*60)
print("USER ACTIVITY STATISTICS")
print("="*60)
print(user_activity["total_plays"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# showing user activity without extreme outliers for clarity
q95_user = user_activity["total_plays"].quantile(0.95)
axes[0].hist(user_activity[user_activity["total_plays"] <= q95_user]["total_plays"],
             bins=50, edgecolor='black')
axes[0].set_title(f"User Activity Distribution (capped at 95th percentile: {q95_user:.0f})")
axes[0].set_xlabel("Total Plays per User")
axes[0].set_ylabel("Number of Users")
axes[0].grid(alpha=0.3)

# using log scale to show the full range of user activity levels
axes[1].hist(user_activity["total_plays"], bins=50, edgecolor='black')
axes[1].set_xscale('log')
axes[1].set_title("User Activity Distribution (Log)")
axes[1].set_xlabel("Total Plays per User (log)")
axes[1].set_ylabel("Number of Users")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

## PLAYCOUNT BY GENRE
# comparing playcount distributions across popular genres
top_genres = final_df["genre"].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=final_df[final_df["genre"].isin(top_genres)],
            x="genre", y="playcount")
plt.yscale('log')
plt.xticks(rotation=45, ha='right')
plt.title("Playcount Distribution by Top 10 Genres (Log Scale)", fontsize=14, fontweight='bold')
plt.xlabel("Genre")
plt.ylabel("Playcount (log scale)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

## UNIQUE SONGS PER USER
# measuring music diversity by counting unique songs each user listened to
songs_per_user = final_df.groupby("user_id")["song_id"].nunique().reset_index()
songs_per_user.columns = ["user_id", "unique_songs"]

plt.figure(figsize=(10, 6))
plt.hist(songs_per_user["unique_songs"], bins=50, edgecolor='black')
plt.title("Distribution of Unique Songs per User", fontsize=14, fontweight='bold')
plt.xlabel("Number of Unique Songs")
plt.ylabel("Number of Users")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

##  TOP SONGS BY TOTAL PLAYCOUNT
# finding the most popular songs based on total plays across all users
if "song_name" in final_df.columns:
    song_col = "song_name"
elif "track_name" in final_df.columns:
    song_col = "track_name"
else:
    song_col = "song_id"

top_songs = final_df.groupby(song_col)["playcount"].sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(12, 8))
top_songs.plot(kind="barh", edgecolor='black')
plt.title(f"Top 20 Songs by Total Playcount", fontsize=14, fontweight='bold')
plt.xlabel("Total Playcount")
plt.ylabel(song_col.replace("_", " ").title())
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

## ADDITIONAL INSIGHTS
# summarizing key metrics to understand overall dataset characteristics
print("\n" + "="*60)
print("ADDITIONAL INSIGHTS")
print("="*60)
print(f"Average plays per interaction: {final_df['playcount'].mean():.2f}")
print(f"Median plays per interaction: {final_df['playcount'].median():.2f}")
print(f"Average songs per user: {songs_per_user['unique_songs'].mean():.2f}")
print(f"Average interactions per user: {len(final_df) / final_df['user_id'].nunique():.2f}")
print(f"Average interactions per song: {len(final_df) / final_df['song_id'].nunique():.2f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

## FEATURE ENGINEERING
# creating new columns that help our models learn better patterns

## USER-LEVEL FEATURES
# aggregating data per user to capture their overall listening behavior
user_features = final_df.groupby("user_id").agg({
    "playcount": "sum",
    "song_id": "nunique"
}).reset_index().rename(columns={
    "playcount": "total_user_plays",
    "song_id": "unique_user_songs"
})

## SONG-LEVEL FEATURES
# aggregating data per song to measure its popularity and reach
song_features = final_df.groupby("song_id").agg({
    "playcount": "sum",
    "user_id": "nunique"
}).reset_index().rename(columns={
    "playcount": "total_song_plays",
    "user_id": "unique_song_users"
})

#  labeling songs as popular if they exceed average playcount
avg_playcount = song_features["total_song_plays"].mean()
song_features["popular"] = song_features["total_song_plays"] > avg_playcount

## ARTIST-LEVEL FEATURES
# aggregating data per artist to understand their overall presence
artist_features = final_df.groupby("artist_name").agg({
    "playcount": "sum",
    "song_id": "nunique"
}).reset_index().rename(columns={
    "playcount": "total_artist_plays",
    "song_id": "artist_song_count"
})

# saving engineered features for use in modeling
user_features.to_csv("data/user_features.csv", index=False)
song_features.to_csv("data/song_features.csv", index=False)
artist_features.to_csv("data/artist_features.csv", index=False)
print("Feature engineering complete")

## MODELING SONG POPULARITY
# preparing features and targt variable for classification
X = song_features[["total_song_plays", "unique_song_users"]]
y = song_features["popular"]

# splitting data into training and testing sets to evaluate model performance
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# creating a random forest model with tuned parameters
model = RandomForestClassifier(
    n_estimators=500,  # using 500 trees for better predictions
    max_depth=10,  # limiting tree depth to prevent overfitting
    min_samples_split=5,  # requiring at least 5 samples to split a node
    random_state=42,
    class_weight="balanced"  # handling class imbalance in popularity labels
)

# training the model on the training data
model.fit(x_train, y_train)

# evaluating model performance on unseen test data
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# performing cross-validation to verify model stability
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Average 5-fold CV score: {cv_scores.mean():.3f}")

# visualizing which features matter most for predictions
importances = model.feature_importances_
plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=["total_song_plays", "unique_song_users"])
plt.title("Feature Importance")
plt.show()

# saving the trained model for future use or to share to orthers
joblib.dump(model, "data/popularity_model.pkl")
print("Song popularity model complete")