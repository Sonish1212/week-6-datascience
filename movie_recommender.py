import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

columns_movies = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

movies = pd.read_csv("week-6/u.item", sep='|', names=columns_movies, encoding='latin-1')


columns_rating = ["user_id", "movie_id", "rating", "timestamp"]

rating = pd.read_csv("week-6/u.data", sep="\t", names=columns_rating)
print(rating.shape)

movie_counts = rating["movie_id"].value_counts()
min_rating = 10
popular_movies = movie_counts[movie_counts >= min_rating].index

rating = rating[rating["movie_id"].isin(popular_movies)]
rating = rating.drop("timestamp", axis=1)

user_counts = rating["user_id"].value_counts()

# Set a threshold (e.g., keep only users who rated at least 5 movies)
min_user_ratings = 5  
active_users = user_counts[user_counts >= min_user_ratings].index

# Filter dataset to keep only active users
rating = rating[rating["user_id"].isin(active_users)]
merged_data = pd.merge(rating, movies, on='movie_id')


print(rating.shape)

print(merged_data.describe())

print(merged_data.info())

# print(merged_data.isnull().sum())


merged_data = merged_data.drop('video_release_date', axis=1)
# print(merged_data.isnull().sum())

merged_data['IMDb_URL'].fillna('URL Not Available', inplace= True)
merged_data['release_date'].fillna('Date Unavilable', inplace= True)
print(merged_data.isnull().sum())


merged_data.boxplot()
plt.show()

# Surprise formatting
reader = Reader(rating_scale=(1,5))
data_surprise = Dataset.load_from_df(merged_data[["user_id", "movie_id","rating"]], reader)

trainset, testset = train_test_split(data_surprise, test_size=0.2, random_state=42)
#Initialize SVD model
svd = SVD()

#Train the model on the training set
svd.fit(trainset)

#predict the test set
predictions = svd.test(testset)

#Compute RMSE
rmse = accuracy.rmse(predictions)
print("Model RMSE: ", rmse)

# Get recommendation for a user

def recommend_movies(user_id, n=5):
    all_movies_id = merged_data["movie_id"].unique()
    
    # Get movies the user has already rated
    rated_movies = merged_data[merged_data["user_id"] == user_id]["movie_id"].tolist()

    # predict rating for all unseen movies
    predictions = [svd.predict(user_id, movie_id) for movie_id in all_movies_id if movie_id not in rated_movies]

    #Sort prediction by estimated rating
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    #Get movie titles
    recommended_movie_ids = [pred.iid for pred in recommendations]
    recommended_movies = movies[movies["movie_id"].isin(recommended_movie_ids)]

    return recommended_movies

print(recommend_movies(196, n=5))


