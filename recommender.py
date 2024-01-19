#importing modules
import pandas as pd
import numpy as np
from surprise import KNNBasic, accuracy


# recommender function
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
modelling_data = pd.merge(movies, ratings, on = "movieId")
#dropping timestamp 
modelling_data.drop(columns= "timestamp", axis= 1, inplace= True)


def recommender_system(userId, n, df, model):
    #unique user
    user_Id = userId
    #N number of recommendations
    N = n
    #list of movies not rated by the user
    to_recommend  = set(df["movieId"].unique()) - set(df[df["userId"] == user_Id]["movieId"].unique())
    #getting predictions for movies to recommend
    preds_to_user = [model.predict(user_Id, movieId) for  movieId in to_recommend]
    #get top N recommendation
    top_N_recommendations = sorted(preds_to_user, key=lambda x: x.est, reverse=True)[:N]
    # Display the top N recommendations
    for recommendation in top_N_recommendations:
        movie_info = modelling_data[modelling_data['movieId'] == recommendation.iid]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
            print(f"MovieId: {recommendation.iid}, Title: {title}, Genres: {genres}, Estimated Rating: {recommendation.est}")


#neural networks recommender function
def neural_recommender(user_id, n, df, svd_model, neural_network_model):
    # Get movies not rated by the user
    to_recommend = set(movies['movieId'].unique()) - set(movies[movies['userId'] == user_id]['movieId'].unique())

    # Get SVD predictions for movies to recommend
    svd_preds = [svd_model.predict(user_id, movie_id).est for movie_id in to_recommend]

    # Create a DataFrame with SVD predictions
    df_with_svd = pd.DataFrame({'userId': [user_id] * len(to_recommend), 'movieId': list(to_recommend), 'svd_preds': svd_preds})

    # Use the neural network to predict ratings
    nn_preds = neural_network_model.predict([df_with_svd['userId'], df_with_svd['movieId'], df_with_svd['svd_preds']])

    # Combine movie IDs with their predicted ratings
    recommendations = pd.DataFrame({'movieId': list(to_recommend), 'predicted_rating': nn_preds.flatten()})

    # Get the top N recommendations
    top_recommendations = recommendations.nlargest(n, 'predicted_rating')

    # Merge with movie information to get titles and genres
    top_recommendations = pd.merge(top_recommendations, df[['movieId', 'title', 'genres']], on='movieId', how='left')

    return top_recommendations[['movieId', 'title', 'genres', 'predicted_rating']]


#KNN caller function
def knn_caller(user_id, n, model):
    user_id = user_id
    N = n
    # Get a list of all movie IDs in your dataset
    all_movie_ids = np.unique(modelling_data['movieId'])

    # Create a list to store predicted ratings for unrated movies
    predicted_ratings = []

    # Predict ratings for the user on unrated movies
    for movie_id in all_movie_ids:
        # Check if the user has already rated the movie
        if not ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)].empty:
            predicted_rating = model.predict(user_id, movie_id)
            predicted_ratings.append((movie_id, predicted_rating.est))
            #Sort the predicted ratings in descending order
            predicted_ratings.sort(key=lambda x: x[1], reverse=True)

            # Get the top 5 movie recommendations
            top_5_recommendations = predicted_ratings[:N]

            # Display the top 5 recommended movies
            for movie_id, predicted_rating in top_5_recommendations:
                # Check if the condition results in a non-empty DataFrame
                if not movies[movies['movieId'] == movie_id].empty:
                    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
                    rounded_rating = round(predicted_rating, 1)
                    print(f"Movie: {movie_title}, Predicted Rating: {rounded_rating}")
                else:
                    print(f"Movie with ID {movie_id} not found in movies_df")
                    
                    


def knn_caller(user_id, knn_model):
    user_id = user_id
    # Get a list of all movie IDs in your dataset
    all_movie_ids = np.unique(modelling_data['movieId'])

    # Create a list to store predicted ratings for unrated movies
    predicted_ratings = []

    # Predict ratings for the user on unrated movies
    for movie_id in all_movie_ids:
        # Check if the user has already rated the movie
        if not ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)].empty:
            predicted_rating = knn_model.predict(user_id, movie_id)
            predicted_ratings.append((movie_id, predicted_rating.est))
            #Sort the predicted ratings in descending order
            predicted_ratings.sort(key=lambda x: x[1], reverse=True)

            # Get the top 5 movie recommendations
            top_5_recommendations = predicted_ratings[:5]

            # Display the top 5 recommended movies
            for movie_id, predicted_rating in top_5_recommendations:
                # Check if the condition results in a non-empty DataFrame
                if not movies[movies['movieId'] == movie_id].empty:
                    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
                    rounded_rating = round(predicted_rating, 1)
                    print(f"Movie: {movie_title}, Predicted Rating: {rounded_rating}")
                else:
                    print(f"Movie with ID {movie_id} not found in movies_df")
                    
                    
                    
