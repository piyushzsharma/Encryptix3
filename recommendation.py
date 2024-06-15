from surprise import KNNBasic

def train_model(trainset):
    """
    Train a KNN-based collaborative filtering model.
    
    Args:
    trainset: The training dataset.
    
    Returns:
    algo: The trained KNN algorithm.
    """
    # Use user-based collaborative filtering with cosine similarity
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }

    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    
    return algo

def get_top_n_recommendations(algo, trainset, user_id, n=10):
    """
    Get top N movie recommendations for a specific user.
    
    Args:
    algo: The trained KNN algorithm.
    trainset: The training dataset.
    user_id: The ID of the user for whom to generate recommendations.
    n: The number of recommendations to generate.
    
    Returns:
    top_n_recommendations: A list of top N recommendations with estimated ratings.
    """
    # Get a list of all movie IDs
    all_movie_ids = [movie_id for movie_id in range(1, 1683)]  # Movie IDs range from 1 to 1682 in MovieLens 100k

    # Get the list of movies the user has already rated
    inner_user_id = trainset.to_inner_uid(user_id)
    rated_movies = {iid for (uid, iid, true_r) in trainset.ur[inner_user_id]}

    # Predict ratings for all movies the user hasn't rated yet
    predictions = [algo.predict(user_id, iid) for iid in all_movie_ids if iid not in rated_movies]

    # Sort the predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top n recommendations
    top_n_recommendations = predictions[:n]

    return top_n_recommendations
