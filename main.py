import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from recommendation import train_model, get_top_n_recommendations
from evaluation import evaluate_model

def main():
    # Load the MovieLens dataset
    data = Dataset.load_builtin('ml-100k')

    # Split the data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train the model
    algo = train_model(trainset)

    # Evaluate the model
    rmse = evaluate_model(algo, testset)
    print(f"RMSE: {rmse}")

    # Example usage: Get recommendations for a specific user
    user_id = str(196)  # User ID should be in string format for the MovieLens dataset
    recommendations = get_top_n_recommendations(algo, trainset, user_id, n=5)

    print(f"Top 5 movie recommendations for user {user_id}:")
    for prediction in recommendations:
        print(f"Movie ID: {prediction.iid}, Estimated Rating: {prediction.est}")

if __name__ == "__main__":
    main()
