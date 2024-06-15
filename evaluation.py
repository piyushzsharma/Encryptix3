from surprise import accuracy

def evaluate_model(algo, testset):
    """
    Evaluate the performance of the recommendation model using RMSE.
    
    Args:
    algo: The trained KNN algorithm.
    testset: The test dataset.
    
    Returns:
    rmse: The Root Mean Squared Error of the model on the test dataset.
    """
    # Test the algorithm on the testset
    predictions = algo.test(testset)

    # Calculate and return the RMSE (Root Mean Squared Error)
    rmse = accuracy.rmse(predictions)
    
    return rmse
