import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # Compute the dot product of the feature matrix X and the weight vector
        predictions = np.dot(X, weights)
        
        # Round the resulting array to 5 decimal places
        return np.round(predictions, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Calculate the Mean Squared Error: Mean of (y_pred - y_true)^2
        mse = np.mean((model_prediction - ground_truth) ** 2)
        
        # Return the final error rounded to 5 decimal places
        return float(round(mse, 5))

