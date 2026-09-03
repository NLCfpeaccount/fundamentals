import numpy as np
from numpy.typing import NDArray

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # Define epsilon to avoid log(0) and log(1) which lead to infinity/NaN
        eps = 1e-7
        
        # Clip y_pred values to be strictly between [eps, 1 - eps]
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Calculate BCE loss per sample, then take the average over all samples
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return float(round(loss, 4))

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # Define epsilon to avoid log(0)
        eps = 1e-7
        
        # Clip y_pred values to be strictly between [eps, 1]
        y_pred = np.clip(y_pred, eps, 1.0)
        
        # Sum log-probabilities along the classes (axis=-1) for one-hot encoded labels, 
        # then take the average across all samples in the batch
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
        
        return float(round(loss, 4))

