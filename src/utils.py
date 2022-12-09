""" Utility functions."""

import numpy as np
from typing import Tuple


def preprocessing_and_sanitization(actual: np.array, predicted: np.array) -> Tuple[np.array, np.array]:
    """ Preprocessing and sanitization data.
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    
    Returns
    -------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    # check if arrays are 1D
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("actual and predicted must be 1D arrays")
    
    # check if arrays are not empty
    if actual.size == 0 or predicted.size == 0:
        raise ValueError("actual and predicted must not be empty")

    # check if arrays have the same shape
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

    return (actual, predicted)
