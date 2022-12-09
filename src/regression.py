import numpy as np
from typing import Tuple
import sys
sys.path.append('.')
from src.utils import preprocessing_and_sanitization


def r2_score(actual: np.array, predicted: np.array) -> float:
    """ R^2 (coefficient of determination) regression score function. 
    actually, it's the same as mean_squared_error, but it's more intuitive.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)
    
    e1 = np.sum((actual - predicted) ** 2)
    e2 = np.sum((actual - np.mean(actual)) ** 2)
    
    return 1 - e1 / e2


def mean_squared_error(actual: np.array, predicted: np.array) -> float:
    """ Mean squared error regression loss.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)
    
    return np.mean((actual - predicted) ** 2)


def mean_absolute_error(actual: np.array, predicted: np.array) -> float:
    """ Mean absolute error regression loss.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)
    
    return np.mean(np.abs(actual - predicted))


def root_mean_squared_error(actual: np.array, predicted: np.array) -> float:
    """ Root mean squared error regression loss.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)
    
    return np.sqrt(np.mean((actual - predicted) ** 2))


