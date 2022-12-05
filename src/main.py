import numpy as np


def r2_score(actual, predicted):
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    
    e1 = np.sum((actual - predicted) ** 2)
    e2 = np.sum((actual - np.mean(actual)) ** 2)
    
    return 1 - e1 / e2



    