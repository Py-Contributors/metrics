import numpy as np
import sys
sys.path.append('.')

from src.utils import preprocessing_and_sanitization
# classifications metrics

def accuracy_score(actual: np.array, predicted: np.array) -> float:
    """ Accuracy classification score.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    
    Returns
    -------
        score: float
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)

    return np.mean(actual == predicted)


def confusion_metrics(actual: np.array, predicted: np.array) -> np.array:
    """ Confusion metrics.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    
    Returns
    -------
        metrics: array-like, shape = [n_classes, n_classes]
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)

    classes = np.unique(actual)
    metrics = np.zeros((classes.size, classes.size))

    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            metrics[i, j] = np.sum((actual == c) & (predicted == d))

    return metrics


def precision(actual: np.array, predicted: np.array) -> float:
    """ Precision classification score.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    
    Returns
    -------
        score: array-like, shape = [n_classes]
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)

    classes = np.unique(actual)
    metrics = confusion_metrics(actual, predicted)
    score = np.zeros(classes.size)

    for i, c in enumerate(classes):
        score[i] = metrics[i, i] / np.sum(metrics[:, i])
    
    return np.mean(score)


def recall(actual: np.array, predicted: np.array) -> float:
    """ Recall classification score.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    
    Returns
    -------
        score: array-like, shape = [n_classes]
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)

    classes = np.unique(actual)
    metrics = confusion_metrics(actual, predicted)
    score = np.zeros(classes.size)

    for i, c in enumerate(classes):
        score[i] = metrics[i, i] / np.sum(metrics[i, :])
    
    return np.mean(score)

def f1_score(actual: np.array, predicted: np.array) -> float:
    
    actual, predicted = preprocessing_and_sanitization(actual, predicted)
    
    precision_ = precision(actual, predicted)
    recall_ = recall(actual, predicted)

    return 2 * (precision_ * recall_) / (precision_ + recall_)


def auc(actual: np.array, predicted: np.array) -> float:
    """ Area under the curve.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    
    Returns
    -------
        score: float
    """
    actual, predicted = preprocessing_and_sanitization(actual, predicted)

    classes = np.unique(actual)
    if classes.size != 2:
        raise ValueError("auc only works for binary classification")

    return np.mean((actual == classes[0]) & (predicted == classes[1]))


