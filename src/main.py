import numpy as np


def r2_score(actual: np.array, predicted: np.array) -> float:
    """ R^2 (coefficient of determination) regression score function. 
    actually, it's the same as mean_squared_error, but it's more intuitive.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    
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
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    
    return np.mean((actual - predicted) ** 2)


def mean_absolute_error(actual: np.array, predicted: np.array) -> float:
    """ Mean absolute error regression loss.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    
    return np.mean(np.abs(actual - predicted))


def root_mean_squared_error(actual: np.array, predicted: np.array) -> float:
    """ Root mean squared error regression loss.
    
    Parameters
    ----------
        actual: array-like, shape = [n_samples]
        predicted: array-like, shape = [n_samples]
    """
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    
    return np.sqrt(np.mean((actual - predicted) ** 2))


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
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

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
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

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
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

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
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

    classes = np.unique(actual)
    metrics = confusion_metrics(actual, predicted)
    score = np.zeros(classes.size)

    for i, c in enumerate(classes):
        score[i] = metrics[i, i] / np.sum(metrics[i, :])
    
    return np.mean(score)

def f1_score(actual: np.array, predicted: np.array) -> float:
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)
    
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    
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
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

    classes = np.unique(actual)
    if classes.size != 2:
        raise ValueError("auc only works for binary classification")

    return np.mean((actual == classes[0]) & (predicted == classes[1]))    
