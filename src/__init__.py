# import all the modules in the src folder
# so that they can be imported from the src folder
# instead of the src.regression or src.classification folder
# for example, instead of importing from {package_name}.regression import r2_score
# we can import from {package_name} import r2_score

# Author:   {author}
# Email:    {email}
# Date:     {date}
# Version:  {version}
# Package:  {package_name}
# Platform: {platform}
# Python:   {python_version}
# Usage:    {usage}
# License:  {license}
# Year:     {year}


# ---------------------------- Regression Metrics ---------------------------- #
from src.regression import r2_score
from src.regression import mean_squared_error
from src.regression import mean_absolute_error
from src.regression import root_mean_squared_error
from src.regression import mean_absolute_percentage_error
from src.regression import mean_squared_log_error
from src.regression import median_absolute_error
from src.regression import regression_report
# -------------------------- Classification Metrics -------------------------- #
from src.classification import accuracy_score
from src.classification import confusion_metrics
from src.classification import precision
from src.classification import recall
from src.classification import f1_score
from src.classification import auc


__all__ = [
    # ---------------------- Regression Metrics ---------------------- #
    "r2_score",
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "mean_squared_log_error",
    "median_absolute_error",
    "regression_report",
    # ---------------------- Classification Metrics ---------------------- #
    "accuracy_score",
    "confusion_metrics",
    "precision",
    "recall",
    "f1_score",
    "auc",
]
