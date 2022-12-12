<p align="center">
  <a href="https://py-contributors.github.io/audiobook/"><img src="https://capsule-render.vercel.app/api?type=rect&color=009ACD&height=100&section=header&text=Metrics&fontSize=80%&fontColor=ffffff" alt="website title image"></a>
</p>

<p align="center">
    <a href="https://twitter.com/pycontributors"><img src="https://img.shields.io/twitter/follow/pycontributors?style=social" alt="Twitter" /></a>
    <a href="https://github.com/codeperfectplus?tab=followers"><img src="https://img.shields.io/github/followers/codeperfectplus.svg?style=social&label=Follow&maxAge=2592000"/></a>
</p>
</br>

Implementation of various metrics for regression and classification problems. For Data Science and Machine Learning projects, it is important to have a good understanding of the metrics used to evaluate the performance of the model. This repository contains the implementation of various metrics for regression and classification problems. The metrics are implemented in Python and are available as a Python package. The metrics are implemented using NumPy and are implemented from scratch. The metrics are implemented using the formulae given in the Wikipedia pages for the respective metrics. The metrics are implemented in the following order:

## Regression Metrics

1. R2 Score 

R2 score, also known as the coefficient of determination, is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

```math
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
```

2. Mean Absolute Error

```math
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
```

3. Mean Squared Error

```math
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
```

4. Root Mean Squared Error

```math
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
```

5. Mean Absolute Percentage Error

```math
MAPE = \frac{100}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{y_i}
```

6. Mean Squared Logarithmic Error

```math
MSLE = \frac{1}{n} \sum_{i=1}^n (log(y_i + 1) - log(\hat{y}_i + 1))^2
```

7. Median Absolute Error

```math
MdAE = median(|y_i - \hat{y}_i|)
```

8. Median Squared Error

```math
MdSE = median((y_i - \hat{y}_i)^2)
```

9. Median Absolute Percentage Error

```math
MdAPE = median(\frac{|y_i - \hat{y}_i|}{y_i})
```

10. Median Squared Logarithmic Error

```math
MdSLE = median((log(y_i + 1) - log(\hat{y}_i + 1))^2)
```

11. Explained Variance Score

```math
EV = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
```

12. Max Error

```math
max_error = max(|y_i - \hat{y}_i|)
```

13. Mean Bias Error

```math
MBE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)
```

14. Mean Percentage Error

```math
MPE = \frac{100}{n} \sum_{i=1}^n \frac{y_i - \hat{y}_i}{y_i}
```

15. Mean Squared Percentage Error

```math
MSPE = \frac{100}{n} \sum_{i=1}^n \frac{(y_i - \hat{y}_i)^2}{y_i^2}
```

16. Median Bias Error

```math
MdBE = median(y_i - \hat{y}_i)
```

17. Median Percentage Error

```math
MdPE = median(\frac{y_i - \hat{y}_i}{y_i})
```

18. Median Squared Percentage Error

```math
MdSPE = median(\frac{(y_i - \hat{y}_i)^2}{y_i^2})
```

19. Mean Absolute Scaled Error

```math
MASE = \frac{1}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{\frac{1}{n-1} \sum_{i=1}^n |y_i - \bar{y}_i|}
```

20. Mean Squared Scaled Error

```math
MSSE = \frac{1}{n} \sum_{i=1}^n \frac{(y_i - \hat{y}_i)^2}{\frac{1}{n-1} \sum_{i=1}^n (y_i - \bar{y}_i)^2}
```

21. Median Absolute Scaled Error

```math
MdASE = median(\frac{|y_i - \hat{y}_i|}{\frac{1}{n-1} \sum_{i=1}^n |y_i - \bar{y}_i|})
```

22. Median Squared Scaled Error

```math
MdSSE = median(\frac{(y_i - \hat{y}_i)^2}{\frac{1}{n-1} \sum_{i=1}^n (y_i - \bar{y}_i)^2})
```

## Classification Metrics

1. Accuracy

```math
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
```

2. Precision

```math
Precision = \frac{TP}{TP + FP}
```

3. Recall

```math
Recall = \frac{TP}{TP + FN}
```

4. F1 Score

```math
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
```

5. Matthews Correlation Coefficient

```math
MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
```

6. Cohen's Kappa

```math
Kappa = \frac{p_o - p_e}{1 - p_e}
```

where

```math
p_o = \frac{TP + TN}{TP + TN + FP + FN}
```

```math
p_e = \frac{TP + FP}{TP + TN + FP + FN} \times \frac{TP + FN}{TP + TN + FP + FN} + \frac{TN + FP}{TP + TN + FP + FN} \times \frac{TN + FN}{TP + TN + FP + FN}
```

7. Area Under the Receiver Operating Characteristic Curve (ROC AUC)

```math
ROC AUC = \frac{1}{2} \sum_{i=1}^{n-1} (TPR_i - TPR_{i+1}) \times (FPR_i + FPR_{i+1})
```

8. Area Under the Precision-Recall Curve (PR AUC)

```math
PR AUC = \frac{1}{2} \sum_{i=1}^{n-1} (Recall_i - Recall_{i+1}) \times (Precision_i + Precision_{i+1})
```

9. Hamming Loss

```math
Hamming Loss = \frac{1}{n} \sum_{i=1}^n \frac{1}{m} \sum_{j=1}^m I(y_{ij} \neq \hat{y}_{ij})
```

10. Zero-One Loss

```math
Zero-One Loss = \frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y}_i)
```

11. Jaccard Similarity Score

```math
Jaccard = \frac{TP}{TP + FP + FN}
```

12. Fowlkes-Mallows Score

```math
FM = \sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}
```

13. Log Loss

```math
Log Loss = - \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m y_{ij} \times log(\hat{y}_{ij})
```

14. Cross-Entropy Loss

```math
Cross-Entropy Loss = - \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m y_{ij} \times log(\hat{y}_{ij}) - (1 - y_{ij}) \times log(1 - \hat{y}_{ij})
```

15. Hinge Loss

```math
Hinge Loss = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m max(0, 1 - y_{ij} \times \hat{y}_{ij})
```

16. Squared Hinge Loss

```math
Squared Hinge Loss = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m (max(0, 1 - y_{ij} \times \hat{y}_{ij}))^2
```

17. Classification Error

```math
Classification Error = \frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y}_i)
```

18. Balanced Classification Error

```math
Balanced Classification Error = \frac{1}{n} \sum_{i=1}^n \frac{1}{m} \sum_{j=1}^m I(y_{ij} \neq \hat{y}_{ij})
```

## Clustering Metrics



