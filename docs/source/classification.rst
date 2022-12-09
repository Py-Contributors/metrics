Classification Metrics
----------------------

1. Accuracy

.. code:: math

   Accuracy = \frac{TP + TN}{TP + TN + FP + FN}

2. Precision

.. code:: math

   Precision = \frac{TP}{TP + FP}

3. Recall

.. code:: math

   Recall = \frac{TP}{TP + FN}

4. F1 Score

.. code:: math

   F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}

5. Matthews Correlation Coefficient

.. code:: math

   MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}

6. Cohen’s Kappa

.. code:: math

   Kappa = \frac{p_o - p_e}{1 - p_e}

where

.. code:: math

   p_o = \frac{TP + TN}{TP + TN + FP + FN}

.. code:: math

   p_e = \frac{TP + FP}{TP + TN + FP + FN} \times \frac{TP + FN}{TP + TN + FP + FN} + \frac{TN + FP}{TP + TN + FP + FN} \times \frac{TN + FN}{TP + TN + FP + FN}

7. Area Under the Receiver Operating Characteristic Curve (ROC AUC)

.. code:: math

   ROC AUC = \frac{1}{2} \sum_{i=1}^{n-1} (TPR_i - TPR_{i+1}) \times (FPR_i + FPR_{i+1})

8. Area Under the Precision-Recall Curve (PR AUC)

.. code:: math

   PR AUC = \frac{1}{2} \sum_{i=1}^{n-1} (Recall_i - Recall_{i+1}) \times (Precision_i + Precision_{i+1})

9. Hamming Loss

.. code:: math

   Hamming Loss = \frac{1}{n} \sum_{i=1}^n \frac{1}{m} \sum_{j=1}^m I(y_{ij} \neq \hat{y}_{ij})

10. Zero-One Loss

.. code:: math

   Zero-One Loss = \frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y}_i)

11. Jaccard Similarity Score

.. code:: math

   Jaccard = \frac{TP}{TP + FP + FN}

12. Fowlkes-Mallows Score

.. code:: math

   FM = \sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}

13. Log Loss

.. code:: math

   Log Loss = - \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m y_{ij} \times log(\hat{y}_{ij})

14. Cross-Entropy Loss

.. code:: math

   Cross-Entropy Loss = - \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m y_{ij} \times log(\hat{y}_{ij}) - (1 - y_{ij}) \times log(1 - \hat{y}_{ij})

15. Hinge Loss

.. code:: math

   Hinge Loss = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m max(0, 1 - y_{ij} \times \hat{y}_{ij})

16. Squared Hinge Loss

.. code:: math

   Squared Hinge Loss = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m (max(0, 1 - y_{ij} \times \hat{y}_{ij}))^2

17. Classification Error

.. code:: math

   Classification Error = \frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y}_i)

18. Balanced Classification Error

.. code:: math

   Balanced Classification Error = \frac{1}{n} \sum_{i=1}^n \frac{1}{m} \sum_{j=1}^m I(y_{ij} \neq \hat{y}_{ij})