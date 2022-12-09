Regression Metrics
------------------

1. R2 Score

R2 score, also known as the coefficient of determination, is a
statistical measure of how close the data are to the fitted regression
line. It is also known as the coefficient of determination, or the
coefficient of multiple determination for multiple regression.

.. code:: math

   R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}

2. Mean Absolute Error

.. code:: math

   MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|

3. Mean Squared Error

.. code:: math

   MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2

4. Root Mean Squared Error

.. code:: math

   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}

5. Mean Absolute Percentage Error

.. code:: math

   MAPE = \frac{100}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{y_i}

6. Mean Squared Logarithmic Error

.. code:: math

   MSLE = \frac{1}{n} \sum_{i=1}^n (log(y_i + 1) - log(\hat{y}_i + 1))^2

7. Median Absolute Error

.. code:: math

   MdAE = median(|y_i - \hat{y}_i|)

8. Median Squared Error

.. code:: math

   MdSE = median((y_i - \hat{y}_i)^2)

9. Median Absolute Percentage Error

.. code:: math

   MdAPE = median(\frac{|y_i - \hat{y}_i|}{y_i})

10. Median Squared Logarithmic Error

.. code:: math

   MdSLE = median((log(y_i + 1) - log(\hat{y}_i + 1))^2)

11. Explained Variance Score

.. code:: math

   EV = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}

12. Max Error

.. code:: math

   max_error = max(|y_i - \hat{y}_i|)

13. Mean Bias Error

.. code:: math

   MBE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)

14. Mean Percentage Error

.. code:: math

   MPE = \frac{100}{n} \sum_{i=1}^n \frac{y_i - \hat{y}_i}{y_i}

15. Mean Squared Percentage Error

.. code:: math

   MSPE = \frac{100}{n} \sum_{i=1}^n \frac{(y_i - \hat{y}_i)^2}{y_i^2}

16. Median Bias Error

.. code:: math

   MdBE = median(y_i - \hat{y}_i)

17. Median Percentage Error

.. code:: math

   MdPE = median(\frac{y_i - \hat{y}_i}{y_i})

18. Median Squared Percentage Error

.. code:: math

   MdSPE = median(\frac{(y_i - \hat{y}_i)^2}{y_i^2})

19. Mean Absolute Scaled Error

.. code:: math

   MASE = \frac{1}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{\frac{1}{n-1} \sum_{i=1}^n |y_i - \bar{y}_i|}

20. Mean Squared Scaled Error

.. code:: math

   MSSE = \frac{1}{n} \sum_{i=1}^n \frac{(y_i - \hat{y}_i)^2}{\frac{1}{n-1} \sum_{i=1}^n (y_i - \bar{y}_i)^2}

21. Median Absolute Scaled Error

.. code:: math

   MdASE = median(\frac{|y_i - \hat{y}_i|}{\frac{1}{n-1} \sum_{i=1}^n |y_i - \bar{y}_i|})

22. Median Squared Scaled Error

.. code:: math

   MdSSE = median(\frac{(y_i - \hat{y}_i)^2}{\frac{1}{n-1} \sum_{i=1}^n (y_i - \bar{y}_i)^2})