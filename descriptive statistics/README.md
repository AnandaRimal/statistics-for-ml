# Descriptive Statistics

## Table of Contents

1.  **Measure of Central Tendency**
    *   Mean
    *   Median
    *   Mode
    *   Weighted Mean
    *   Trimmed Mean
2.  **Measure of Dispersion**
    *   Range
    *   Variance
    *   Standard Deviation
    *   Coefficient of Variation (CV)
    *   Interquartile Range (IQR)
3.  **Plotting Graphs in Statistics**
    *   Graphs for Univariate Analysis
        *   Categorical Data
        *   Numerical Data Visualization
    *   Graphs for Bivariate Analysis
        *   Categorical - Categorical Variables
        *   Numerical - Numerical Variables
        *   Categorical - Numerical Variables
4.  **Five Number Summary and Box Plot**
    *   Calculating Percentile
    *   Boxplot Using Python
5.  **Covariance and Correlation**
    *   Covariance
    *   Correlation
6.  **Graphs for Multivariate Analysis**
    *   3D Scatter Plot
    *   Hue Parameter in Seaborn
    *   FacetGrids
    *   Pair Plots
    *   Bubble Plot

---

# Chapter 1 — Measure of Central Tendency (Full Detailed Explanation)

**Topics covered in this chapter:**
*   Mean
*   Median
*   Mode
*   Weighted Mean
*   Trimmed Mean
*   Why central tendency matters in ML

---

## ⭐ 1. Mean (Arithmetic Average)

### Definition
The mean is the sum of all values divided by the number of values. It represents the "center of gravity" of the data.

$$
\mu = \frac{\sum_{i=1}^{n} x_i}{n}
$$

### Example Dataset
Suppose you have students’ CGPA:

| Student | CGPA |
| :--- | :--- |
| A | 3.0 |
| B | 3.5 |
| C | 2.8 |
| D | 3.7 |
| E | 3.0 |

**Calculation:**
$$
\mu = \frac{3.0 + 3.5 + 2.8 + 3.7 + 3.0}{5}
$$
$$
\mu = \frac{16.0}{5} = 3.2
$$

**Interpretation:**
The average CGPA is **3.2**.

### Visualization
![Mean Visualization](images/mean.png)

### Why it is Important
*   **Summarizes Data:** Provides a single value that represents the entire dataset.
*   **Basis for Other Metrics:** Essential for calculating Variance, Standard Deviation, and Correlation.
*   **Minimizes Error:** The sum of squared deviations from the mean is minimal compared to any other number.

### Why it is Needed in ML
*   **Data Imputation:** Filling missing numerical values with the mean is a common strategy.
*   **Feature Scaling:** Used in **Standardization (Z-score normalization)** ($x' = \frac{x - \mu}{\sigma}$), which is crucial for algorithms like SVM, KNN, and Neural Networks to converge faster.
*   **Model Evaluation:** Metrics like **MSE (Mean Squared Error)** rely on the mean of errors to quantify model performance.

---

## ⭐ 2. Median

### Definition
The median is the middle value when data is sorted. It splits the data into two equal halves.

### Steps
1.  Sort the data.
2.  Find the middle value.
3.  If even number of values → average of middle two.

### Example
**Raw Data:** 3.0, 3.5, 2.8, 3.7, 3.0
**Sorted Data:** 2.8, 3.0, **3.0**, 3.5, 3.7

**Median = 3.0**

### Visualization
![Median Visualization](images/median.png)

### Why it is Important
*   **Robustness:** Unlike the mean, the median is not skewed by extreme outliers.
*   **Real-World Representation:** Better represents the "typical" value in skewed distributions (e.g., Household Income, House Prices).

### Why it is Needed in ML
*   **Robust Imputation:** Preferred over mean for filling missing values when the data has outliers.
*   **Loss Functions:** **MAE (Mean Absolute Error)** is optimized by the median, making it robust to outliers in regression tasks.
*   **Anomaly Detection:** Used in **Median Absolute Deviation (MAD)** to detect outliers effectively.

---

## ⭐ 3. Mode

### Definition
The mode is the value that occurs most frequently in the dataset.

### Example
**CGPA values:** 3.0, **3.0**, 3.5, 3.7, 2.8
**Mode = 3.0** (appears twice)

### Visualization
![Mode Visualization](images/mode.png)

### Why it is Important
*   **Categorical Analysis:** The only measure of central tendency that works for non-numerical (nominal) data.
*   **Business Insight:** Identifies the most popular product, most common error type, etc.

### Why it is Needed in ML
*   **Categorical Imputation:** The standard method for filling missing values in categorical features.
*   **Classification Baseline:** The "Zero Rule" baseline model predicts the mode (most frequent class) to set a benchmark accuracy.
*   **K-Nearest Neighbors (KNN):** In classification, KNN predicts the class based on the **mode** of the k-nearest neighbors.

---

## ⭐ 4. Weighted Mean

### Definition
In some cases, not all observations contribute equally. Each data point has a **weight** ($w_i$) representing its importance.

$$
\text{Weighted Mean} = \frac{\sum w_i x_i}{\sum w_i}
$$

### Example
Your semester grades with different credit hours:

| Subject | Grade ($x_i$) | Credit ($w_i$) |
| :--- | :--- | :--- |
| Math | 3.7 | 4 |
| AI | 3.3 | 3 |
| Statistics | 3.0 | 3 |

**Calculation:**
$$
\frac{(3.7 \times 4) + (3.3 \times 3) + (3.0 \times 3)}{4 + 3 + 3}
$$
$$
= \frac{14.8 + 9.9 + 9.0}{10} = \frac{33.7}{10} = 3.37
$$

**Your weighted GPA = 3.37**

### Visualization
![Weighted Mean Visualization](images/weighted_mean.png)

### Why it is Important
*   **Fair Representation:** Accounts for the varying significance or reliability of different data points.
*   **Physics & Engineering:** Used to calculate Center of Mass.

### Why it is Needed in ML
*   **Ensemble Learning:** In **Boosting** and **Random Forests**, predictions from better-performing models are given higher weights.
*   **Class Imbalance:** Weighted loss functions assign higher weights to the minority class to prevent the model from ignoring it.
*   **Time-Series Forecasting:** Recent data points are often given higher weights than older ones (Exponential Moving Average).

---

## ⭐ 5. Trimmed Mean

### Definition
The trimmed mean removes a small percentage of the largest and smallest values before calculating the mean. This makes it robust to outliers.

### Example (10% trimmed mean)
**Data:** 10, 11, 12, 13, 14, **100** (Outlier)

1.  Remove top 10% and bottom 10%.
2.  Remaining: 11, 12, 13, 14.
3.  **Mean = 12.5** (Much closer to the "real" center than the raw mean).

### Visualization
![Trimmed Mean Visualization](images/trimmed_mean.png)

### Why it is Important
*   **Compromise:** Offers a middle ground between the Mean (efficient but sensitive) and Median (robust but ignores magnitude).
*   **Sports Scoring:** Used in Olympics (diving, gymnastics) to remove biased judging (highest and lowest scores dropped).

### Why it is Needed in ML
*   **Robust Regression:** Algorithms like **RANSAC** use similar principles to ignore outliers during model training.
*   **Metric Stability:** Calculating trimmed means of evaluation metrics during cross-validation ensures that one bad fold doesn't skew the overall performance assessment.
*   **Noise Reduction:** Effective in signal processing and sensor data cleaning before feeding into ML models.

---

## ⭐ 6. Why Central Tendency Matters in Machine Learning

ML relies heavily on central tendency for:

### 1. Normalization
$$
x' = \frac{x - \text{mean}}{\text{std}}
$$
Used in: Neural Networks, SVM, Logistic Regression, K-Means.

### 2. Feature Understanding
*   Helps in detecting **skewness**.
*   Understanding **distribution**.
*   **Outlier detection**.

### 3. Loss Functions Are Based on Means
*   **Mean Absolute Error (MAE)**
*   **Mean Squared Error (MSE)**
*   **Root Mean Squared Error (RMSE)**

---

# ⭐ **CHAPTER 2 — MEASURES OF DISPERSION (FULL DETAILED EXPLANATION)**

Dispersion tells us **how spread out the data is**.
Central tendency (mean/median/mode) tells you **where the center is**, but **dispersion tells you how variable or stable** your data is.

**Topics covered in this chapter:**
1.  **Range**
2.  **Variance**
3.  **Standard Deviation (SD)**
4.  **Coefficient of Variation (CV)**
5.  **Interquartile Range (IQR)**
6.  **Why dispersion matters in Machine Learning**

---

## ⭐ 1. RANGE

### Definition
Range is the **difference between the maximum and minimum** value.

$$
\text{Range} = \max(x) - \min(x)
$$

### Example
**IQ scores:** 85, 90, 95, 100, 110, 120

$$
\text{Range} = 120 - 85 = 35
$$

### Visualization
> *[Image: Range Visualization - Generation Pending due to Quota Limit]*

### Why it is Important
*   **Simplicity:** The easiest way to get a quick sense of the data's spread.
*   **Boundary Detection:** Immediately identifies the limits of the dataset.

### Why it is Needed in ML
*   **Feature Scaling:** Used in **Min-Max Normalization** ($x' = \frac{x - \min}{\max - \min}$) to scale features to a fixed range [0, 1].
*   **Anomaly Detection:** Extreme range values can indicate data entry errors or anomalies.
*   **Data Validation:** Ensuring new data falls within the expected range of training data.

---

## ⭐ 2. Variance ($\sigma^2$)

### Definition
Variance tells **how far each data point is from the mean** on average. It measures the average squared deviation.

$$
\sigma^2 = \frac{\sum (x_i - \mu)^2}{n}
$$

### Example
**Dataset:** 5, 7, 9
**Mean:** 7

| $x$ | $x - \mu$ | $(x - \mu)^2$ |
| :--- | :--- | :--- |
| 5 | -2 | 4 |
| 7 | 0 | 0 |
| 9 | 2 | 4 |

$$
\sigma^2 = \frac{4 + 0 + 4}{3} = \frac{8}{3} = 2.67
$$

### Visualization
> *[Image: Variance Visualization - Generation Pending due to Quota Limit]*

### Why it is Important
*   **Mathematical Foundation:** The basis for Standard Deviation, Covariance, and Correlation.
*   **Risk Assessment:** In finance, variance measures the volatility (risk) of an asset.

### Why it is Needed in ML
*   **Principal Component Analysis (PCA):** PCA reduces dimensionality by finding directions (principal components) that maximize **variance** (information).
*   **Feature Selection:** Features with near-zero variance carry little information and are often dropped.
*   **Bias-Variance Tradeoff:** A core concept in ML; high variance models (overfitting) capture noise, while low variance models (underfitting) miss patterns.

---

## ⭐ 3. Standard Deviation (SD)

### Definition
SD is the **square root of variance**. It brings the measure of spread back to the original unit of the data.

$$
\sigma = \sqrt{\sigma^2}
$$

### Example
Using previous variance of 2.67:
$$
\sigma = \sqrt{2.67} = 1.63
$$

### Visualization
> *[Image: Standard Deviation Visualization - Generation Pending due to Quota Limit]*

### Why it is Important
*   **Interpretability:** Unlike variance, SD is in the same units as the data (e.g., "spread of $10" vs "variance of $100^2").
*   **Normal Distribution:** In a normal distribution, ~68% of data lies within $\mu \pm 1\sigma$.

### Why it is Needed in ML
*   **Standardization:** Used to scale features to have $\mu=0$ and $\sigma=1$ ($x' = \frac{x - \mu}{\sigma}$), essential for gradient descent optimization.
*   **Weight Initialization:** Methods like **Xavier** and **He** initialization use SD to set initial weights, preventing vanishing/exploding gradients.
*   **Outlier Detection:** Z-score method identifies outliers as points beyond $\pm 3\sigma$.

---

## ⭐ 4. Coefficient of Variation (CV)

### Definition
CV measures **relative spread**. It is unitless, allowing comparison of variability between datasets with different scales.

$$
CV = \frac{\sigma}{\mu}
$$

### Example
*   **Dataset A:** Mean = 50, SD = 10 $\rightarrow CV = 0.2$
*   **Dataset B:** Mean = 100, SD = 10 $\rightarrow CV = 0.1$
*   **Dataset B is more stable.**

### Visualization
> *[Image: Coefficient of Variation Visualization - Generation Pending due to Quota Limit]*

### Why it is Important
*   **Comparability:** Allows comparing the volatility of stocks with different prices or the variability of height vs weight.

### Why it is Needed in ML
*   **Feature Selection:** Helps identify features that are relatively stable vs highly volatile regardless of their magnitude.
*   **Model Stability:** Comparing CV of model performance metrics across cross-validation folds to assess stability.

---

## ⭐ 5. Interquartile Range (IQR)

### Definition
IQR measures the spread of the **middle 50%** of the data. It is the difference between the 75th percentile (Q3) and the 25th percentile (Q1).

$$
IQR = Q3 - Q1
$$

### Example
**Data:** 4, 7, 8, 10, 12, 15, 18, 21
**Q1 (25%):** 8
**Q3 (75%):** 18

$$
IQR = 18 - 8 = 10
$$

### Visualization
> *[Image: IQR Visualization - Generation Pending due to Quota Limit]*

### Why it is Important
*   **Robustness:** Like the median, IQR is not affected by extreme outliers.
*   **Data Distribution:** Gives a clear picture of where the bulk of the data lies.

### Why it is Needed in ML
*   **Outlier Removal:** The **IQR Rule** (Values < $Q1 - 1.5IQR$ or > $Q3 + 1.5IQR$) is a standard robust method for detecting and removing outliers.
*   **Robust Scaling:** Scaling features using median and IQR ($x' = \frac{x - Q1}{IQR}$) is preferred when data contains many outliers.
*   **Box Plots:** The core component of box plots, used extensively in EDA.

---

## ⭐ Final Section: Why Dispersion Matters in Machine Learning

Dispersion measures are **critical** in ML for:

1.  **Feature Scaling:** Algorithms like SVM, KNN, and Neural Networks require scaled data (using Range or SD) to function correctly.
2.  **Gradient Behavior:** The variance of weights affects signal propagation in Deep Networks (Vanishing/Exploding gradients).
3.  **Outlier Detection:** SD and IQR are the primary tools for identifying anomalies that can skew models.
4.  **Bias-Variance Tradeoff:** Understanding the variance of model predictions is key to diagnosing overfitting.
5.  **Dimensionality Reduction:** PCA relies entirely on maximizing variance to preserve information.

