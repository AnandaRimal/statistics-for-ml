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
