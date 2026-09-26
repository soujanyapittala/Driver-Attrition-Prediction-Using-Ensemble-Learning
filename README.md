# Driver-Attrition-Prediction-Using-Ensemble-Learning
Driver attrition prediction using machine learning, featuring EDA, KNN imputation, feature engineering, class imbalance handling, standardization, and ensemble learning with Bagging and Boosting techniques.
# 🚗 Driver Attrition Prediction Using Ensemble Learning

> **Machine Learning | Classification | Ensemble Learning | Business Analytics**

## 📌 Overview

Driver attrition is a significant business challenge in the mobility and service industry. Frequent driver turnover increases recruitment and onboarding costs while affecting operational continuity.

This project develops a **machine learning-based driver attrition prediction system** to identify drivers who are more likely to leave based on demographic characteristics, tenure, income, performance, grade, business value, and rating-related features.

The project applies **Exploratory Data Analysis, KNN Imputation, Feature Engineering, SMOTE, Standardization, Bagging, Boosting, and Model Evaluation using ROC-AUC and classification metrics.**

---

## 🎯 Business Objective

The primary objective is to predict whether a driver is likely to leave the organization.

### Target Variable

```text
0 → Driver stays
1 → Driver leaves
```

A driver is considered to have left when a `LastWorkingDate` is available.

This prediction can support data-driven retention strategies by helping organizations identify potential attrition risk and understand the factors associated with driver turnover.

---

## 📊 Dataset

The dataset contains **monthly driver-level observations** covering a 24-month reporting period.

### Dataset Statistics

| Metric               |        Value |
| -------------------- | -----------: |
| Total observations   |       19,104 |
| Unique drivers       |        2,381 |
| Reporting periods    |    24 months |
| Features             |           14 |
| Final modeling level | Driver-level |

The raw dataset contains multiple monthly records for individual drivers, with an average of approximately **8 records per driver**. Therefore, driver-level aggregation was performed before model development.

---

## 🧾 Key Features

| Feature                | Description                           |
| ---------------------- | ------------------------------------- |
| `Driver_ID`            | Unique driver identifier              |
| `Age`                  | Driver age                            |
| `Gender`               | Driver gender                         |
| `City`                 | Driver city code                      |
| `Education_Level`      | Education category                    |
| `Income`               | Monthly average income                |
| `Dateofjoining`        | Driver joining date                   |
| `LastWorkingDate`      | Last recorded working date            |
| `Joining Designation`  | Designation at the time of joining    |
| `Grade`                | Driver grade                          |
| `Total Business Value` | Business value acquired by the driver |
| `Quarterly Rating`     | Driver quarterly performance rating   |

The dataset contains 29 cities, five joining-designation levels, five grade levels, and quarterly ratings ranging from 1–4 in the supplied data.

---

# 🔬 Project Workflow

```text
                 Raw Driver Data
                       │
                       ▼
              Data Understanding
                       │
                       ▼
             Exploratory Data Analysis
                       │
                       ▼
                Date Conversion
                       │
                       ▼
             Missing Value Analysis
                       │
                       ▼
                 KNN Imputation
                       │
                       ▼
              Feature Engineering
                       │
                       ▼
             Driver-Level Aggregation
                       │
                       ▼
               Target Creation
                       │
                       ▼
             Categorical Encoding
                       │
                       ▼
              Train/Test Split
                       │
                       ▼
              Class Imbalance
                    (SMOTE)
                       │
                       ▼
                Standardization
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Random Forest         AdaBoost
          Bagging             Boosting
              │                 │
              └────────┬────────┘
                       ▼
                Model Evaluation
                       │
                       ▼
              Business Insights
```

---

# 1. 🔍 Exploratory Data Analysis

The initial analysis focused on understanding the structure, distributions, missing values, relationships, and business meaning of the dataset.

### Dataset Structure

The original dataset contains:

* **19,104 records**
* **14 columns**
* **2,381 unique drivers**
* **24 reporting periods**

The monthly structure means that the same driver can appear multiple times, making driver-level aggregation necessary for modeling.

### Missing Values

The major missing-value pattern was:

| Feature           | Missing % |
| ----------------- | --------: |
| `LastWorkingDate` |    97.03% |
| `Age`             |     0.32% |
| `Gender`          |     0.27% |

`LastWorkingDate` was **not treated as an ordinary missing-value problem**, because its missingness has business meaning and is directly related to the target definition.

### Distribution Insights

* Age is concentrated around the **30–40** range.
* Income is **right-skewed**.
* Total Business Value is **strongly right-skewed**.
* Total Business Value contains negative values that may represent cancellations, refunds, or EMI adjustments and therefore were not automatically treated as erroneous outliers.
* The driver population is male-dominated.
* Grades 1–3 contain most observations.
* Lower quarterly ratings are more common than higher ratings.

---

# 2. 🧹 Data Preprocessing

## Date Conversion

The following fields were converted to datetime:

```text
MMM-YY
Dateofjoining
LastWorkingDate
```

The reporting period covers **January 2019 to December 2020**.

## KNN Imputation

K-Nearest Neighbors imputation was used for appropriate numerical variables.

The selected variables included:

```text
Age
Gender
Education_Level
Income
Joining Designation
Grade
Total Business Value
Quarterly Rating
```

The numerical data was standardized before KNN imputation using `StandardScaler`, with:

```python
KNNImputer(
    n_neighbors=5,
    weights="distance"
)
```

After imputation, missing `Age` and `Gender` values were successfully handled.

---

# 3. ⚙️ Feature Engineering

Feature engineering was performed to transform the monthly driver data into meaningful driver-level attributes.

### Quarterly Rating Improvement

A binary feature was created:

```text
Quarterly_Rating_Increased

1 → Final rating > Initial rating
0 → Otherwise
```

### Income Improvement

A binary feature was created:

```text
Income_Increased

1 → Final income > Initial income
0 → Otherwise
```

Both features compare the driver's first and last observed values after chronological ordering.

### Driver-Level Aggregation

Monthly observations were aggregated by `Driver_ID`.

The aggregation included:

* Latest reporting month
* Latest demographic values
* First joining date
* Last working date
* Latest grade
* Total business value
* Latest quarterly rating
* Target variable

This converted the longitudinal monthly dataset into a driver-level modeling dataset.

---

# 4. 🎯 Target Variable

The target was created using `LastWorkingDate`:

```python
df["target"] = df["LastWorkingDate"].notna().astype(int)
```

The supplied notebook identifies:

```text
0 → Stayed
1 → Left
```

The raw monthly observations are highly imbalanced, making imbalance treatment important before model training.

---

# 5. ⚖️ Class Imbalance Treatment

Because driver attrition represents the minority class, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

The test dataset was kept untouched so that model evaluation remained representative of the original class distribution.

This is particularly important because accuracy alone can be misleading when the target classes are imbalanced.

---

# 6. 🤖 Ensemble Learning

Two ensemble learning approaches were implemented.

## 🌲 Random Forest — Bagging

Random Forest was used as the **Bagging** algorithm.

The model combines predictions from multiple decision trees to improve robustness and reduce variance.

**Number of estimators:**

```text
200
```

## ⚡ AdaBoost — Boosting

AdaBoost was used as the **Boosting** algorithm.

The algorithm sequentially combines weak learners while focusing on observations that were previously misclassified.

**Number of estimators:**

```text
200
```

Both models were trained using the SMOTE-balanced training dataset.

---

# 7. 📈 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

### Model Comparison

| Model         | Approach | Accuracy | Precision* |  Recall* |      F1* |    ROC-AUC |
| ------------- | -------- | -------: | ---------: | -------: | -------: | ---------: |
| Random Forest | Bagging  |     0.69 |       0.40 |     0.57 |     0.47 | **0.7407** |
| AdaBoost      | Boosting |     0.62 |       0.36 | **0.80** | **0.50** |     0.7343 |

*Precision, recall and F1 shown for the **attrition/Left class**.

Random Forest achieved a ROC-AUC of **0.7407**, while AdaBoost achieved **0.7343**. AdaBoost produced higher recall for the attrition class, while Random Forest produced higher precision.

### Why ROC-AUC Matters

Because the dataset is imbalanced, ROC-AUC provides a useful measure of the models' ability to distinguish between drivers who stay and drivers who leave across classification thresholds.

The project therefore considers **ROC-AUC and minority-class recall alongside accuracy**, rather than relying on accuracy alone.

---

# 8. 🧮 Confusion Matrix Interpretation

For the attrition prediction problem:

### True Negative — TN

Driver actually stayed and the model predicted stayed.

### True Positive — TP

Driver actually left and the model predicted left.

### False Positive — FP

Driver stayed but the model predicted that the driver would leave.

### False Negative — FN

Driver left but the model predicted that the driver would stay.

From a retention perspective, false negatives can be particularly important because a potentially at-risk driver may not receive a retention intervention.

---

# 9. 💡 Business Insights

The analysis highlights several areas that can be monitored when studying driver attrition:

### Performance

Quarterly rating and changes in performance can provide useful signals for understanding driver behavior.

### Income

Income distribution is right-skewed, and income-related changes were incorporated into the feature set.

### Business Performance

Total Business Value is highly variable and strongly skewed, making it an important business-performance feature.

### Grade

Driver grades are concentrated in lower-to-middle levels, with relatively few observations in Grade 5.

### Geography

The dataset contains **29 cities**, allowing city-level attrition patterns to be investigated.

### Tenure

Driver tenure was incorporated into the analytical framework to study whether time spent with the organization is associated with attrition.

These relationships should be interpreted as **associations in the dataset rather than causal effects**.

---

# 10. 🚀 Actionable Recommendations

Based on the analytical framework, organizations can consider:

### 1. Early Attrition Monitoring

Use model-generated risk signals to identify drivers who may require closer engagement.

### 2. Retention Segmentation

Segment drivers based on factors such as:

* Tenure
* Income
* Grade
* Performance rating
* Business value
* City

### 3. Performance-Based Interventions

Monitor changes in quarterly ratings and business performance to identify drivers whose engagement or performance may be changing.

### 4. Income & Incentive Analysis

Investigate whether income changes are associated with attrition and evaluate targeted incentive strategies.

### 5. City-Level Analysis

Compare attrition patterns across cities to identify locations requiring deeper operational investigation.

### 6. Focus on Recall

Where the business cost of missing an at-risk driver is high, classification thresholds can be evaluated with greater emphasis on attrition recall.

---

# 🛠️ Technology Stack

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn
Jupyter Notebook
Git & GitHub
```

### Machine Learning Techniques

```text
Exploratory Data Analysis
KNN Imputation
Feature Engineering
Categorical Encoding
SMOTE
Feature Standardization
Random Forest
AdaBoost
Hyperparameter Tuning
Classification
ROC-AUC Analysis
Confusion Matrix
```

---



> **Dataset note:** If redistribution of the original dataset is not permitted, do not upload the raw CSV. Instead, provide dataset-source information and instructions for obtaining it.

---

# 📌 Key Takeaways

This project demonstrates an end-to-end machine learning workflow for a real-world **imbalanced classification problem**:

```text
Business Problem
      ↓
EDA
      ↓
Data Cleaning
      ↓
KNN Imputation
      ↓
Feature Engineering
      ↓
Driver-Level Aggregation
      ↓
SMOTE
      ↓
Standardization
      ↓
Bagging + Boosting
      ↓
Model Evaluation
      ↓
Business Insights
```

The project demonstrates how machine learning can move beyond model training toward **business-oriented risk identification and actionable analysis**.

---

## ⚠️ Disclaimer

This project is an educational machine-learning case study. Model outputs represent patterns learned from the supplied dataset and should not be interpreted as definitive explanations of individual driver behavior or as guaranteed predictions of future attrition.

---

## 👩‍💻 Author

**Soujanya Pittala**

**Data Analyst | Data Science & Machine Learning**

Skills: Python • SQL • Machine Learning • Data Analysis • Statistics • Visualization • Business Analytics

---

⭐ **If you found this project useful, consider starring the repository.**
