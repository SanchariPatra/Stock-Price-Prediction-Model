# 📈 Stock Price Prediction (Tesla - TSLA)

This project builds and evaluates machine learning models to predict the **next day's price movement** (up or down) for Tesla (TSLA) stock based on historical data.

## 📂 Dataset
- **Source**: `TSLA.csv` (Tesla historical daily prices)
- **Columns**:
  - `Date`
  - `Open`
  - `High`
  - `Low`
  - `Close`
  - `Adj Close`
  - `Volume`

**Data Range:** 2010–2022  
**Number of Records:** 2,956 rows

---

## 🛠 Features Created
From the raw data, several features were engineered:
- `year`, `month`, `day` → Extracted from `Date`
- `is_quarter_end` → 1 if month is a quarter end, else 0
- `open-close` → Difference between Open and Close prices
- `low-high` → Difference between Low and High prices
- `target` → **1** if the next day's closing price is higher than the current day's close, else **0**

---

## 📊 Exploratory Data Analysis (EDA)
The project includes:
- Distribution plots for price columns
- Boxplots for outlier detection
- Price trends over time
- Bar plots grouped by year
- Heatmap of correlations

---

## 🧠 Machine Learning Models
Three classifiers were trained:
1. **Logistic Regression**
2. **Support Vector Classifier (SVC)** – Polynomial kernel
3. **XGBoost Classifier**

**Feature scaling** was done using `StandardScaler`.

### Train/Test Split
- **Train**: 90%
- **Validation**: 10%
- **Random State**: 2022

---

## 📈 Results

| Model                  | Train ROC AUC | Validation ROC AUC |
|------------------------|--------------:|--------------------:|
| Logistic Regression    | 0.5145        | 0.5385              |
| SVC (poly)              | 0.5004        | 0.4596              |
| XGBoost                | 0.9241        | 0.4769              |

**Best validation performance:** Logistic Regression (ROC AUC ≈ 0.54)

---

## 🔍 Evaluation Metrics (Logistic Regression)
- **Accuracy**: 0.5236  
- **Precision**: 0.5256  
- **Recall**: 0.9872  
- **F1-score**: 0.6860  

> ⚠️ High recall but relatively low precision suggests the model predicts "up" frequently, leading to more false positives.

---

## 📦 Requirements
Install the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
