# 📊 Stock Data Preprocessing & Linear Regression Model

## 📌 Project Overview

This project demonstrates a simple machine learning pipeline for stock market data analysis. It includes data preprocessing, feature scaling, feature selection, and building a predictive model using Linear Regression.

The main goal of this project is to prepare raw stock data for modeling and use it to predict target values based on selected features.

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

---

## 📁 Project Workflow

### 1. Data Preprocessing

* Selected numerical features from the dataset:

  ```
  ['Prev Close','Open', 'High', 'Low','Last','Close', 'Volume','VWAP','Turnover','Trades']
  ```
* Applied **Min-Max Scaling** to normalize the data between 0 and 1.
* Saved the processed dataset into:

  ```
  preprocessed_stock_dataset.csv
  ```

---

### 2. Feature Scaling

* Used `MinMaxScaler` from `sklearn.preprocessing`.
* Ensures all features are on the same scale, improving model performance.

---

### 3. Dataset Preparation

* Loaded the preprocessed dataset.
* Split the dataset into:

  * Training set (70%)
  * Testing set (30%)

---

### 4. Model Building

* Used **Linear Regression** for prediction.
* Trained the model using:

  ```
  lm.fit(x_train, y_train)
  ```

---

### 5. Prediction

* Generated predictions on test data:

  ```
  yhat = lm.predict(x_test)
  ```
* Displayed the first 10 predictions.

---

## 📌 Code Implementation

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Identify numerical features
numerical_features = ['Prev Close','Open', 'High', 'Low','Last','Close', 'Volume','VWAP','Turnover','Trades']

# Scaling
scaler = MinMaxScaler()
scaler.fit(df[numerical_features])
normalized_features = scaler.transform(df[numerical_features])
df[numerical_features] = normalized_features

# Save preprocessed data
df.to_csv('preprocessed_stock_dataset.csv', index=False)

# Load dataset
stocks_df = pd.read_csv("preprocessed_stock_dataset.csv")

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Model training
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

# Prediction
yhat = lm.predict(x_test)
print(yhat[:10])
```

---

## 📊 Key Concepts Covered

* Data Normalization (Min-Max Scaling)
* Train-Test Split
* Linear Regression Model
* Prediction on unseen data

---

## 🚀 How to Run the Project

1. Install required libraries:

   ```
   pip install pandas numpy scikit-learn
   ```
2. Place your dataset in the working directory.
3. Run the script:

   ```
   python main.py
   ```

---

## 📈 Future Improvements

* Add feature selection techniques (e.g., SelectKBest).
* Evaluate model performance using metrics like:

  * Mean Squared Error (MSE)
  * R² Score
* Try advanced models like:

  * Random Forest
  * Gradient Boosting

---

## 📝 Conclusion

This project provides a basic pipeline for preprocessing stock data and building a regression model. It serves as a foundation for more advanced financial prediction systems.

---
