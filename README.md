# 🏠 California House Price Prediction

A complete end-to-end machine learning project to predict house prices in California using advanced feature engineering, ensemble methods, and XGBoost.

---

## 📌 Project Overview

This project focuses on building a high-performance regression model to predict housing prices based on demographic, geographic, and housing-related features.

The workflow covers the full ML lifecycle:
- Data preprocessing
- Feature engineering
- Model training & evaluation
- Hyperparameter tuning
- Model deployment using Streamlit

---

## 🚀 Key Highlights

- 🔧 Custom preprocessing pipeline using `Pipeline` & `ColumnTransformer`
- 🧠 Custom transformers:
  - Bedroom imputation based on grouped bins
  - Location clustering using KMeans
- 📊 Advanced feature engineering:
  - Ratio-based features (rooms, population, income)
  - Geographic features (distance to city & coast)
- 🤖 Models used:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- ⚙️ Hyperparameter tuning using `RandomizedSearchCV`
- 🔥 Ensemble learning (RF + XGB)
- 🌐 Interactive UI using Streamlit

---

## 📊 Model Performance

| Model                | MAE (USD) |
|---------------------|----------|
| Linear Regression   | ~47,951  |
| Decision Tree       | ~47,422  |
| Random Forest       | ~30,136  |
| XGBoost             | ~27,450  |
| Ensemble (Final)    | **~27,418** |

---

## 🧠 Feature Engineering

Key engineered features include:

- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`
- `income_per_person`
- `rooms_per_person`
- `bedrooms_per_person`
- `distance_to_city`
- `distance_to_coast`
- `location_cluster` (via KMeans)

These features significantly improved model performance.

---

## ⚙️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---


## 🧠 Key Learnings

- Importance of proper preprocessing pipelines

- Difference between stateless and stateful transformations

- Impact of feature engineering on model performance

- When to use Random Forest vs XGBoost

- Practical use of hyperparameter tuning

- Building deployable ML systems

## 👤 Author

**Diab Saeed** 
