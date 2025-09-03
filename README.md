## ✈️ Flight Price Prediction
### 📌 Project Overview

This project predicts flight ticket prices based on multiple features such as airline, source city, destination city, stops, travel class, flight duration, and days left before departure.

Built end-to-end:

- Data preprocessing

- Model training & evaluation

- Saving model artifacts

- Flask API deployment

- Frontend integration (HTML + JS)

### 📂 Dataset

Source: Flight-Price.csv (~300,000 rows)

-> Features:

airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left

Target: price

### ⚙️ Tech Stack

- Python: Pandas, NumPy, Scikit-learn

- Models: Linear Regression, Random Forest, Gradient Boosting

- Deployment: Flask API, Joblib

- Frontend: HTML + JavaScript

### 📊 Model Performance
🔹 Random Forest (Best Model)

R² Score: 0.98

MAE: ~₹1,600

RMSE: ~₹3,300

🔹 Gradient Boosting

R² Score: 0.95

MAE: ~₹2,900

RMSE: ~₹5,000

Both train and test performance were evaluated — Random Forest showed the best generalization.

### 🚀 Project Workflow

- Data Preprocessing

- Stratified train-test split (by price categories)

- Pipeline: imputation, scaling, one-hot encoding

- Model Training & Evaluation

- Compared LR, GBR, RF

- Metrics: R², MAE, RMSE, Cross-Validation

- Saving Artifacts

- Model + preprocessing pipeline saved with Joblib

- Deployment

- Flask API (/predict endpoint)

- Accepts JSON input → returns predicted price

- Frontend

- Clean HTML + JavaScript form

- Connects to Flask API and shows predicted price

### 🖼️ Screenshots

- Model evaluation results

- Cross-validation output

- Frontend form (before prediction)

- Frontend prediction output