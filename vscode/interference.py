import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attrbs, cat_attrbs):
    # For Numerical Columns
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown = "ignore")),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrbs),
        ("cat", cat_pipeline, cat_attrbs),
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("Flight-Price.csv")

    df['price_cat'] = pd.cut(df['price'], bins = [0, 5000, 10000, 20000, 40000, np.inf], labels = [1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_set, test_set in split.split(df, df['price_cat']):
        strat_train_set = df.iloc[train_set]
        strat_test_set = df.iloc[test_set]

    flight_train = strat_train_set.copy()
    flight_train_features = flight_train.drop(["price", "price_cat"], axis = 1)
    flight_train_labels = flight_train['price']
    flight_train_num = ['duration', 'days_left']
    flight_train_cat = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    
    pipeline = build_pipeline(flight_train_num, flight_train_cat)
    flight_prepared_train = pipeline.fit_transform(flight_train_features)
    model = GradientBoostingRegressor(random_state = 42)
    model.fit(flight_prepared_train, flight_train_labels)
    print("Dumping the Model and Pipeline Files.............")
    joblib.dump(model, MODEL_FILE) 
    joblib.dump(pipeline, PIPELINE_FILE) 
    print("Model is Trained and Saved.")
