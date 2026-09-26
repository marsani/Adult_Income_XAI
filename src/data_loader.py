import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv(DATA_URL, names=COLUMNS, skipinitialspace=True)
    
    # Handle missing values (represented as '?')
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop('income', axis=1)
    y = df['income']
    
    # Save encoders for later use (decoding in UI)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))
    
    # Save feature names
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, 'feature_names.pkl'))
    
    print("Data loaded and preprocessed.")
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    load_and_preprocess_data()
