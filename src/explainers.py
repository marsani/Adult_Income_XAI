import shap
import lime
import lime.lime_tabular
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to the root directory (/mount/src/adult_income_xai)
BASE_DIR = Path(__file__).resolve().parent.parent

# Point to your model file location (adjust subfolder and filename as needed)
MODEL_PATH = BASE_DIR / "models" / "rf_model.joblib"

# Load the model
# rf_model = joblib.load(MODEL_PATH)

# Load artifacts
# MODEL_PATH = 'models/rf_model.pkl'
X_TRAIN_SAMPLE_PATH = 'models/X_train_sample.pkl'
FEATURE_NAMES_PATH = 'models/feature_names.pkl'

rf_model = joblib.load(MODEL_PATH)
X_train_sample = joblib.load(X_TRAIN_SAMPLE_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# Initialize Explainers
# LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_sample),
    feature_names=feature_names,
    class_names=['<=50K', '>50K'],
    mode='classification'
)

# SHAP
shap_explainer = shap.TreeExplainer(rf_model)

def update_model(new_model):
    global rf_model, shap_explainer
    rf_model = new_model
    shap_explainer = shap.TreeExplainer(rf_model)

def predict_proba_wrapper(data):
    # LIME passes a numpy array, but RF expects a DataFrame with feature names
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=feature_names)
    return rf_model.predict_proba(data)

def get_lime_explanation(instance):
    # instance should be a 1D numpy array
    exp = lime_explainer.explain_instance(
        data_row=instance, 
        predict_fn=predict_proba_wrapper
    )
    return exp

def get_shap_values(instance):
    # instance should be a DataFrame or 2D array
    shap_values = shap_explainer.shap_values(instance)
    return shap_values, shap_explainer.expected_value
