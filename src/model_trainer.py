import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
    from src.data_loader import load_and_preprocess_data
except ImportError:
    from data_loader import load_and_preprocess_data

def train_model(n_estimators=100, max_depth=None, min_samples_split=2, class_balancing="SMOTE"):
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print(f"Training Random Forest Classifier (n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, class_balancing={class_balancing})...")
    
    # Handle Class Imbalance
    if class_balancing == "SMOTE":
        try:
            import importlib
            imblearn_os = importlib.import_module("imblearn.over_sampling")
            SMOTE = getattr(imblearn_os, "SMOTE")
            print("Applying SMOTE to balance the dataset...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"Original dataset shape: {y_train.value_counts().to_dict()}")
            print(f"Resampled dataset shape: {y_train_balanced.value_counts().to_dict()}")
            
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            rf.fit(X_train_balanced, y_train_balanced)
        except (ImportError, ModuleNotFoundError, AttributeError):
            print("imbalanced-learn not found. Falling back to class_weight='balanced'.")
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)
            class_balancing = "Class Weight (Balanced) (SMOTE fallback)"
            
    elif class_balancing == "Random Over-sampling":
        try:
            import importlib
            imblearn_os = importlib.import_module("imblearn.over_sampling")
            RandomOverSampler = getattr(imblearn_os, "RandomOverSampler")
            print("Applying Random Over-sampling to balance the dataset...")
            ros = RandomOverSampler(random_state=42)
            X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
            print(f"Original dataset shape: {y_train.value_counts().to_dict()}")
            print(f"Resampled dataset shape: {y_train_balanced.value_counts().to_dict()}")
            
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            rf.fit(X_train_balanced, y_train_balanced)
        except (ImportError, ModuleNotFoundError, AttributeError):
            print("imbalanced-learn not found. Falling back to class_weight='balanced'.")
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, class_weight='balanced', random_state=42)
            rf.fit(X_train, y_train)
            class_balancing = "Class Weight (Balanced) (ROS fallback)"
            
    elif class_balancing == "Class Weight (Balanced)":
        print("Using class_weight='balanced' in RandomForestClassifier...")
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        
    else:
        # None
        print("Training model without class balancing...")
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC-AUC and ROC Curve
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Calculate Permutation Importance
    print("Calculating Permutation Importance...")
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    
    # Organize feature importance
    feature_names = X_train.columns.tolist()
    
    # 1. Random Forest Feature Importance (MDI)
    rf_importance = dict(zip(feature_names, rf.feature_importances_))
    
    # 2. Permutation Importance
    perm_imp_dict = dict(zip(feature_names, perm_importance.importances_mean))
    
    metrics = {
        "accuracy": acc,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1_score": report['weighted avg']['f1-score'],
        "roc_auc": roc_auc,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "confusion_matrix": cm.tolist(),
        "rf_importance": rf_importance,
        "permutation_importance": perm_imp_dict,
        "class_balancing": class_balancing
    }
    
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save model
    model_path = os.path.join(MODELS_DIR, 'rf_model.pkl')
    joblib.dump(rf, model_path, compress=3)
    print(f"Model saved to {model_path}")
    
    # Save training data sample for LIME/SHAP initialization
    joblib.dump(X_train.iloc[:100], os.path.join(MODELS_DIR, 'X_train_sample.pkl'))
    
    return rf, metrics

if __name__ == "__main__":
    train_model()
