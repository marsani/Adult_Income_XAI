# 📊 Adult Income Prediction & Explainability (XAI)

An interactive web application built with **Streamlit** to predict whether an individual's income exceeds **$50,000/year (>50K)** using a **Random Forest Classifier**, equipped with comprehensive **Explainable AI (XAI)** analysis using **SHAP**, **LIME**, and automated natural language explanations powered by **LLMs (OpenAI ChatGPT / Google Gemini)**.

---

## ✨ Key Features

1. **Exploratory Data Analysis & Model Training**:
   - Model performance metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC Curve.
   - Hyperparameter tuning (*n_estimators, max_depth, min_samples_split*).
   - Class imbalance handling options: **SMOTE**, **Random Over-sampling**, **Class Weight (Balanced)**, or **None**.
2. **Interactive Prediction**:
   - Real-time input for individual demographic and employment features (age, workclass, education, hours per week, capital gain/loss, etc.).
   - Prediction probabilities for `<=50K` and `>50K` classes.
3. **Explainable AI (XAI)**:
   - **SHAP (SHapley Additive exPlanations)**: Force plots, waterfall plots, and summary plots for feature contributions.
   - **LIME (Local Interpretable Model-Agnostic Explanations)**: Local instance explanations highlighting positive/negative feature weights.
   - **Feature Importance**: Random Forest Mean Decrease in Impurity (MDI) and Permutation Importance.
4. **AI Narrative Explanations**:
   - Automatically interprets and explains complex SHAP & LIME results into clear, non-technical human language using OpenAI (GPT-3.5/GPT-4) or Google Gemini.

---

## 📁 Project Structure

```text
adult_income_xai/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
│
├── .streamlit/
│   └── config.toml             # Streamlit theme & server configuration
│
├── models/                     # Saved artifacts & metadata
│   ├── rf_model.pkl            # Pre-trained Random Forest model
│   ├── encoders.pkl            # Label Encoders for categorical features
│   ├── feature_names.pkl       # Feature names list
│   ├── X_train_sample.pkl      # Training sample for XAI baseline
│   └── metrics.json            # Model evaluation metrics
│
└── src/                        # Source modules
    ├── data_loader.py          # Download & preprocess UCI Adult dataset
    ├── model_trainer.py        # Model training, balancing & evaluation
    ├── explainers.py           # SHAP & LIME explainer setup
    └── llm_utils.py            # LLM API integration (OpenAI & Gemini)
```

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/adult_income_xai.git
cd adult_income_xai
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS/Linux:
source .venv/bin/activate

# Activate on Windows (Command Prompt / PowerShell):
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📤 Pushing to GitHub

Run the following commands in your project terminal:

```bash
# 1. Initialize Git repository (if not already initialized)
git init

# 2. Stage all files
git add .

# 3. Commit your changes
git commit -m "Initial commit: Adult Income Prediction with XAI & Streamlit"

# 4. Set default branch to main
git branch -M main

# 5. Link to your GitHub repository (create an empty repository on github.com first)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<REPO_NAME>.git

# 6. Push code to GitHub
git push -u origin main
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click the **"New app"** button.
3. Configure your deployment:
   - **Repository**: Select your GitHub repository (e.g., `your-username/adult_income_xai`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. *(Optional)* If you want to use OpenAI or Google Gemini API keys server-side, add them under **Advanced settings -> Secrets**.
5. Click **"Deploy!"**.
6. Your application is now live and publicly accessible! 🎉

---

## 📦 Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `seaborn`
- `matplotlib`
- `shap`
- `lime`
- `joblib`
- `requests`
- `openai`
- `google-generativeai`
- `importlib-metadata`

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
