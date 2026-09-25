import openai
import os
import sys

# Monkeypatch for Python < 3.10 to fix google-generativeai dependency issue
if sys.version_info < (3, 10):
    import importlib.metadata
    import importlib_metadata
    if not hasattr(importlib.metadata, 'packages_distributions'):
        importlib.metadata.packages_distributions = importlib_metadata.packages_distributions

import google.generativeai as genai

def generate_explanation_prompt(data, method, prediction_class):
    """
    Constructs a prompt for the LLM based on the XAI method and data.
    """
    if method == "LIME":
        # data is a list of tuples (feature_name, contribution)
        features_text = "\n".join([f"- {feat}: {val:.4f}" for feat, val in data])
        prompt = f"""
        You are an expert data scientist explaining a machine learning model's prediction to a non-technical user.
        
        The model predicts whether a person's income is >50K or <=50K.
        We are using LIME (Local Interpretable Model-Agnostic Explanations) to explain a specific prediction.
        
        Here are the top features and their contributions to the prediction (positive values support >50K, negative support <=50K):
        {features_text}
        
        Please provide a concise, easy-to-understand explanation of why the model made this prediction based on these features. 
        Focus on the most important factors. Avoid technical jargon where possible.
        """
    elif method == "SHAP":
        # data is a list of tuples (feature_name, shap_value)
        features_text = "\n".join([f"- {feat}: {val:.4f}" for feat, val in data])
        prompt = f"""
        You are an expert data scientist explaining a machine learning model's prediction to a non-technical user.
        
        The model predicts whether a person's income is >50K or <=50K.
        We are using SHAP (SHapley Additive exPlanations) to explain the contribution of each feature.
        
        Here are the feature contributions (SHAP values):
        {features_text}
        
        Positive values push the prediction towards >50K, negative values push towards <=50K.
        
        Please provide a concise, easy-to-understand explanation of the model's reasoning for this specific individual.
        Highlight which factors were the strongest drivers for the decision.
        """
    else:
        prompt = "Explain the model prediction."
        
    return prompt

def get_openai_explanation(prompt, api_key):
    """
    Generates explanation using OpenAI API.
    """
    if not api_key:
        return "Error: OpenAI API Key is missing."
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant explaining ML models."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def get_gemini_explanation(prompt, api_key):
    """
    Generates explanation using Google Gemini API.
    """
    if not api_key:
        return "Error: Google Gemini API Key is missing."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"
