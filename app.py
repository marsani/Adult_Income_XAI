import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
from src.explainers import get_lime_explanation, get_shap_values, rf_model, feature_names, update_model, X_train_sample, shap_explainer
from src.model_trainer import train_model
from src.llm_utils import get_openai_explanation, get_gemini_explanation, generate_explanation_prompt
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier

import os
import seaborn as sns
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load Encoders to display readable options
ENCODERS_PATH = os.path.join(MODELS_DIR, 'encoders.pkl')
encoders = joblib.load(ENCODERS_PATH)

# Load Metrics
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')
try:
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = None

# Helper to load raw data for EDA
@st.cache_data
def load_raw_data():
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    COLUMNS = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    df = pd.read_csv(DATA_URL, names=COLUMNS, skipinitialspace=True)
    initial_count = len(df)
    
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    processed_count = len(df)
    
    return df, initial_count, processed_count

st.set_page_config(page_title="Income XAI", layout="wide")

st.title("Income Prediction & Explainability (XAI)")
st.markdown("Predicting if income exceeds $50K/year using **Random Forest**, explained by **LIME** and **SHAP**.")

# Sidebar for User Input
st.sidebar.header("Model Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 250, 100, step=10)
max_depth = st.sidebar.slider("Max Depth", 1, 50, 20)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 25, 2)
class_balancing = st.sidebar.selectbox(
    "Class Balancing Technique",
    options=["SMOTE", "Random Over-sampling", "Class Weight (Balanced)", "None"],
    index=0,
    help="Teknik untuk menangani ketidakseimbangan kelas pada data target label (income >50K vs <=50K)."
)

if st.sidebar.button("Train New Model"):
    with st.spinner("Training model... Please wait."):
        new_model, new_metrics = train_model(n_estimators, max_depth, min_samples_split, class_balancing=class_balancing)
        update_model(new_model)
        metrics = new_metrics # Update local metrics variable
        st.sidebar.success("Model retrained successfully!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("AI Explanation Settings")
ai_provider = st.sidebar.selectbox("Select AI Provider", ["None", "OpenAI (ChatGPT)", "Google Gemini"])
api_key = ""
if ai_provider != "None":
    api_key = st.sidebar.text_input(f"Enter {ai_provider} API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("User Input Features")

def user_input_features():
    inputs = {}
    
    # Numerical Features
    inputs['age'] = st.sidebar.slider('Age', 17, 90, 30)
    inputs['fnlwgt'] = st.sidebar.number_input('Final Weight (fnlwgt)', value=200000)
    inputs['education-num'] = st.sidebar.slider('Education Num', 1, 16, 10)
    inputs['capital-gain'] = st.sidebar.number_input('Capital Gain', value=0)
    inputs['capital-loss'] = st.sidebar.number_input('Capital Loss', value=0)
    inputs['hours-per-week'] = st.sidebar.slider('Hours per Week', 1, 99, 40)
    
    # Categorical Features
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country']
    
    for col in categorical_cols:
        le = encoders[col]
        options = list(le.classes_)
        selected_option = st.sidebar.selectbox(col.replace('-', ' ').title(), options)
        inputs[col] = le.transform([selected_option])[0]
        
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()
input_df = input_df[feature_names]

# Main Tabs
tab_desc, tab_pred, tab_eda, tab_xai, tab_eval, tab_tuning = st.tabs(["Data Description", "Prediction", "EDA Analysis", "XAI (LIME & SHAP)", "Model Evaluation", "Hyperparameter Tuning"])

# --- Tab 1: Data Description ---
with tab_desc:
    st.subheader("Deskripsi Data (Data Description)")
    
    st.markdown("### Table 1. Snippet of the dataset")
    st.write("Berikut adalah 5 baris pertama dari dataset yang digunakan.")
    
    raw_df, initial_count, processed_count = load_raw_data()
    st.dataframe(raw_df.head())
    
    st.markdown("### Full Dataset")
    st.write("Berikut adalah tampilan keseluruhan dataset.")
    st.dataframe(raw_df)

    st.write(f"**Total Data:** {raw_df.shape[0]} baris, {raw_df.shape[1]} kolom")
    
    # --- Summary of Dataset Descriptive Statistics ---
    st.markdown("### Summary of Dataset Descriptive Statistics")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Dataset Awal", f"{initial_count:,}")
    col_stat2.metric("Dataset Setelah Preprocessing", f"{processed_count:,}")
    col_stat3.metric("Data Dihapus (Missing Values)", f"{initial_count - processed_count:,}")
    # -------------------------------------------------
    st.markdown("""
    **Keterangan Kolom:**
    - **age**: Umur responden.
    - **workclass**: Kelas pekerjaan (Private, Self-emp, dll).
    - **education**: Tingkat pendidikan terakhir.
    - **marital-status**: Status pernikahan.
    - **occupation**: Jenis pekerjaan.
    - **relationship**: Hubungan dalam keluarga.
    - **race**: Ras.
    - **sex**: Jenis kelamin.
    - **hours-per-week**: Jam kerja per minggu.
    - **native-country**: Negara asal.
    - **income**: Target prediksi (>50K atau <=50K).
    """)
    
    # Statistical Summary
    st.markdown("---")
    st.subheader("📊 Statistik Dataset")
    
    # Numerical features statistics
    st.markdown("### Statistik Fitur Numerik")
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    numerical_stats = raw_df[numerical_cols].describe().T
    numerical_stats['missing'] = raw_df[numerical_cols].isnull().sum()
    numerical_stats = numerical_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing']]
    
    # Round for better display
    numerical_stats = numerical_stats.round(2)
    st.dataframe(numerical_stats, use_container_width=True)
    
    # Categorical features statistics
    st.markdown("### Statistik Fitur Kategorikal")
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
    
    cat_stats_list = []
    for col in categorical_cols:
        unique_count = raw_df[col].nunique()
        most_common = raw_df[col].mode()[0] if len(raw_df[col]) > 0 else 'N/A'
        most_common_freq = raw_df[col].value_counts().iloc[0] if len(raw_df[col]) > 0 else 0
        most_common_pct = (most_common_freq / len(raw_df) * 100) if len(raw_df) > 0 else 0
        
        cat_stats_list.append({
            'Kolom': col,
            'Jumlah Kategori': unique_count,
            'Kategori Terbanyak': most_common,
            'Frekuensi': most_common_freq,
            'Persentase': f"{most_common_pct:.1f}%"
        })
    
    cat_stats_df = pd.DataFrame(cat_stats_list)
    st.dataframe(cat_stats_df, use_container_width=True)
    
    # Income distribution
    st.markdown("### Distribusi Target (Income)")
    income_dist = raw_df['income'].value_counts()
    income_dist_df = pd.DataFrame({
        'Income': income_dist.index,
        'Jumlah': income_dist.values,
        'Persentase': [f"{(v/len(raw_df)*100):.2f}%" for v in income_dist.values]
    })
    st.dataframe(income_dist_df, use_container_width=True)


# --- Tab 2: Prediction ---
with tab_pred:
    st.subheader("Prediksi Income (Income Prediction)")
    
    subtab_single, subtab_sample, subtab_batch = st.tabs([
        "Prediksi Tunggal (Single)", 
        "10 Data Awal & Prediksi",
        "Prediksi per Konsumen (Batch)"
    ])
    
    with subtab_single:
        st.write("Prediksi untuk satu data berdasarkan input manual.")
        # Move existing prediction logic here
        if st.button('Predict (Single)'):
            prediction = rf_model.predict(input_df)[0]
            probability = rf_model.predict_proba(input_df)[0][1]
            prob_low = rf_model.predict_proba(input_df)[0][0]
            
            st.markdown("### 📊 Prediction Results")
            
            # Visual probability gauge
            col_gauge, col_info = st.columns([2, 1])
            
            with col_gauge:
                st.markdown("#### Prediction Probability")
                
                # Create horizontal gauge chart
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Draw the probability bar
                ax.barh(0, prob_low, left=0, height=0.5, color='#4CAF50', label='<=50K')
                ax.barh(0, probability, left=prob_low, height=0.5, color='#F44336', label='>50K')
                
                # Add percentage labels
                if prob_low > 0.1:
                    ax.text(prob_low/2, 0, f'{prob_low:.1%}', ha='center', va='center', 
                           fontweight='bold', fontsize=12, color='white')
                if probability > 0.1:
                    ax.text(prob_low + probability/2, 0, f'{probability:.1%}', ha='center', va='center',
                           fontweight='bold', fontsize=12, color='white')
                
                # Add labels above
                ax.text(0, 0.6, '<=50K', ha='left', va='bottom', fontsize=10, fontweight='bold')
                ax.text(1, 0.6, '>50K', ha='right', va='bottom', fontsize=10, fontweight='bold')
                
                # Styling
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.3, 1)
                ax.axis('off')
                ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold', pad=20)
                
                st.pyplot(fig)
                plt.close()
            
            with col_info:
                st.markdown("#### Final Prediction")
                if prediction == 1:
                    st.error("### >50K")
                    st.metric("Confidence", f"{probability:.2%}")
                else:
                    st.success("### <=50K")
                    st.metric("Confidence", f"{prob_low:.2%}")
            
            # Feature values table
            st.markdown("---")
            st.markdown("#### 📋 Input Feature Values")
            
            # Create a readable feature table
            feature_display = []
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                              'relationship', 'race', 'sex', 'native-country']
            
            # Get original input before encoding
            for col in input_df.columns:
                value = input_df[col].values[0]
                
                # Decode categorical features back to readable format
                if col in categorical_cols and col in encoders:
                    try:
                        readable_value = encoders[col].inverse_transform([int(value)])[0]
                    except:
                        readable_value = value
                else:
                    readable_value = value
                
                feature_display.append({
                    'Feature': col.replace('-', ' ').title(),
                    'Value': readable_value
                })
            
            # Display in two columns for better layout
            col_left, col_right = st.columns(2)
            mid_point = len(feature_display) // 2
            
            with col_left:
                for item in feature_display[:mid_point]:
                    if isinstance(item['Value'], (int, float)):
                        st.markdown(f"**{item['Feature']}**: `{item['Value']:.0f}`")
                    else:
                        st.markdown(f"**{item['Feature']}**: `{item['Value']}`")
            
            with col_right:
                for item in feature_display[mid_point:]:
                    if isinstance(item['Value'], (int, float)):
                        st.markdown(f"**{item['Feature']}**: `{item['Value']:.0f}`")
                    else:
                        st.markdown(f"**{item['Feature']}**: `{item['Value']}`")
                
            st.info("💡 Go to the **XAI** tab to understand why this prediction was made.")

    with subtab_sample:
        st.write("Menampilkan 10 data awal dari dataset beserta hasil prediksi untuk analisa.")
        
        # Load first 10 samples from raw data
        sample_size = 10
        sample_raw = raw_df.head(sample_size).copy()
        
        # Preprocess for prediction (Encoding)
        sample_encoded = sample_raw.copy()
        
        # Store actual income if exists for comparison
        actual_income = None
        if 'income' in sample_encoded.columns:
            actual_income = sample_raw['income'].values
            sample_encoded = sample_encoded.drop('income', axis=1)
        
        # Encode categorical columns
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                            'relationship', 'race', 'sex', 'native-country']
        
        for col in categorical_cols:
            le = encoders[col]
            sample_encoded[col] = le.transform(sample_encoded[col])
        
        # Ensure columns order
        sample_encoded = sample_encoded[feature_names]
        
        # Predict
        preds = rf_model.predict(sample_encoded)
        probs = rf_model.predict_proba(sample_encoded)[:, 1]
        
        # Prepare display dataframe with selected key features
        display_df = pd.DataFrame({
            'No': range(1, sample_size + 1),
            'Age': sample_raw['age'].values,
            'Education': sample_raw['education'].values,
            'Occupation': sample_raw['occupation'].values,
            'Hours/Week': sample_raw['hours-per-week'].values,
            'Prediction': ['>50K' if p == 1 else '<=50K' for p in preds],
            'Probability (>50K)': [f"{p:.1%}" for p in probs]
        })
        
        # Add actual income if available
        if actual_income is not None:
            display_df.insert(6, 'Actual Income', actual_income)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Show summary statistics
        st.markdown("### 📊 Ringkasan Prediksi")
        col1, col2, col3 = st.columns(3)
        
        pred_high = sum(preds == 1)
        pred_low = sample_size - pred_high
        avg_prob = np.mean(probs)
        
        col1.metric("Prediksi >50K", f"{pred_high} orang", f"{pred_high/sample_size:.0%}")
        col2.metric("Prediksi <=50K", f"{pred_low} orang", f"{pred_low/sample_size:.0%}")
        col3.metric("Rata-rata Probability", f"{avg_prob:.1%}")
        
        st.info("💡 Data ini adalah 10 data pertama dari dataset untuk memberikan gambaran hasil prediksi model.")

    with subtab_batch:
        st.write("Prediksi untuk banyak konsumen sekaligus (Contoh: 20 data pertama dari dataset).")
        
        if st.button('Generate Batch Predictions'):
            # Load a sample of raw data
            batch_size = 20
            sample_raw = raw_df.head(batch_size).copy()
            
            # Preprocess for prediction (Encoding)
            sample_encoded = sample_raw.copy()
            # Drop target if exists
            if 'income' in sample_encoded.columns:
                sample_encoded = sample_encoded.drop('income', axis=1)
                
            # Encode categorical columns
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                'relationship', 'race', 'sex', 'native-country']
            
            for col in categorical_cols:
                le = encoders[col]
                # Handle unknown labels if any (though for head(20) of training data it should be fine)
                # Using map/apply to be safe or just transform
                sample_encoded[col] = le.transform(sample_encoded[col])
            
            # Ensure columns order
            sample_encoded = sample_encoded[feature_names]
            
            # Predict
            preds = rf_model.predict(sample_encoded)
            probs = rf_model.predict_proba(sample_encoded)[:, 1]
            
            # Add results to display dataframe
            results_df = sample_raw.drop('income', axis=1, errors='ignore').copy()
            results_df['Prediction'] = ['>50K' if p == 1 else '<=50K' for p in preds]
            results_df['Probability (>50K)'] = [f"{p:.2%}" for p in probs]
            
            st.dataframe(results_df)

# --- Tab 3: EDA Analysis ---
with tab_eda:
    st.subheader("Exploratory Data Analysis (Analisis Data Eksploratif)")
    st.write("Visualisasi distribusi data latih untuk memahami karakteristik dataset.")
    
    # raw_df is already loaded in Tab 1, but scope might be local if not careful.
    # load_raw_data is cached so calling it again is cheap.
    raw_df_eda, _, _ = load_raw_data()
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.write("**Distribusi Target (Income)**")
        fig_target, ax_target = plt.subplots()
        sns.countplot(x='income', data=raw_df, ax=ax_target, palette='viridis', hue='income', legend=False)
        ax_target.set_title("Jumlah Data per Kategori Income")
        st.pyplot(fig_target)
        
    with col_eda2:
        st.write("**Distribusi Umur berdasarkan Income**")
        fig_age, ax_age = plt.subplots()
        sns.boxplot(x='income', y='age', data=raw_df, ax=ax_age, palette='coolwarm', hue='income', legend=False)
        ax_age.set_title("Sebaran Umur untuk Income <=50K vs >50K")
        st.pyplot(fig_age)
        
    st.write("**Hubungan Pendidikan dan Income**")
    fig_edu, ax_edu = plt.subplots(figsize=(12, 6))
    sns.countplot(y='education', hue='income', data=raw_df, ax=ax_edu, palette='Set2', 
                  order=raw_df['education'].value_counts().index)
    ax_edu.set_title("Jumlah Income >50K vs <=50K berdasarkan Tingkat Pendidikan")
    st.pyplot(fig_edu)
    
    st.write("**Correlation Heatmap (Fitur Numerik)**")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    numeric_df = raw_df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# --- Tab 3: XAI ---
with tab_xai:
    st.header("Analisis Explainability (XAI)")
    st.markdown("""
    Bagian ini menjelaskan **mengapa** model memberikan prediksi tersebut.
    - **LIME**: Menjelaskan prediksi secara lokal (fokus pada satu data ini saja).
    - **SHAP**: Menjelaskan kontribusi setiap fitur terhadap hasil prediksi (global & lokal).
    """)
    
    subtab_lime, subtab_shap = st.tabs(["Penjelasan LIME", "Penjelasan SHAP"])
    
    with subtab_lime:
        st.subheader("Local Interpretable Model-Agnostic Explanations (LIME)")
        st.info("LIME mencoba meniru cara kerja model Random Forest hanya di sekitar data input Anda untuk melihat fitur mana yang paling berpengaruh.")
        
        st.info("LIME mencoba meniru cara kerja model Random Forest hanya di sekitar data input Anda untuk melihat fitur mana yang paling berpengaruh.")
        
        # --- Data Selection for LIME ---
        st.markdown("#### Pilih Data untuk Analisis")
        data_source = st.radio("Sumber Data:", ["Input Manual (Sidebar)", "Pilih dari Dataset"], horizontal=True)
        
        selected_instance = None
        selected_display_df = None
        
        if data_source == "Input Manual (Sidebar)":
            selected_instance = input_df.values[0]
            selected_display_df = input_df
            st.write("Menggunakan data dari input sidebar.")
        else:
            # Need to access raw dataframe again securely
            raw_df_lime_select, _, _ = load_raw_data()
            max_idx = len(raw_df_lime_select) - 1
            selected_index = st.number_input(f"Pilih Indeks Data (0 - {max_idx})", min_value=0, max_value=max_idx, value=0, step=1)
            
            # Fetch and preprocess the selected row
            row_raw = raw_df_lime_select.iloc[[selected_index]].copy()
            
            # Display selected raw data
            st.write("Data Terpilih (Raw):")
            st.dataframe(row_raw)
            
            # Preprocess
            row_encoded = row_raw.copy()
            if 'income' in row_encoded.columns:
                row_encoded = row_encoded.drop('income', axis=1)
            
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                'relationship', 'race', 'sex', 'native-country']
            
            for col in categorical_cols:
                le = encoders[col]
                row_encoded[col] = le.transform(row_encoded[col])
            
            row_encoded = row_encoded[feature_names]
            selected_instance = row_encoded.values[0]
            selected_display_df = row_encoded
            
        if st.button("Generate LIME Explanation"):
            exp = get_lime_explanation(selected_instance)
            
            # --- Standard LIME HTML Report (Matches User Request) ---
            st.markdown("### LIME Detailed Report (Standard Output)")
            st.write("Visualisasi ini menunjukkan Probabilitas Prediksi, Kontribusi Fitur, dan Tabel Nilai Fitur.")
            components.html(exp.as_html(), height=800, scrolling=True)
            
            st.divider()
   

            # --- Existing Custom Visualization ---
            st.markdown("### Custom Visualization")
            
            col_l1, col_l2 = st.columns([2, 1])
            
            with col_l1:
                # Display Prediction Probabilities
                # Need to predict on the specific instance
                # RF expects 2D array
                instance_2d = selected_instance.reshape(1, -1)
                # But RF trained with DataFrame might expect DF with names
                # predict_proba_wrapper handles numpy array conversion if needed, but here we invoke model directly
                # Let's use the DF form for safety if available, otherwise numpy
                
                probs = rf_model.predict_proba(selected_display_df)[0]
                prob_low = probs[0]
                prob_high = probs[1]
                
                st.markdown("#### Prediction Probabilities")
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Probability <=50K", f"{prob_low:.2%}")
                col_p2.metric("Probability >50K", f"{prob_high:.2%}")
                
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
                st.caption("**Interpretasi Grafik:** Bar berwarna **Hijau** artinya fitur tersebut mendukung prediksi ke arah '>50K'. Bar **Merah** mendukung ke arah '<=50K'.")
            
            with col_l2:
                st.write("**Fitur Paling Berpengaruh**")
                lime_list = exp.as_list()
                
                # Prepare data for the new table
                table_data = []
                categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                  'relationship', 'race', 'sex', 'native-country']
                
                for condition, contribution in lime_list:
                    # Find which feature this condition belongs to
                    found_feature = None
                    for feature in feature_names:
                        # Simple check: if feature name is in the condition string
                        # Note: This might be fragile if feature names are substrings of others, 
                        # but for this dataset it should be fine.
                        if feature in condition:
                            found_feature = feature
                            break
                    
                    actual_value = "N/A"
                    if found_feature:
                        val = selected_display_df[found_feature].values[0]
                        # Decode if categorical
                        if found_feature in categorical_cols and found_feature in encoders:
                            try:
                                actual_value = encoders[found_feature].inverse_transform([int(val)])[0]
                            except:
                                actual_value = val
                        else:
                            actual_value = val
                            
                    # Format value
                    if isinstance(actual_value, (float, np.floating)):
                        val_str = f"{actual_value:.2f}"
                    else:
                        val_str = str(actual_value)
                        
                    table_data.append({
                        'Feature': condition,
                        'Value': val_str,
                        'Contribution': contribution
                    })
                
                lime_table_df = pd.DataFrame(table_data)

                # Apply color styling: positive = Orange, negative = Blue
                def highlight_row_lime(row):
                    val = row['Contribution']
                    # Colors from the user image/request
                    # Blue for negative (<= 0.00 in image seems blue)
                    # Orange for positive
                    if val > 0:
                        return ['background-color: #ff7f0e; color: white'] * len(row) # Orange
                    else:
                        return ['background-color: #1f77b4; color: white'] * len(row) # Blue
                
                # Display Feature, Value, and Contribution columns
                st.dataframe(lime_table_df.style.apply(highlight_row_lime, axis=1), 
                             use_container_width=True, 
                             column_config={
                                 "Contribution": st.column_config.NumberColumn(
                                     "Nilai Kontribusi",
                                     format="%.4f"
                                 )
                             })
                
                st.write("**Kesimpulan LIME:**")
                top_feature = lime_list[0]
                st.write(f"Fitur yang paling dominan adalah **{top_feature[0]}**. Nilai kontribusinya sebesar **{top_feature[1]:.4f}**.")
                
                if ai_provider != "None" and api_key:
                    if st.button("Explain with AI (LIME)"):
                        with st.spinner(f"Generating explanation using {ai_provider}..."):
                            prompt = generate_explanation_prompt(lime_list, "LIME", ">50K") # Assuming positive class focus
                            if ai_provider == "OpenAI (ChatGPT)":
                                explanation = get_openai_explanation(prompt, api_key)
                            else:
                                explanation = get_gemini_explanation(prompt, api_key)
                            
                            st.markdown("### AI Explanation")
                            st.write(explanation)
                elif ai_provider != "None" and not api_key:
                    st.warning("Please enter your API Key in the sidebar.")

        # --- Global LIME Analysis ---
        st.markdown("---")
        st.subheader("Global LIME Analysis (Feature Average Importance)")
        st.write("Analisis ini menjalankan LIME pada sekumpulan data acak untuk melihat fitur mana yang secara rata-rata paling penting (Feature Importance) dan seberapa sering fitur tersebut muncul sebagai fitur berpengaruh (Feature Frequency).")

        # UI Controls
        col_glob1, col_glob2 = st.columns([1, 2])
        with col_glob1:
            lime_sample_size = st.slider("Sample Size for Global LIME", min_value=10, max_value=100, value=20, step=10, help="Semakin besar sample size, semakin lama prosesnya.")

        if st.button("Generate Global LIME Analysis"):
            with st.spinner(f"Running LIME on {lime_sample_size} samples... This may take a while."):
                # 1. Load Random Samples
                # We reuse load_raw_data logic but need to ensure we get a fresh sample that matches model input format
                raw_df_lime, _, _ = load_raw_data()
                
                # Sample from raw_df
                if len(raw_df_lime) > lime_sample_size:
                    sample_indices = np.random.choice(raw_df_lime.index, size=lime_sample_size, replace=False)
                    batch_raw = raw_df_lime.loc[sample_indices].copy()
                else:
                    batch_raw = raw_df_lime.copy()
                
                # Preprocess batch (Encoding) - mirroring logic from prediction tab
                batch_encoded = batch_raw.copy()
                if 'income' in batch_encoded.columns:
                    batch_encoded = batch_encoded.drop('income', axis=1)
                    
                categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                    'relationship', 'race', 'sex', 'native-country']
                
                for col in categorical_cols:
                    le = encoders[col]
                    # Use map/apply to handle potential unseen labels safely or just transform
                    batch_encoded[col] = le.transform(batch_encoded[col])
                
                batch_encoded = batch_encoded[feature_names]
                batch_values = batch_encoded.values # Numpy array for LIME

                # 2. Run LIME on each instance and aggregate
                feature_importance_map = {f: [] for f in feature_names}
                feature_frequency_map = {f: 0 for f in feature_names}

                progress_bar = st.progress(0)
                
                for i, instance in enumerate(batch_values):
                    # Update progress
                    progress_bar.progress((i + 1) / lime_sample_size)
                    
                    # Get explanation
                    exp = get_lime_explanation(instance)
                    lime_list = exp.as_list()
                    
                    # Parse explanation
                    # lime_list is [(condition_string, weight), ...]
                    # We need to map condition_string back to feature name
                    
                    found_features_in_sample = set()
                    
                    for condition, weight in lime_list:
                        # Find feature name in condition string
                        matched_feature = None
                        for f_name in feature_names:
                            if f_name in condition:
                                matched_feature = f_name
                                break
                        
                        if matched_feature:
                            # Store absolute weight for importance
                            feature_importance_map[matched_feature].append(abs(weight))
                            found_features_in_sample.add(matched_feature)
                    
                    # Update frequency count
                    for f in found_features_in_sample:
                        feature_frequency_map[f] += 1
                
                progress_bar.empty()

                # 3. Prepare Data for Plotting
                plot_data = []
                for f in feature_names:
                    weights = feature_importance_map[f]
                    if weights:
                        # Use all collected weights for boxplot distribution
                        # And calculate frequency
                        plot_data.append({
                            'Feature': f,
                            'Weights': weights,
                            'Frequency': feature_frequency_map[f]
                        })
                
                # Sort features by median importance if possible, or just by name/predefined order
                # Let's sort by max frequency or median importance to make it look nice
                # Using median importance for sorting
                plot_data.sort(key=lambda x: np.median(x['Weights']) if x['Weights'] else 0, reverse=True)

                features_sorted = [x['Feature'] for x in plot_data]
                weights_sorted = [x['Weights'] for x in plot_data]
                freqs_sorted = [x['Frequency'] for x in plot_data]

                # 4. Generate Visualization
                fig, ax1 = plt.subplots(figsize=(14, 6))

                # Box Plot (Feature Importance) - Left Axis
                # We need to manually construct boxplot positions
                bp = ax1.boxplot(weights_sorted, patch_artist=True, positions=range(len(features_sorted)), showfliers=False)
                
                # Style Box Plot (Blue as per image)
                for patch in bp['boxes']:
                    patch.set_facecolor('#1f77b4') 
                
                ax1.set_ylabel('Feature importance', color='#1f77b4', fontsize=14)
                ax1.tick_params(axis='y', labelcolor='#1f77b4')
                ax1.set_xticks(range(len(features_sorted)))
                ax1.set_xticklabels(features_sorted, rotation=45, ha='right')

                # Line Plot (Feature Frequency) - Right Axis
                ax2 = ax1.twinx()
                ax2.plot(range(len(features_sorted)), freqs_sorted, color='#ff7f0e', marker='*', markersize=12, linestyle='-', label='Feature frequency')
                
                ax2.set_ylabel('Feature frequency', color='#ff7f0e', fontsize=14)
                ax2.tick_params(axis='y', labelcolor='#ff7f0e')
                
                # Align x-limits
                ax1.set_xlim(-0.5, len(features_sorted) - 0.5)
                
                # Add Grid
                ax1.grid(axis='x', linestyle='-', alpha=0.7)
                ax1.grid(axis='y', linestyle='-', alpha=0.3)
                
                plt.title("Feature Average Importance vs Feature Frequency (Global LIME)", fontsize=16)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                st.info("Boxplot (Biru) menunjukkan sebaran tingkat kepentingan fitur (Feature Importance). Garis (Oranye) menunjukkan seberapa sering fitur tersebut muncul (Frequency) dalam penjelasan.")

    with subtab_shap:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        st.info("SHAP menghitung 'nilai fair' kontribusi setiap fitur. Jika fitur mendorong prediksi naik (ke arah >50K), warnanya Merah. Jika mendorong turun (ke arah <=50K), warnanya Biru.")
        
        if 'shap_local' not in st.session_state:
            st.session_state.shap_local = False
        if 'shap_global' not in st.session_state:
            st.session_state.shap_global = False

        if st.button("Generate SHAP Explanation"):
            st.session_state.shap_local = True
            
        if st.session_state.shap_local:
            shap_vals, expected_val = get_shap_values(input_df)
            
            # Logic to extract Class 1 (Positive) values
            if isinstance(shap_vals, list):
                shap_val_class = shap_vals[1]
                base_val = expected_val[1]
            elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                shap_val_class = shap_vals[0, :, 1]
                base_val = expected_val[1]
            else:
                shap_val_class = shap_vals
                base_val = expected_val

            # Ensure base_val is scalar
            if isinstance(base_val, (np.ndarray, list)):
                if np.size(base_val) == 1:
                    base_val = np.array(base_val).item()
                else:
                    base_val = base_val[1]
            
            base_val = float(base_val)

            # Ensure shap_val_class is 1D
            if isinstance(shap_val_class, np.ndarray) and len(shap_val_class.shape) > 1:
                 shap_val_class = shap_val_class.flatten()

            st.write("**Force Plot (Visualisasi Gaya Dorong)**")
            st.write("Grafik ini menunjukkan tarik-menarik antara fitur yang menaikkan peluang (Merah) dan menurunkan peluang (Biru).")
            p = shap.force_plot(base_val, shap_val_class, input_df.iloc[0], matplotlib=False)
            shap_html = f"<head>{shap.getjs()}</head><body>{p.html()}</body>"
            components.html(shap_html, height=200)
            
            st.write("**Bar Plot (Kontribusi Fitur)**")
            st.write("Urutan fitur berdasarkan besarnya pengaruh (magnitude) terhadap prediksi ini.")
            fig_shap, ax = plt.subplots()
            shap_exp = shap.Explanation(values=shap_val_class, 
                                      base_values=base_val, 
                                      data=input_df.iloc[0].values, 
                                      feature_names=feature_names)
            shap.plots.bar(shap_exp, show=False)
            st.pyplot(fig_shap)
            
            st.write("**Waterfall Plot (Detail Kontribusi)**")
            st.write("Grafik ini menunjukkan bagaimana setiap fitur berkontribusi secara kumulatif dari nilai dasar (base value) ke nilai prediksi akhir.")
            fig_waterfall, ax_waterfall = plt.subplots()
            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(fig_waterfall)

            if ai_provider != "None" and api_key:
                if st.button("Explain with AI (SHAP)"):
                    with st.spinner(f"Generating explanation using {ai_provider}..."):
                        # Prepare data for prompt: list of (feature, value)
                        # shap_val_class is 1D array of values, feature_names is list
                        shap_data = list(zip(feature_names, shap_val_class))
                        # Sort by absolute value to send most important features
                        shap_data.sort(key=lambda x: abs(x[1]), reverse=True)
                        top_shap_data = shap_data[:10] # Top 10 features
                        
                        prompt = generate_explanation_prompt(top_shap_data, "SHAP", ">50K")
                        if ai_provider == "OpenAI (ChatGPT)":
                            explanation = get_openai_explanation(prompt, api_key)
                        else:
                            explanation = get_gemini_explanation(prompt, api_key)
                        
                        st.markdown("### AI Explanation")
                        st.write(explanation)
            elif ai_provider != "None" and not api_key:
                st.warning("Please enter your API Key in the sidebar.")
                
            # --- Example Cases with Feature Contributions ---
            st.markdown("---")
            st.subheader("Example Cases with Feature Contributions")
            st.write("Analisis kontribusi fitur untuk 5 data pertama dari dataset.")
            
            if st.button("Show Example Cases Analysis"):
                with st.spinner("Analyzing example cases..."):
                    # Load first 5 samples
                    example_raw, _, _ = load_raw_data()
                    example_raw = example_raw.head(5).copy()
                    
                    # Preprocess
                    example_encoded = example_raw.copy()
                    if 'income' in example_encoded.columns:
                        example_encoded = example_encoded.drop('income', axis=1)
                        
                    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                        'relationship', 'race', 'sex', 'native-country']
                    
                    for col in categorical_cols:
                        le = encoders[col]
                        example_encoded[col] = le.transform(example_encoded[col])
                    
                    example_encoded = example_encoded[feature_names]
                    
                    # Calculate SHAP
                    shap_vals_examples = shap_explainer.shap_values(example_encoded)
                    
                    # Handle binary classification
                    if isinstance(shap_vals_examples, list):
                        shap_vals_examples = shap_vals_examples[1]
                    elif len(np.array(shap_vals_examples).shape) == 3:
                        shap_vals_examples = shap_vals_examples[:, :, 1]
                        
                    # Create Summary Table
                    example_analysis = []
                    for i in range(len(example_encoded)):
                        vals = shap_vals_examples[i]
                        # Find top positive and negative features
                        top_pos_idx = np.argmax(vals)
                        top_neg_idx = np.argmin(vals)
                        
                        top_pos_feat = feature_names[top_pos_idx]
                        top_pos_val = vals[top_pos_idx]
                        
                        top_neg_feat = feature_names[top_neg_idx]
                        top_neg_val = vals[top_neg_idx]
                        
                        # Get prediction
                        pred_prob = rf_model.predict_proba(example_encoded.iloc[[i]])[0][1]
                        pred_label = ">50K" if pred_prob > 0.5 else "<=50K"
                        
                        example_analysis.append({
                            "No": i + 1,
                            "Prediction": pred_label,
                            "Prob (>50K)": f"{pred_prob:.2%}",
                            "Top Positive Feature": f"{top_pos_feat} (+{top_pos_val:.3f})",
                            "Top Negative Feature": f"{top_neg_feat} ({top_neg_val:.3f})"
                        })
                    
                    st.dataframe(pd.DataFrame(example_analysis))

        # --- Global SHAP ---
        st.markdown("---")
        st.subheader("Global Feature Importance (SHAP)")
        st.write("Grafik ini menunjukkan rata-rata dampak absolut setiap fitur terhadap prediksi model secara keseluruhan.")
        
        # Dynamic Controls
        col_shap1, col_shap2 = st.columns(2)
        with col_shap1:
            sample_size_shap = st.slider("Sample Size (Jumlah Data)", min_value=50, max_value=500, value=100, step=50)
        
        with col_shap2:
            use_filter = st.checkbox("Filter by Feature (Filter Data)")
            filter_col = None
            filter_val = None
            
            if use_filter:
                # Use categorical columns for filtering as they make more sense for grouping
                filter_col = st.selectbox("Pilih Fitur untuk Filter", 
                                        ['workclass', 'education', 'marital-status', 'occupation', 
                                         'relationship', 'race', 'sex', 'native-country', 'income'])
                
                # Get unique values for the selected column from raw data
                # We need to access raw_df. Since it's cached, we can call load_raw_data()
                raw_df_filter, _, _ = load_raw_data()
                unique_vals = raw_df_filter[filter_col].unique()
                filter_val = st.selectbox(f"Pilih Nilai untuk {filter_col}", unique_vals)

        if st.button("Generate Global SHAP Importance"):
            st.session_state.shap_global = True

        if st.session_state.shap_global:
            with st.spinner("Calculating global SHAP values..."):
                # 1. Load Data
                df_shap, _, _ = load_raw_data()
                
                # 2. Apply Filter
                if use_filter and filter_col and filter_val:
                    df_shap = df_shap[df_shap[filter_col] == filter_val]
                    if len(df_shap) == 0:
                        st.error(f"Tidak ada data dengan {filter_col} = {filter_val}")
                        st.stop()
                
                # 3. Sample Data
                if len(df_shap) > sample_size_shap:
                    df_shap_sample = df_shap.sample(n=sample_size_shap, random_state=42)
                else:
                    df_shap_sample = df_shap
                    st.warning(f"Jumlah data tersedia ({len(df_shap)}) lebih kecil dari sample size ({sample_size_shap}). Menggunakan semua data.")
                
                # 4. Preprocess (Encoding)
                # We need to encode the sample data to match the model's expected input
                shap_data_encoded = df_shap_sample.copy()
                
                # Drop target if present
                if 'income' in shap_data_encoded.columns:
                    shap_data_encoded = shap_data_encoded.drop('income', axis=1)
                
                categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                                    'relationship', 'race', 'sex', 'native-country']
                
                try:
                    for col in categorical_cols:
                        le = encoders[col]
                        # Use map/apply to handle potential unseen labels safely or just transform
                        # Here we assume the raw data matches the training data distribution mostly
                        shap_data_encoded[col] = le.transform(shap_data_encoded[col])
                    
                    # Ensure column order matches feature_names
                    shap_data_encoded = shap_data_encoded[feature_names]
                    
                    # 5. Calculate SHAP values
                    shap_vals_global = shap_explainer.shap_values(shap_data_encoded)
                    
                    # Handle binary classification
                    if isinstance(shap_vals_global, list):
                        shap_vals_global = shap_vals_global[1]
                    elif len(np.array(shap_vals_global).shape) == 3:
                        shap_vals_global = shap_vals_global[:, :, 1]
                    
                    # Create Explanation object for plotting
                    expected_val = shap_explainer.expected_value
                    if isinstance(expected_val, (list, np.ndarray)):
                        expected_val = expected_val[1]
                        
                    shap_explanation = shap.Explanation(
                        values=shap_vals_global,
                        base_values=expected_val,
                        data=shap_data_encoded.values,
                        feature_names=feature_names
                    )
                    
                    # Plot
                    fig_global, ax_global = plt.subplots()
                    shap.plots.bar(shap_explanation, show=False, max_display=15)
                    
                    title_suffix = ""
                    if use_filter:
                        title_suffix = f" (Filter: {filter_col}={filter_val})"
                    plt.title(f"Global Feature Importance{title_suffix}", fontsize=14)
                    
                    st.pyplot(fig_global)
                    
                    # Beeswarm Plot
                    st.write("**Beeswarm Plot (Sebaran Dampak Fitur)**")
                    st.write("Grafik ini menunjukkan bagaimana nilai fitur (tinggi/rendah) mempengaruhi prediksi. Titik merah berarti nilai fitur tinggi, biru berarti rendah. Posisi di kanan garis tengah berarti mendorong prediksi ke arah >50K.")
                    
                    fig_beeswarm, ax_beeswarm = plt.subplots()
                    shap.plots.beeswarm(shap_explanation, show=False, max_display=15)
                    st.pyplot(fig_beeswarm)

                    st.write(f"**Catatan:** Analisis dilakukan pada {len(df_shap_sample)} data sampel.")
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses data: {e}")

# --- Tab 4: Evaluation ---
with tab_eval:
    st.subheader("Evaluasi Model (Model Evaluation)")
    st.markdown("Berikut adalah performa model Random Forest pada data uji (20% dari total data).")
    
    if metrics:
        balancing_method = metrics.get("class_balancing", "SMOTE (Default/Legacy)")
        st.info(f"ℹ️ **Teknik Class Balancing yang digunakan:** {balancing_method}")
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col_m2.metric("Precision", f"{metrics['precision']:.2%}")
        col_m3.metric("Recall", f"{metrics['recall']:.2%}")
        col_m4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        col_m5.metric("ROC AUC", f"{metrics.get('roc_auc', 'N/A') if isinstance(metrics.get('roc_auc'), (int, float)) else metrics.get('roc_auc', 'N/A')}")
        
        # Handle formatting if it is a number
        if 'roc_auc' in metrics and isinstance(metrics['roc_auc'], (int, float)):
             col_m5.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
        else:
             col_m5.metric("ROC AUC", "N/A (Retrain Model)")
        
        st.markdown("### Laporan Klasifikasi Detail (Detailed Report)")
        st.write("Metrik performa untuk setiap kelas (<=50K dan >50K).")
        
        st.markdown("### Confusion Matrix")
        st.write("Visualisasi kebenaran prediksi model.")
        
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                        xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
            
        st.markdown("### ROC Curve")
        st.write("Visualisasi performa model pada berbagai threshold klasifikasi.")
        
        if 'roc_curve' in metrics:
            fpr = metrics['roc_curve']['fpr']
            tpr = metrics['roc_curve']['tpr']
            roc_auc = metrics.get('roc_auc', 0.5)
            
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        else:
            st.warning("ROC Curve data not available. Please train a new model.")
            
        st.markdown("---")
        st.subheader("Perbandingan Ranking Fitur (Feature Importance)")
        st.write("Membandingkan fitur mana yang paling penting menurut berbagai metode.")
        
        # 1. RF Feature Importance (MDI)
        rf_imp = pd.Series(metrics.get('rf_importance', {}), name='RF (MDI)')
        
        # 2. Permutation Importance
        perm_imp = pd.Series(metrics.get('permutation_importance', {}), name='Permutation')
        
        # 3. SHAP Global Importance (Mean |SHAP value|)
        # Calculate on the fly using the sample data
        # (X_train_sample and shap_explainer are imported globally)
        
        # Check if X_train_sample is DataFrame or array
        if isinstance(X_train_sample, pd.DataFrame):
             shap_data_sample = X_train_sample
        else:
             shap_data_sample = pd.DataFrame(X_train_sample, columns=feature_names)

        shap_vals_global = shap_explainer.shap_values(shap_data_sample)
        
        # Handle SHAP values shape (for binary classification, it might return list of arrays or array)
        if isinstance(shap_vals_global, list):
            # Take the positive class
            shap_vals_global = shap_vals_global[1]
        elif len(shap_vals_global.shape) == 3:
             shap_vals_global = shap_vals_global[:,:,1]
             
        # Calculate mean absolute SHAP values
        shap_imp_vals = np.abs(shap_vals_global).mean(axis=0)
        shap_imp = pd.Series(shap_imp_vals, index=feature_names, name='SHAP (Global)')
        
        # Combine into one DataFrame
        importance_df = pd.concat([rf_imp, perm_imp, shap_imp], axis=1)
        
        # Normalize to 0-1 range for fair comparison
        importance_df_norm = importance_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        # Sort by SHAP importance
        importance_df_norm = importance_df_norm.sort_values('SHAP (Global)', ascending=False)
        
        st.dataframe(importance_df_norm.style.background_gradient(cmap='Greens'), use_container_width=True)
        
        st.markdown("""
        **Keterangan Metode:**
        1. **RF (MDI)**: Importance bawaan Random Forest (Mean Decrease in Impurity). Bias ke fitur dengan kardinalitas tinggi.
        2. **Permutation**: Mengukur penurunan akurasi saat nilai fitur diacak. Lebih akurat untuk melihat dampak nyata fitur.
        3. **SHAP (Global)**: Rata-rata dampak absolut fitur terhadap prediksi. Konsisten dan memperhitungkan interaksi antar fitur.
        """)
        
        st.info("Catatan: LIME tidak disertakan dalam perbandingan global karena sifatnya yang lokal (per-instance) dan komputasi yang berat untuk agregasi global.")
        
        st.markdown("---")
        st.subheader("Penjelasan Metrik Evaluasi:")
        st.markdown("""
        1. **Accuracy (Akurasi)**: 
           - Seberapa sering model memprediksi dengan benar secara keseluruhan.
           
        2. **Precision (Presisi)**: 
           - Dari semua orang yang diprediksi berpenghasilan **>50K**, berapa persen yang **benar-benar** >50K?
           
        3. **Recall (Sensitivitas)**: 
           - Dari semua orang yang **sebenarnya** berpenghasilan **>50K**, berapa persen yang berhasil **ditemukan** oleh model?
           
        4. **F1 Score**: 
           - Rata-rata harmonis antara Precision dan Recall.
        """)
    else:
        st.warning("Metrics not found. Please train the model first.")

# --- Tab 5: Hyperparameter Tuning ---
with tab_tuning:
    st.subheader("Analysis Hyperparameter (Hyperparameter Tuning)")
    st.markdown("""
    Bagian ini membantu Anda menemukan nilai optimal untuk parameter model Random Forest.
    Sistem akan menguji berbagai nilai untuk setiap parameter dan menunjukkan mana yang memberikan akurasi terbaik.
    """)
    
    # Configuration controls for Hyperparameter Analysis
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        cv_folds = st.number_input("Cross-Validation Folds (cv)", min_value=2, max_value=10, value=3, step=1, help="Jumlah fold untuk Cross Validation.")
    with col_c2:
        tuning_random_state = st.number_input("Random State", value=42, step=1, help="Nilai seed acak untuk menjamin reproduksi hasil analisis.")
        
    st.info("💡 **Tips:** Untuk mendapatkan hasil optimal di **150** seperti pada contoh sebelumnya, Anda dapat mengatur **cv = 5** dan **Random State = 1**.")
    
    if st.button("Mulai Analisis Hyperparameter"):
        with st.spinner("Starting hyperparameter analysis... Please wait (this may take a few minutes)."):
            # Load Data
            # Use the same data loading logic as training
            from src.data_loader import load_and_preprocess_data
            X_train, X_test, y_train, y_test = load_and_preprocess_data()
            
            # Combine for cross-validation
            X_combined = pd.concat([X_train, X_test])
            y_combined = pd.concat([y_train, y_test])
            
            # Define parameters to test
            param_ranges = {
                "n_estimators": list(range(25, 251, 25)),
                "max_depth": list(range(5, 51, 5)),
                "min_samples_split": [2, 5, 10, 15, 20, 25]
            }
            
            best_params = {}
            results_storage = {} # Store results for tables
            
            # Create 3 columns for plots
            col_t1, col_t2, col_t3 = st.columns(3)
            cols = [col_t1, col_t2, col_t3]
            
            for i, (param_name, param_range) in enumerate(param_ranges.items()):
                # Use a base model with the custom random state
                rf = RandomForestClassifier(random_state=int(tuning_random_state))
                
                # Calculate validation curve with custom cv
                train_scores, test_scores = validation_curve(
                    rf, X_combined, y_combined, param_name=param_name, param_range=param_range,
                    cv=int(cv_folds), scoring="accuracy", n_jobs=-1
                )
                
                # Calculate mean and std
                test_mean = np.mean(test_scores, axis=1)
                
                # Find optimal value
                best_idx = np.argmax(test_mean)
                best_val = param_range[best_idx]
                best_score = test_mean[best_idx]
                best_params[param_name] = best_val
                
                # Store results
                results_storage[param_name] = {
                    'range': param_range,
                    'scores': test_mean,
                    'best_idx': best_idx
                }
                
                # Plot
                with cols[i]:
                    fig, ax = plt.subplots()
                    ax.plot(param_range, test_mean, marker='o', label="Validation Score", color='#1f77b4')
                    
                    # Highlight optimal point
                    ax.scatter([best_val], [best_score], color='red', s=100, zorder=5, label=f"Optimal: {best_val}")
                    
                    ax.set_title(f"Analysis {param_name}")
                    ax.set_xlabel(param_name)
                    ax.set_ylabel("Accuracy")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                    st.success(f"**Optimal {param_name}: {best_val}** (Acc: {best_score:.2%})")
            
            st.markdown("---")
            st.subheader("Rekomendasi Parameter Optimal")
            st.json(best_params)
            st.info("💡 Anda dapat menggunakan nilai-nilai ini di sidebar 'Model Hyperparameters' untuk melatih ulang model.")
            
            st.markdown("---")
            st.subheader("Detail Hasil Analisis")
            
            for param_name, data in results_storage.items():
                st.markdown(f"### Tabel {param_name}")
                
                # Create DataFrame
                df_results = pd.DataFrame({
                    'No': range(1, len(data['range']) + 1),
                    'Nilai Parameter': data['range'],
                    'Score Validation': data['scores'],
                    'Persentase Optimal': data['scores'] # Will format later
                })
                
                # Styling function
                def highlight_optimal(row):
                    if row['Nilai Parameter'] == best_params[param_name]:
                        return ['background-color: #d4edda; color: black'] * len(row) # Light green
                    return [''] * len(row)
                
                # Display table
                st.dataframe(
                    df_results.style.apply(highlight_optimal, axis=1).format({
                        'Score Validation': '{:.4f}',
                        'Persentase Optimal': '{:.2%}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
