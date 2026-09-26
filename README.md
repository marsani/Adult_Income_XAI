# 📊 Adult Income Prediction & Explainability (XAI)

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi apakah pendapatan seseorang melebihi **$50.000/tahun (>50K)** menggunakan **Random Forest Classifier**, dilengkapi dengan analisis **Explainable AI (XAI)** menggunakan **SHAP**, **LIME**, serta narasi penjelasan otomatis bertenaga **LLM (OpenAI / Google Gemini)**.

---

## ✨ Fitur Utama

1. **Eksplorasi Data & Training Model**:
   - Analisis performa model: Akurasi, Precision, Recall, F1-Score, Confusion Matrix, dan ROC-AUC Curve.
   - Penyesuaian Hyperparameter (*n_estimators, max_depth, min_samples_split*).
   - Opsi penanganan *imbalance data*: **SMOTE**, **Random Over-sampling**, **Class Weight (Balanced)**, atau **None**.
2. **Prediksi Interaktif**:
   - Input data personal (umur, jam kerja, pendidikan, pekerjaan, dll.) secara real-time.
   - Menghasilkan probabilitas prediksi kelas `<=50K` atau `>50K`.
3. **Explainable AI (XAI)**:
   - **SHAP (SHapley Additive exPlanations)**: Force Plot, Waterfall Plot, dan Summary Plot untuk kontribusi fitur.
   - **LIME (Local Interpretable Model-Agnostic Explanations)**: Penjelasan lokal berbasis bobot fitur per instance.
   - **Feature Importance**: Random Forest MDI & Permutation Importance.
4. **AI Narrative Explanations**:
   - Menjelaskan hasil SHAP & LIME dalam bahasa yang mudah dipahami menggunakan OpenAI ChatGPT atau Google Gemini.

---

## 📁 Struktur Proyek

```text
adult_income_xai/
│
├── app.py                      # File utama aplikasi Streamlit
├── requirements.txt            # Daftar dependensi Python
├── .gitignore                  # File konfigurasi file yang diabaikan git
│
├── .streamlit/
│   └── config.toml             # Konfigurasi tampilan & tema Streamlit
│
├── models/                     # Model tersimpan dan metadata
│   ├── rf_model.pkl            # Pre-trained Random Forest model
│   ├── encoders.pkl            # Label Encoders untuk variabel kategori
│   ├── feature_names.pkl       # Daftar nama fitur
│   ├── X_train_sample.pkl      # Sampel data training untuk XAI
│   └── metrics.json            # Metrik evaluasi model
│
└── src/                        # Modul pendukung
    ├── data_loader.py          # Script download & preprocessing dataset Adult UCI
    ├── model_trainer.py        # Script training & evaluasi model
    ├── explainers.py           # Inisialisasi explainer LIME & SHAP
    └── llm_utils.py            # Integrasi LLM (OpenAI & Gemini)
```

---

## 🚀 Panduan Menjalankan Secara Lokal

### 1. Clone Repository & Masuk ke Direktori
```bash
git clone https://github.com/<username-anda>/adult_income_xai.git
cd adult_income_xai
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)
```bash
# Menggunakan venv
python3 -m venv .venv

# Aktivasi di macOS/Linux:
source .venv/bin/activate

# Aktivasi di Windows (PowerShell/CMD):
.venv\Scripts\activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

---

## 📤 Panduan Upload ke GitHub

Jalankan perintah berikut di terminal pada direktori proyek:

```bash
# 1. Inisialisasi Git repository (jika belum)
git init

# 2. Tambahkan semua file
git add .

# 3. Buat commit pertama
git commit -m "Initial commit: Adult Income Prediction with XAI & Streamlit"

# 4. Ubah nama branch utama ke main
git branch -M main

# 5. Hubungkan ke repository GitHub Anda (buat repository kosong di github.com terlebih dahulu)
git remote add origin https://github.com/<USERNAME_GITHUB>/<NAMA_REPO>.git

# 6. Push kode ke GitHub
git push -u origin main
```

---

## ☁️ Panduan Deploy ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io) dan login menggunakan akun GitHub Anda.
2. Klik tombol **"New app"**.
3. Isi konfigurasi deployment:
   - **Repository**: Pilih repository GitHub Anda (misal: `username/adult_income_xai`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. (Opsional) Jika menggunakan API Key OpenAI/Gemini di server, Anda dapat memasukkannya di menu **Advanced settings -> Secrets**.
5. Klik **"Deploy!"**.
6. Aplikasi Anda siap diakses secara publik dan dibagikan! 🎉

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
- `openai`
- `google-generativeai`
