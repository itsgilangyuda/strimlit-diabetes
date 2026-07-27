import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Aplikasi Prediksi Risiko Diabetes")
st.write("Masukkan data klinis pasien di bawah ini untuk memprediksi potensi risiko diabetes.")

# ---------------------------------------------------------
# 1. Load Data & Latih Model Sederhana (Atau Load Model Joblib)
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # Mengunduh/membaca dataset
    df = pd.read_csv("https://raw.githubusercontent.com/datasets/diabetes-prediction/main/diabetes_prediction_dataset.csv") 
    df.drop_duplicates(inplace=True)

    # Preprocessing
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])

    # One-hot encoding untuk smoking_history
    df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True, dtype=int)

    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Latih model Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, le_gender, X.columns

# Memuat model dan transformer
try:
    with st.spinner("Memuat model machine learning..."):
        model, scaler, le_gender, feature_cols = load_and_train_model()
except Exception as e:
    st.error(f"Gagal memuat dataset online: {e}")
    st.info("Pastikan Anda terhubung ke internet atau load file CSV/model lokal (.joblib).")
    st.stop()

# ---------------------------------------------------------
# 2. Form Input Data Pasien
# ---------------------------------------------------------
st.subheader("Data Pasien")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", options=["Female", "Male", "Other"])
    age = st.number_input("Umur (Tahun)", min_value=1.0, max_value=120.0, value=40.0, step=1.0)
    hypertension = st.selectbox("Riwayat Hipertensi", options=["Tidak", "Ya"])
    heart_disease = st.selectbox("Riwayat Penyakit Jantung", options=["Tidak", "Ya"])

with col2:
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hba1c = st.number_input("Tingkat HbA1c (%)", min_value=3.5, max_value=10.0, value=5.5, step=0.1)
    blood_glucose = st.number_input("Kadar Gula Darah (mg/dL)", min_value=50, max_value=300, value=100, step=1)
    smoking_history = st.selectbox(
        "Riwayat Merokok", 
        options=["never", "No Info", "former", "current", "not current", "ever"]
    )

# Konversi input biner
hypertension_val = 1 if hypertension == "Ya" else 0
heart_disease_val = 1 if heart_disease == "Ya" else 0

# ---------------------------------------------------------
# 3. Prediksi saat tombol diklik
# ---------------------------------------------------------
st.markdown("---")
if st.button("🔍 Analisis Risiko Diabetes", type="primary"):
    # Encode gender
    try:
        gender_encoded = le_gender.transform([gender])[0]
    except:
        gender_encoded = 0

    # Buat dictionary input sesuai struktur feature columns
    input_data = {col: 0 for col in feature_cols}
    
    input_data['gender'] = gender_encoded
    input_data['age'] = age
    input_data['hypertension'] = hypertension_val
    input_data['heart_disease'] = heart_disease_val
    input_data['bmi'] = bmi
    input_data['HbA1c_level'] = hba1c
    input_data['blood_glucose_level'] = blood_glucose

    # Handle One-Hot Encoded Column untuk smoking_history
    dummy_col_name = f"smoking_history_{smoking_history}"
    if dummy_col_name in input_data:
        input_data[dummy_col_name] = 1

    # Ubah ke DataFrame & Scaled
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Lakukan Prediksi
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Tampilkan Hasil
    st.subheader("Hasil Analisis")
    if prediction == 1:
        st.error(f"⚠️ **Risiko Tinggi Diabetes** (Probabilitas: {probability:.1f}%)")
        st.write("Pasien terindikasi memiliki karakteristik klinis yang berkaitan dengan diabetes. Disarankan untuk pemeriksaan lanjutan.")
    else:
        st.success(f"✅ **Risiko Rendah / Normal** (Probabilitas Risiko: {probability:.1f}%)")
        st.write("Parameter klinis pasien berada dalam rentang risiko yang rendah.")