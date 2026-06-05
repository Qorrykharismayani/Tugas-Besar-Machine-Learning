import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ==========================================
# LOAD MODELS & ARTIFACTS
# ==========================================
@st.cache_resource
def load_svm_artifacts():
    with open('models/svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/svm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Model Prediksi:",
    ["🌲 Random Forest (HR Prediction)", "📈 Support Vector Machine (SVM)", "🔵 Model Lain (Coming Soon)"]
)

# ==========================================
# HALAMAN SVM
# ==========================================
if menu == "📈 Support Vector Machine (SVM)":
    st.title("📈 HR Analytics - Support Vector Machine")
    st.write("Aplikasi prediksi retensi karyawan menggunakan algoritma SVM yang telah dioptimasi.")

    svm_model, svm_scaler = load_svm_artifacts()

    # Menyusun layouting form input dengan container agar rapi
    with st.container():
        st.subheader("Form Input Data Karyawan")
        
        col1, col2 = st.columns(2)
        with col1:
            city_development_index = st.number_input("City Development Index", min_value=0.0, max_value=1.0, value=0.8)
            training_hours = st.number_input("Training Hours", min_value=1, max_value=500, value=50)
            # Tambahkan input fitur lainnya di sini...
            
        with col2:
            education_level = st.selectbox("Education Level", ["Graduate", "Undergraduate", "Masters", "Phd"])
            experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=5)
            # Tambahkan input fitur lainnya di sini...

    # Tombol Prediksi
    if st.button("Jalankan Prediksi SVM"):
        # 1. Ubah inputan menjadi DataFrame / bentuk array yang sesuai dengan training data
        # (Pastikan urutan kolom sama persis dengan X_train saat training notebook)
        input_data = pd.DataFrame([{
            'city_development_index': city_development_index,
            'training_hours': training_hours,
            # ... petakan semua fitur hasil encoding
        }])
        
        # 2. WAJIB RE-SCALE DATA INPUT BERDASARKAN SCALER TRAINING
        input_scaled = svm_scaler.transform(input_data)
        
        # 3. Prediksi
        prediction = svm_model.predict(input_scaled)[0]
        prediction_proba = svm_model.predict_proba(input_scaled)[0][1] # Probabilitas pindah kerja
        
        # Tampilkan Hasil dengan layout visual yang jelas
        st.write("---")
        st.subheader("Hasil Analisis Prediksi")
        
        if prediction == 1:
            st.error(f"⚠️ Karyawan diprediksi akan **Pindah Kerja** (Probabilitas: {prediction_proba:.2%})")
        else:
            st.success(f"✅ Karyawan diprediksi akan **Bertahan** (Probabilitas Bertahan: {(1 - prediction_proba):.2%})")