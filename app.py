import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d3d5c;
    }
    [data-testid="stMetricLabel"] { color: #a0a0b0 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    .result-box {
        padding: 20px; border-radius: 10px; text-align: center;
        font-size: 1.2em; font-weight: bold; margin-top: 20px;
    }
    .stay  { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .leave { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    models = {}

    # Random Forest
    try:
        models['rf'] = {
            'model'    : joblib.load('models/random_forest_model.pkl'),
            'features' : joblib.load('models/selected_features.pkl'),
            'metadata' : joblib.load('models/model_metadata.pkl'),
            'params'   : json.load(open('models/best_params_rf.json')),
            'scaler'   : joblib.load('models/rf_scaler.pkl'),
            'le_dict'  : joblib.load('models/rf_label_encoders.pkl'),
        }
    except Exception as e:
        st.error(f"RF Error: {e}")
        models['rf'] = None

    # KNN 
    try:
        models['knn'] = {
            'model'       : joblib.load('models/knn_model.pkl'),
            'preprocessor': joblib.load('models/knn_preprocessor.pkl'),
            'metadata'    : joblib.load('models/knn_metadata.pkl'),
            'params'      : json.load(open('models/best_params_knn.json')),
        }
    except:
        models['knn'] = None

    # Decision Tree
    try:
        models['dt'] = {
            'model': joblib.load('models/dt_model.pkl'),
            'preprocessor': joblib.load('models/dt_preprocessor.pkl'),
            'features': joblib.load('models/dt_selected_features.pkl'),
            'metadata': joblib.load('models/dt_metadata.pkl'),
            'params': json.load(open('models/best_params_dt.json'))
        }
    except Exception as e:
        print(f"Error loading DT: {e}")
        models['dt'] = None

    # Model 4 (teman 3) — uncomment setelah file tersedia
    # try:
    #     models['svm'] = {
    #         'model'   : joblib.load('models/svm_model.pkl'),
    #         'metadata': joblib.load('models/svm_metadata.pkl'),
    #         'params'  : json.load(open('models/best_params_svm.json')),
    #     }
    # except:
    #     models['svm'] = None

    return models

models = load_all_models()

# ══════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════
def input_form():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Info Pelanggan**")
        gender         = st.selectbox("Gender", ['Male', 'Female'])
        senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
        partner        = st.selectbox("Partner", ['Yes', 'No'])
        dependents     = st.selectbox("Dependents", ['No', 'Yes'])
        tenure         = st.slider("Tenure (bulan)", 0, 72, 12)

    with col2:
        st.markdown("**📞 Layanan**")
        phone_service   = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines  = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        internet_service= st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
        online_backup   = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        device_protection=st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
        tech_support    = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
        streaming_tv    = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
        streaming_movies= st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])

    with col3:
        st.markdown("**💳 Billing**")
        contract        = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless       = st.selectbox("Paperless Billing", ['Yes', 'No'])
        payment_method  = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

    return {
        'gender'          : gender,
        'SeniorCitizen'   : 1 if senior_citizen == 'Yes' else 0,
        'Partner'         : partner,
        'Dependents'      : dependents,
        'tenure'          : tenure,
        'PhoneService'    : phone_service,
        'MultipleLines'   : multiple_lines,
        'InternetService' : internet_service,
        'OnlineSecurity'  : online_security,
        'OnlineBackup'    : online_backup,
        'DeviceProtection': device_protection,
        'TechSupport'     : tech_support,
        'StreamingTV'     : streaming_tv,
        'StreamingMovies' : streaming_movies,
        'Contract'        : contract,
        'PaperlessBilling': paperless,
        'PaymentMethod'   : payment_method,
        'MonthlyCharges'  : monthly_charges,
        'TotalCharges'    : total_charges,
    }

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING RF
# ══════════════════════════════════════════════════════════════
def preprocess_rf(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Missing flag
    df['TotalCharges_missing'] = df['TotalCharges'].isnull().astype(int)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Label Encoding
    le_dict = models['rf']['le_dict']
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        if col in le_dict:
            try:
                df[col] = le_dict[col].transform(df[col].astype(str))
            except:
                df[col] = 0

    # Feature Engineering
    df['loyal_monthly'] = (
        (df['tenure'] > 24) & (df['Contract'] == 0)
    ).astype(int)

    df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[service_cols].sum(axis=1)

    # Scaling
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'charge_per_tenure', 'total_services']
    scaler = models['rf']['scaler']
    df[num_cols] = scaler.transform(df[num_cols])

    # Feature Engineering setelah scaling
    df['new_high_charge'] = (
        (df['tenure'] < 0) & (df['MonthlyCharges'] > 0)
    ).astype(int)

    df['no_protection'] = (
        (df['OnlineSecurity'] == 0) &
        (df['OnlineBackup'] == 0) &
        (df['DeviceProtection'] == 0)
    ).astype(int)

    df['no_support'] = (
        (df['TechSupport'] == 0) &
        (df['StreamingTV'] == 0) &
        (df['StreamingMovies'] == 0)
    ).astype(int)

    df['easy_cancel'] = (
        (df['Contract'] == 0) & (df['PaperlessBilling'] == 1)
    ).astype(int)

    df['charge_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.001)

    return df[models['rf']['features']]

# ══════════════════════════════════════════════════════════════
#  SHOW PREDICTION
# ══════════════════════════════════════════════════════════════
def show_prediction(pred, proba, label_pos='Churn', label_neg='No Churn'):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(f"Probabilitas {label_pos}",  f"{proba[1]*100:.1f}%")
    col_b.metric(f"Probabilitas {label_neg}", f"{proba[0]*100:.1f}%")
    col_c.metric("Prediksi", f"🚨 {label_pos}" if pred == 1 else f"✅ {label_neg}")

    if pred == 1:
        st.markdown(f'<div class="result-box leave">⚠️ Pelanggan diprediksi akan CHURN</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box stay">✅ Pelanggan diprediksi akan BERTAHAN</div>',
                    unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([''], [proba[1]], color='tomato' if pred==1 else 'steelblue', height=0.4)
    ax.barh([''], [proba[0]], left=[proba[1]], color='#e9ecef', height=0.4)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilitas')
    ax.set_title('Distribusi Probabilitas')
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("📡 Telco Churn App")
    st.markdown("---")

    page = st.selectbox("📌 Pilih Model", [
        "🌲 Random Forest",
        "🔵 KNN",
        "🌳 Decision Tree",
        "🔥 Model 4 (Coming Soon)",
    ])

    st.markdown("---")
    if "Random Forest" in page and models['rf']:
        meta = models['rf']['metadata']
        st.markdown("### 📊 Info Model RF")
        st.metric("ROC-AUC",  f"{meta['auc']:.4f}")
        st.metric("Macro F1", f"{meta['macro_f1']:.4f}")
        st.metric("Threshold",f"{meta['threshold']}")
        st.metric("Features", f"{meta['n_features']}")

    elif "KNN" in page and models['knn']:
        meta = models['knn']['metadata']
        st.markdown("### 📊 Info Model KNN")
        st.metric("ROC-AUC",  f"{meta['auc']:.4f}")
        st.metric("Macro F1", f"{meta['macro_f1']}")
        st.metric("Features", f"{meta['n_features']}")
    
    elif "Decision Tree" in page and models['dt']:
        meta = models['dt']['metadata']
        st.markdown("### 📊 Info Model Decision Tree")
        st.metric("ROC-AUC",  f"{meta['auc']:.4f}")
        st.metric("Macro F1", f"{meta['macro_f1']:.4f}")
        st.metric("Features", f"{meta['n_features']}")

# ══════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════
if "Random Forest" in page:
    st.title("🌲 Random Forest — Telco Customer Churn")
    st.markdown("Prediksi apakah pelanggan akan **berhenti berlangganan (churn)** atau tidak.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Model Info", "📋 Panduan"])

    with tab1:
        st.subheader("Input Data Pelanggan")
        input_data = input_form()
        st.markdown("---")

        if st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary"):
            try:
                X_input   = preprocess_rf(input_data)
                proba     = models['rf']['model'].predict_proba(X_input)[0]
                threshold = models['rf']['metadata'].get('threshold', 0.5)
                pred      = int(proba[1] >= threshold)

                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                st.caption(f"📌 Threshold optimal: {threshold}")
                show_prediction(pred, proba)
            except Exception as e:
                st.error(f"Error prediksi: {e}")

    with tab2:
        st.subheader("📊 Informasi Model RF")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚙️ Hyperparameter Terbaik")
            st.dataframe(pd.DataFrame(
                models['rf']['params'].items(),
                columns=['Parameter', 'Value']
            ), use_container_width=True)

        with col2:
            st.markdown("### 📈 Performa")
            st.dataframe(pd.DataFrame({
                'Metrik' : ['ROC-AUC', 'Macro F1', 'F1 Churn',
                            'Recall Churn', 'Threshold', 'Total Fitur'],
                'Nilai'  : ['0.8357', '0.7350', '0.59',
                            '0.6791', '0.56', '28']
            }), use_container_width=True)

        st.markdown("### 🔧 Teknik yang Digunakan")
        c1, c2, c3 = st.columns(3)
        c1.info("**Oversampling**\nSMOTE untuk handle\nimbalance 73:27")
        c2.info("**Feature Engineering**\n5 fitur baru dari\ndomain knowledge Telco")
        c3.info("**Tuning**\nOptuna TPE Sampler\n50 trials")

    with tab3:
        st.markdown("""
        ### Cara Menggunakan
        1. Isi semua input data pelanggan di tab **Prediksi**
        2. Klik tombol **Prediksi Sekarang**
        3. Lihat hasil prediksi dan probabilitasnya

        ### Interpretasi Hasil
        | Probabilitas Churn | Interpretasi |
        |---|---|
        | 0% - 30% | Pelanggan sangat likely bertahan |
        | 30% - 56% | Pelanggan cenderung bertahan |
        | >56% | Pelanggan diprediksi akan churn |

        ### Tentang Dataset
        - **Sumber**: Telco Customer Churn (IBM)
        - **Total data**: 7.043 pelanggan
        - **Target**: Prediksi churn pelanggan
        - **Imbalance**: 73% No Churn, 27% Churn
        """)

# ══════════════════════════════════════════════════════════════
#  PAGE: KNN
# ══════════════════════════════════════════════════════════════
elif "KNN" in page:
    st.title("🔵 KNN — Telco Customer Churn")
    st.markdown("---")

    if models['knn'] is None:
        st.warning("⚠️ Model KNN belum tersedia. Teman perlu retrain dengan dataset Telco.")
    else:
        tab1, tab2 = st.tabs(["🔮 Prediksi", "📊 Model Info"])

        with tab1:
            st.subheader("Input Data Pelanggan")
            input_data = input_form()
            st.markdown("---")

            if st.button("🔮 Prediksi", use_container_width=True, type="primary"):
                try:
                    preprocessor = models['knn']['preprocessor']
                    X_input = preprocessor.transform(pd.DataFrame([input_data]))
                    proba   = models['knn']['model'].predict_proba(X_input)[0]
                    pred    = models['knn']['model'].predict(X_input)[0]
                    st.markdown("---")
                    show_prediction(pred, proba)
                except Exception as e:
                    st.error(f"Error: {e}")

        with tab2:
            meta = models['knn']['metadata']
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ⚙️ Hyperparameter")
                st.dataframe(pd.DataFrame(
                    models['knn']['params'].items(),
                    columns=['Parameter', 'Value']
                ), use_container_width=True)
            with col2:
                st.markdown("### 📈 Performa")
                st.dataframe(pd.DataFrame({
                    'Metrik': ['ROC-AUC', 'Macro F1', 'Total Fitur'],
                    'Nilai' : [meta['auc'], meta['macro_f1'], meta['n_features']]
                }), use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: DECISION TREE
# ══════════════════════════════════════════════════════════════
elif "Decision Tree" in page:
    st.title("🌳 Decision Tree — Telco Customer Churn")
    st.markdown("---")

    if models['dt'] is None:
        st.warning("⚠️ Model DT belum tersedia. Pastikan file model sudah ada di folder models/")
    else:
        tab1, tab2 = st.tabs(["🔮 Prediksi", "📊 Model Info"])

        with tab1:
            st.subheader("Input Data Pelanggan")
            input_data = input_form()
            st.markdown("---")

            if st.button("🔮 Prediksi", use_container_width=True, type="primary"):
                try:
                    preprocessor = models['dt']['preprocessor']
                    # Ubah input ke dataframe lalu transform
                    X_input = preprocessor.transform(pd.DataFrame([input_data]))
                    proba   = models['dt']['model'].predict_proba(X_input)[0]
                    pred    = models['dt']['model'].predict(X_input)[0]
                    st.markdown("---")
                    show_prediction(pred, proba)
                except Exception as e:
                    st.error(f"Error: {e}")

        with tab2:
            meta = models['dt']['metadata']
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ⚙️ Hyperparameter")
                st.dataframe(pd.DataFrame(
                    models['dt']['params'].items(),
                    columns=['Parameter', 'Value']
                ), use_container_width=True)
            with col2:
                st.markdown("### 📈 Performa")
                st.dataframe(pd.DataFrame({
                    'Metrik': ['ROC-AUC', 'Macro F1', 'Total Fitur'],
                    'Nilai' : [meta['auc'], meta['macro_f1'], meta['n_features']]
                }), use_container_width=True)         

# ══════════════════════════════════════════════════════════════
#  PAGE: COMING SOON
# ══════════════════════════════════════════════════════════════
else:
    st.title("🚧 Coming Soon")
    st.info("Halaman ini akan diisi oleh anggota kelompok lain.")
    st.markdown("""
    ### Yang perlu disiapkan:
    - Simpan model ke folder `models/` dengan nama yang jelas
    - Tambahkan blok `elif` baru di `app.py`
    - Push ke branch masing-masing
    """)