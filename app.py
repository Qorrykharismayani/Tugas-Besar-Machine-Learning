import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2d2d4e;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #3d3d6b;
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #6c63ff;
    }
    [data-testid="stMetricLabel"] {
        color: #8888aa !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    [data-testid="stTab"] {
        color: #8888aa;
        font-weight: 500;
    }
    [data-testid="stTab"][aria-selected="true"] {
        color: #6c63ff !important;
        border-bottom: 2px solid #6c63ff !important;
    }

    /* Buttons */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
    }
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.5);
    }

    /* Input fields */
    [data-testid="stSelectbox"], [data-testid="stNumberInput"], [data-testid="stSlider"] {
        background: transparent;
    }
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: #1e1e3a !important;
        border: 1px solid #3d3d6b !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6c63ff;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #2d2d4e;
    }

    /* Result boxes */
    .result-churn {
        background: linear-gradient(135deg, #2d1b1b 0%, #3d1f1f 100%);
        border: 1px solid #ff4757;
        border-left: 4px solid #ff4757;
        padding: 20px 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1em;
        font-weight: 600;
        color: #ff6b6b;
        margin-top: 16px;
    }
    .result-stay {
        background: linear-gradient(135deg, #1b2d1b 0%, #1f3d1f 100%);
        border: 1px solid #2ed573;
        border-left: 4px solid #2ed573;
        padding: 20px 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1em;
        font-weight: 600;
        color: #7bed9f;
        margin-top: 16px;
    }

    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
        border: 1px solid #3d3d6b;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-card-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6c63ff;
        margin-bottom: 6px;
    }
    .info-card-value {
        font-size: 0.95rem;
        color: #e0e0f0;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    /* Warning/info boxes */
    .stWarning, .stInfo {
        border-radius: 10px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: #2d2d4e;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    models = {}
    try:
        models['rf'] = {
            'model'   : joblib.load('models/rf_model.pkl'),
            'features': joblib.load('models/rf_selected_features.pkl'),
            'metadata': joblib.load('models/rf_metadata.pkl'),
            'params'  : json.load(open('models/best_params_rf.json')),
            'scaler'  : joblib.load('models/rf_scaler.pkl'),
            'le_dict' : joblib.load('models/rf_label_encoders.pkl'),
        }
    except Exception as e:
        models['rf'] = None

    try:
        models['knn'] = {
            'model'       : joblib.load('models/knn_model.pkl'),
            'preprocessor': joblib.load('models/knn_preprocessor.pkl'),
            'metadata'    : joblib.load('models/knn_metadata.pkl'),
            'params'      : json.load(open('models/best_params_knn.json')),
        }
    except:
        models['knn'] = None

    try:
        models['dt'] = {
            'model'       : joblib.load('models/dt_model.pkl'),
            'preprocessor': joblib.load('models/dt_preprocessor.pkl'),
            'features'    : joblib.load('models/dt_selected_features.pkl'),
            'metadata'    : joblib.load('models/dt_metadata.pkl'),
            'params'      : json.load(open('models/best_params_dt.json'))
        }
    except:
        models['dt'] = None

    try:
        models['svm'] = {
            'model'   : joblib.load('models/svm_model.pkl'),
            'scaler'  : joblib.load('models/svm_scaler.pkl'),
            'features': joblib.load('models/svm_feature_names.pkl'),
            'metadata': joblib.load('models/svm_model_metadata.pkl'),
            'params'  : json.load(open('models/best_params_svm.json')),
        }
    except:
        models['svm'] = None

    return models

models = load_all_models()

# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def get_model_status(key):
    if models.get(key):
        return "🟢 Ready"
    return "🔴 Not loaded"

def show_prediction_result(pred, proba, threshold):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Prob. Churn",   f"{proba[1]*100:.1f}%")
    col_b.metric("🟢 Prob. Stay",    f"{proba[0]*100:.1f}%")
    col_c.metric("📌 Threshold",     f"{threshold}")

    if pred == 1:
        st.markdown(
            '<div class="result-churn">🚨 Pelanggan ini diprediksi akan <strong>CHURN</strong> — perlu tindakan retensi segera</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-stay">✅ Pelanggan ini diprediksi akan <strong>BERTAHAN</strong> — tidak ada tindakan mendesak</div>',
            unsafe_allow_html=True
        )

    # Probability gauge
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor('#1e1e3a')
    ax.set_facecolor('#1e1e3a')

    bar_color = '#ff4757' if pred == 1 else '#2ed573'
    ax.barh([''], [proba[1]], color=bar_color, height=0.5, zorder=3)
    ax.barh([''], [proba[0]], left=[proba[1]], color='#2d2d4e', height=0.5, zorder=3)
    ax.axvline(x=threshold, color='#f9ca24', linestyle='--',
               linewidth=2, zorder=4, label=f'Threshold ({threshold})')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilitas', color='#8888aa', fontsize=9)
    ax.tick_params(colors='#8888aa')
    ax.spines[:].set_color('#3d3d6b')
    ax.legend(loc='upper right', fontsize=8,
              facecolor='#1e1e3a', edgecolor='#3d3d6b', labelcolor='white')

    for spine in ax.spines.values():
        spine.set_color('#3d3d6b')
    ax.tick_params(axis='x', colors='#8888aa')
    ax.tick_params(axis='y', colors='#8888aa')

    ax.text(proba[1]/2, 0, f'{proba[1]*100:.1f}%',
            ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════
def input_form():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">👤 Info Pelanggan</div>', unsafe_allow_html=True)
        gender         = st.selectbox("Gender", ['Male', 'Female'])
        senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
        partner        = st.selectbox("Partner", ['Yes', 'No'])
        dependents     = st.selectbox("Dependents", ['No', 'Yes'])
        tenure         = st.slider("Tenure (bulan)", 0, 72, 12,
                                   help="Berapa lama pelanggan telah berlangganan")

    with col2:
        st.markdown('<div class="section-header">📞 Layanan</div>', unsafe_allow_html=True)
        phone_service    = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines   = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        online_security  = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
        online_backup    = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        device_protection= st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
        tech_support     = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
        streaming_tv     = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])

    with col3:
        st.markdown('<div class="section-header">💳 Billing</div>', unsafe_allow_html=True)
        contract        = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless       = st.selectbox("Paperless Billing", ['Yes', 'No'])
        payment_method  = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0,
                                          help="Tagihan bulanan pelanggan")
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0,
                                          help="Total tagihan selama berlangganan")

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

def show_model_info(model_key, perf_data, tech_data):
    meta   = models[model_key]['metadata']
    params = models[model_key]['params']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚙️ Hyperparameter Terbaik")
        st.dataframe(
            pd.DataFrame(params.items(), columns=['Parameter', 'Value']),
            use_container_width=True, hide_index=True
        )
    with col2:
        st.markdown("#### 📈 Performa Model")
        st.dataframe(
            pd.DataFrame(perf_data),
            use_container_width=True, hide_index=True
        )

    st.markdown("#### 🔧 Teknik yang Digunakan")
    cols = st.columns(len(tech_data))
    for i, (title, desc) in enumerate(tech_data.items()):
        cols[i].info(f"**{title}**\n\n{desc}")

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════
def preprocess_rf(data):
    df = pd.DataFrame([data])
    df['TotalCharges_missing'] = df['TotalCharges'].isnull().astype(int)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    le_dict  = models['rf']['le_dict']
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        if col in le_dict:
            try:
                df[col] = le_dict[col].transform(df[col].astype(str))
            except:
                df[col] = 0

    df['loyal_monthly']    = ((df['tenure'] > 24) & (df['Contract'] == 0)).astype(int)
    df['charge_per_tenure']= df['MonthlyCharges'] / (df['tenure'] + 1)

    service_cols = ['PhoneService','MultipleLines','OnlineSecurity',
                    'OnlineBackup','DeviceProtection','TechSupport',
                    'StreamingTV','StreamingMovies']
    df['total_services'] = df[service_cols].sum(axis=1)

    num_cols = ['tenure','MonthlyCharges','TotalCharges','charge_per_tenure','total_services']
    df[num_cols] = models['rf']['scaler'].transform(df[num_cols])

    df['new_high_charge'] = ((df['tenure'] < 0) & (df['MonthlyCharges'] > 0)).astype(int)
    df['no_protection']   = ((df['OnlineSecurity']==0)&(df['OnlineBackup']==0)&(df['DeviceProtection']==0)).astype(int)
    df['no_support']      = ((df['TechSupport']==0)&(df['StreamingTV']==0)&(df['StreamingMovies']==0)).astype(int)
    df['easy_cancel']     = ((df['Contract']==0)&(df['PaperlessBilling']==1)).astype(int)
    df['charge_ratio']    = df['TotalCharges'] / (df['MonthlyCharges'] + 0.001)

    return df[models['rf']['features']]

def preprocess_svm(data):
    df = pd.DataFrame([data])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=models['svm']['features'], fill_value=0)
    return models['svm']['scaler'].transform(df_encoded)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size: 2.5rem;'>📡</div>
        <div style='font-size: 1.2rem; font-weight: 700; color: white; margin-top: 8px;'>Telco Churn</div>
        <div style='font-size: 0.75rem; color: #8888aa; margin-top: 4px;'>Prediction Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:8px;">Pilih Model</div>', unsafe_allow_html=True)

    page = st.selectbox("", [
        "🌲 Random Forest",
        "🔵 KNN",
        "🌳 Decision Tree",
        "⚡ SVM",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Model status
    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:10px;">Status Model</div>', unsafe_allow_html=True)
    status_map = {
        '🌲 Random Forest': 'rf',
        '🔵 KNN'          : 'knn',
        '🌳 Decision Tree': 'dt',
        '⚡ SVM'          : 'svm',
    }
    for label, key in status_map.items():
        status = "🟢" if models.get(key) else "🔴"
        st.markdown(f'<div style="font-size:0.85rem;color:#c0c0d0;padding:3px 0;">{status} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Active model info
    active_key = status_map.get(page)
    if active_key and models.get(active_key):
        meta = models[active_key]['metadata']
        st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:10px;">Model Aktif</div>', unsafe_allow_html=True)
        st.metric("ROC-AUC",  f"{meta['auc']:.4f}")
        st.metric("Macro F1", f"{meta['macro_f1']:.4f}" if isinstance(meta['macro_f1'], float) else meta['macro_f1'])
        if 'threshold' in meta:
            st.metric("Threshold", f"{meta['threshold']}")
        st.metric("Features", f"{meta['n_features']}")

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#555577;text-align:center;">Dataset: Telco Customer Churn<br>IBM © 2024</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
model_labels = {
    '🌲 Random Forest': ('Random Forest', '🌲', '#6c63ff'),
    '🔵 KNN'          : ('K-Nearest Neighbors', '🔵', '#00b4d8'),
    '🌳 Decision Tree': ('Decision Tree', '🌳', '#2ed573'),
    '⚡ SVM'          : ('Support Vector Machine', '⚡', '#f9ca24'),
}
model_name, model_icon, model_color = model_labels[page]

st.markdown(f"""
<div style='padding: 24px 0 8px 0;'>
    <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;
                letter-spacing:0.12em;color:{model_color};margin-bottom:6px;'>
        {model_icon} {model_name}
    </div>
    <h1 style='font-size:2rem;font-weight:700;color:white;margin:0;line-height:1.2;'>
        Telco Customer Churn Prediction
    </h1>
    <p style='color:#8888aa;margin-top:8px;font-size:0.95rem;'>
        Prediksi apakah pelanggan akan berhenti berlangganan berdasarkan data historis.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════
if "Random Forest" in page:
    if not models['rf']:
        st.error("❌ Model Random Forest tidak ditemukan. Pastikan file `.pkl` ada di folder `models/`.")
    else:
        tab1, tab2, tab3 = st.tabs(["🔮  Prediksi", "📊  Model Info", "📋  Panduan"])

        with tab1:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input   = preprocess_rf(input_data)
                        proba     = models['rf']['model'].predict_proba(X_input)[0]
                        threshold = models['rf']['metadata'].get('threshold', 0.5)
                        pred      = int(proba[1] >= threshold)
                        st.markdown("---")
                        st.markdown("#### 📊 Hasil Prediksi")
                        show_prediction_result(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

        with tab2:
            show_model_info('rf',
                perf_data={
                    'Metrik': ['ROC-AUC','Macro F1','F1 Churn','Recall Churn','Threshold','Total Fitur'],
                    'Nilai' : ['0.8357','0.7350','0.59','0.6791','0.56','28']
                },
                tech_data={
                    'SMOTE'             : 'Handle class imbalance 73:27 dengan oversampling sintetis',
                    'Feature Engineering': '8 fitur baru dari domain knowledge Telco',
                    'Optuna Tuning'     : 'TPE Sampler, 50 trials untuk hyperparameter optimal',
                    'Threshold Opt.'    : 'Threshold digeser dari 0.5 → 0.56 untuk Macro F1 optimal',
                }
            )

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### 📖 Cara Menggunakan
                1. Isi semua input data pelanggan di tab **Prediksi**
                2. Klik tombol **Jalankan Prediksi**
                3. Lihat hasil prediksi dan probabilitasnya

                #### 🎯 Interpretasi Hasil
                | Probabilitas Churn | Interpretasi |
                |---|---|
                | 0% – 30% | Sangat likely bertahan |
                | 30% – 56% | Cenderung bertahan |
                | > 56% | Diprediksi akan churn |
                """)
            with col2:
                st.markdown("""
                #### 📦 Tentang Dataset
                - **Sumber**: IBM Telco Customer Churn
                - **Total data**: 7.043 pelanggan
                - **Fitur**: 19 input → 28 setelah engineering
                - **Target**: Churn (Yes/No)
                - **Imbalance**: 73% No Churn, 27% Churn

                #### 🔬 Tentang Model
                - **Algoritma**: Random Forest Classifier
                - **Teknik**: SMOTE + FE + Optuna
                - **AUC**: 0.8357 | **F1**: 0.7350
                """)

# ══════════════════════════════════════════════════════════════
#  PAGE: KNN
# ══════════════════════════════════════════════════════════════
elif "KNN" in page:
    if not models['knn']:
        st.warning("⚠️ Model KNN belum tersedia. Pastikan file model sudah di-push ke folder `models/`.")
    else:
        tab1, tab2 = st.tabs(["🔮  Prediksi", "📊  Model Info"])

        with tab1:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input = models['knn']['preprocessor'].transform(pd.DataFrame([input_data]))
                        proba   = models['knn']['model'].predict_proba(X_input)[0]
                        pred    = models['knn']['model'].predict(X_input)[0]
                        threshold = models['knn']['metadata'].get('threshold', 0.5)
                        st.markdown("---")
                        show_prediction_result(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

        with tab2:
            meta = models['knn']['metadata']
            show_model_info('knn',
                perf_data={
                    'Metrik': ['ROC-AUC','Macro F1','Total Fitur'],
                    'Nilai' : [meta['auc'], meta['macro_f1'], meta['n_features']]
                },
                tech_data={
                    'SMOTE'         : 'Handle class imbalance dengan oversampling',
                    'StandardScaler': 'Scaling wajib untuk KNN agar jarak antar fitur seimbang',
                    'Optuna Tuning' : 'Pencarian n_neighbors dan parameter optimal',
                }
            )

# ══════════════════════════════════════════════════════════════
#  PAGE: DECISION TREE
# ══════════════════════════════════════════════════════════════
elif "Decision Tree" in page:
    if not models['dt']:
        st.warning("⚠️ Model Decision Tree belum tersedia. Pastikan file model sudah di-push ke folder `models/`.")
    else:
        tab1, tab2 = st.tabs(["🔮  Prediksi", "📊  Model Info"])

        with tab1:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input   = models['dt']['preprocessor'].transform(pd.DataFrame([input_data]))
                        proba     = models['dt']['model'].predict_proba(X_input)[0]
                        pred      = models['dt']['model'].predict(X_input)[0]
                        threshold = models['dt']['metadata'].get('threshold', 0.5)
                        st.markdown("---")
                        show_prediction_result(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

        with tab2:
            meta = models['dt']['metadata']
            show_model_info('dt',
                perf_data={
                    'Metrik': ['ROC-AUC','Macro F1','Total Fitur'],
                    'Nilai' : [meta['auc'], meta['macro_f1'], meta['n_features']]
                },
                tech_data={
                    'SMOTE'        : 'Handle class imbalance dengan oversampling',
                    'Pruning'      : 'max_depth dikontrol untuk mencegah overfitting',
                    'Optuna Tuning': 'Pencarian parameter pohon optimal',
                }
            )

# ══════════════════════════════════════════════════════════════
#  PAGE: SVM
# ══════════════════════════════════════════════════════════════
elif "SVM" in page:
    if not models['svm']:
        st.warning("⚠️ Model SVM belum tersedia. Pastikan file model sudah di-push ke folder `models/`.")
    else:
        tab1, tab2, tab3 = st.tabs(["🔮  Prediksi", "📊  Model Info", "📋  Panduan"])

        with tab1:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input   = preprocess_svm(input_data)
                        proba     = models['svm']['model'].predict_proba(X_input)[0]
                        threshold = models['svm']['metadata'].get('threshold', 0.63)
                        pred      = int(proba[1] >= threshold)
                        st.markdown("---")
                        show_prediction_result(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

        with tab2:
            meta = models['svm']['metadata']
            show_model_info('svm',
                perf_data={
                    'Metrik': ['ROC-AUC','Macro F1','Threshold','Total Fitur'],
                    'Nilai' : [meta['auc'], meta['macro_f1'], meta.get('threshold','—'), meta['n_features']]
                },
                tech_data={
                    'SMOTE'         : 'Handle class imbalance dengan oversampling sintetis',
                    'StandardScaler': 'Scaling wajib untuk SVM agar margin optimal',
                    'Kernel RBF'    : 'Kernel terbaik hasil GridSearchCV / Optuna',
                }
            )

        with tab3:
            st.markdown("""
            #### 📖 Cara Menggunakan
            1. Isi semua input data pelanggan di tab **Prediksi**
            2. Klik tombol **Jalankan Prediksi**
            3. Lihat hasil dan probabilitasnya

            #### 🎯 Interpretasi Hasil
            | Probabilitas Churn | Interpretasi |
            |---|---|
            | 0% – 30% | Sangat likely bertahan |
            | 30% – 63% | Cenderung bertahan |
            | > 63% | Diprediksi akan churn |
            """)