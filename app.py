import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
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
            'model'     : joblib.load('models/random_forest_model.pkl'),
            'features'  : joblib.load('models/selected_features.pkl'),
            'metadata'  : joblib.load('models/model_metadata.pkl'),
            'params'    : json.load(open('models/best_params_rf.json')),
        }
    except:
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

    # Slot model 3 & 4 (isi nama file sesuai teman)
    # try:
    #     models['dt'] = {
    #         'model'    : joblib.load('models/decision_tree_model.pkl'),
    #         'features' : joblib.load('models/dt_selected_features.pkl'),
    #         'metadata' : joblib.load('models/dt_metadata.pkl'),
    #         'params'   : json.load(open('models/best_params_dt.json')),
    #     }
    # except:
    #     models['dt'] = None

    return models

models = load_all_models()

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════

# ── Preprocessing RF ───────────────────────────────────────────
def preprocess_rf(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df.drop(columns=['city'], inplace=True, errors='ignore')

    flag_cols = ['gender', 'company_size', 'company_type',
                 'major_discipline', 'last_new_job', 'enrolled_university']
    for col in flag_cols:
        df[f'{col}_missing'] = (df[col] == 'Unknown').astype(int)

    df['job_hopper'] = (
        (df['last_new_job'].isin(['1', 'never'])) &
        (df['experience'].isin(['<1','1','2','3','4','5']))
    ).astype(int)

    df['high_value_candidate'] = (
        (df['experience'].isin(['10','11','12','13','14','15','>20'])) &
        (df['education_level'].isin(['Masters', 'Phd']))
    ).astype(int)

    df['is_fresher'] = (
        (df['experience'].isin(['<1', '1', '2'])) &
        (df['enrolled_university'] != 'no_enrollment')
    ).astype(int)

    df['mismatch_candidate'] = (
        (df['relevent_experience'] == 'No relevent experience') &
        (df['company_size'].isin(['<10', '10/49', '50-99', 'Unknown']))
    ).astype(int)

    df['ambitious'] = (
        (df['city_development_index'] < 0.75) &
        (df['training_hours'] > 50)
    ).astype(int)

    df['career_stagnant'] = (
        (df['last_new_job'].isin(['>4', '4'])) &
        (df['experience'].isin(['10','11','12','13','14','15','>20']))
    ).astype(int)

    exp_map = {'<1': 0.5, '>20': 21}
    exp_numeric = df['experience'].replace(exp_map)
    exp_numeric = pd.to_numeric(exp_numeric, errors='coerce').fillna(1)
    df['training_per_exp'] = df['training_hours'] / (exp_numeric + 1)

    df['city_tier'] = pd.cut(
        df['city_development_index'],
        bins=[0, 0.62, 0.78, 0.90, 1.0],
        labels=[0, 1, 2, 3]
    ).astype(int)

    ordinal_config = {
        'experience'     : ['<1','1','2','3','4','5','6','7','8','9','10',
                            '11','12','13','14','15','16','17','18','19','20','>20'],
        'last_new_job'   : ['never','1','2','3','4','>4'],
        'company_size'   : ['<10','10/49','50-99','100-500','500-999',
                            '1000-4999','5000-9999','10000+','Unknown'],
        'education_level': ['Primary School','High School','Graduate','Masters','Phd'],
    }
    for col, order in ordinal_config.items():
        enc = OrdinalEncoder(
            categories=[order],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        df[col] = enc.fit_transform(df[[col]])

    nominal_cols = ['gender', 'relevent_experience', 'enrolled_university',
                    'major_discipline', 'company_type']
    for col in nominal_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df[models['rf']['features']]


# ── Preprocessing KNN ──────────────────────────────────────────
def preprocess_knn(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    # KNN pakai preprocessor (ColumnTransformer) yang sudah di-fit
    preprocessor = models['knn']['preprocessor']
    X_transformed = preprocessor.transform(df)
    return X_transformed


# ══════════════════════════════════════════════════════════════
#  INPUT FORM (sama untuk semua model)
# ══════════════════════════════════════════════════════════════
def input_form():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Lokasi**")
        city = st.selectbox("City", ['city_103', 'city_21', 'city_50',
                                      'city_67', 'city_16', 'city_other'])
        city_dev = st.slider("City Development Index", 0.448, 0.949, 0.900)

        st.markdown("**🎓 Pendidikan**")
        education = st.selectbox("Education Level", [
            'Primary School','High School','Graduate','Masters','Phd'
        ])
        major = st.selectbox("Major Discipline", [
            'STEM','Business Degree','Arts','Humanities',
            'No Major','Other','Unknown'
        ])

    with col2:
        st.markdown("**💼 Pengalaman**")
        experience = st.selectbox("Years of Experience", [
            '<1','1','2','3','4','5','6','7','8','9','10',
            '11','12','13','14','15','16','17','18','19','20','>20'
        ])
        relevent_exp = st.selectbox("Relevant Experience", [
            'Has relevent experience', 'No relevent experience'
        ])
        last_job = st.selectbox("Last New Job (years ago)", [
            'never','1','2','3','4','>4'
        ])

    with col3:
        st.markdown("**🏢 Perusahaan**")
        company_size = st.selectbox("Company Size", [
            '<10','10/49','50-99','100-500','500-999',
            '1000-4999','5000-9999','10000+','Unknown'
        ])
        company_type = st.selectbox("Company Type", [
            'Pvt Ltd','Funded Startup','Early Stage Startup',
            'Public Sector','NGO','Other','Unknown'
        ])
        st.markdown("**👤 Profil**")
        gender = st.selectbox("Gender", ['Male','Female','Other','Unknown'])
        enrolled = st.selectbox("Enrolled University", [
            'no_enrollment','Part time course','Full time course','Unknown'
        ])
        training_hours = st.number_input(
            "Training Hours", min_value=1, max_value=336, value=50
        )

    return {
        'city'                  : city,
        'city_development_index': city_dev,
        'gender'                : gender,
        'relevent_experience'   : relevent_exp,
        'enrolled_university'   : enrolled,
        'education_level'       : education,
        'major_discipline'      : major,
        'experience'            : experience,
        'company_size'          : company_size,
        'company_type'          : company_type,
        'last_new_job'          : last_job,
        'training_hours'        : training_hours,
    }


# ══════════════════════════════════════════════════════════════
#  HASIL PREDIKSI (reusable untuk semua model)
# ══════════════════════════════════════════════════════════════
def show_prediction(pred, proba):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Probabilitas Pindah",   f"{proba[1]*100:.1f}%")
    col_b.metric("Probabilitas Bertahan", f"{proba[0]*100:.1f}%")
    col_c.metric("Prediksi", "🚀 Pindah" if pred == 1 else "🏢 Bertahan")

    if pred == 1:
        st.markdown('<div class="result-box leave">⚠️ Kandidat diprediksi akan PINDAH KERJA</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box stay">✅ Kandidat diprediksi akan BERTAHAN</div>',
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
    st.title("🤖 ML App")
    st.markdown("---")

    page = st.selectbox("📌 Pilih Model", [
        "🌲 Random Forest",
        "🔵 KNN",
        "🌳 Model 3 (Coming Soon)",
        "🔥 Model 4 (Coming Soon)",
    ])

    st.markdown("---")

    # Info model sesuai yang dipilih
    if "Random Forest" in page and models['rf']:
        meta = models['rf']['metadata']
        st.markdown("### 📊 Info Model RF")
        st.metric("ROC-AUC", f"{meta['auc']:.4f}")
        st.metric("Macro F1", f"{meta['macro_f1']}")
        st.metric("Features", f"{meta['n_features']}")

    elif "KNN" in page and models['knn']:
        meta = models['knn']['metadata']
        st.markdown("### 📊 Info Model KNN")
        st.metric("ROC-AUC", f"{meta['auc']:.4f}")
        st.metric("Macro F1", f"{meta['macro_f1']}")
        st.metric("Features", f"{meta['n_features']}")


# ══════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════
if "Random Forest" in page:
    st.title("🌲 Random Forest — HR Job Change Prediction")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Model Info", "📋 Panduan"])

    with tab1:
        st.subheader("Input Data Kandidat")
        input_data = input_form()
        st.markdown("---")

    if st.button("🔮 Prediksi", use_container_width=True, type="primary"):
        X_input   = preprocess_rf(input_data)
        proba     = models['rf']['model'].predict_proba(X_input)[0]
        threshold = models['rf']['metadata'].get('threshold', 0.5)
        pred      = int(proba[1] >= threshold)
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        st.caption(f"📌 Menggunakan threshold optimal: {threshold}")
        show_prediction(pred, proba)

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
                'Metrik' : ['ROC-AUC','Macro F1','F1 Class 1',
                            'Recall Class 1','Threshold','Total Fitur'],
                'Nilai'  : ['0.7986','0.7348','0.62','0.73','0.46','25']
            }), use_container_width=True)

        st.markdown("### 🔧 Teknik yang Digunakan")
        c1, c2, c3 = st.columns(3)
        c1.info("**Oversampling**\nSMOTE")
        c2.info("**Feature Engineering**\n8 fitur baru")
        c3.info("**Tuning**\nOptuna 50 trials")

    with tab3:
        st.markdown("""
        ### Cara Menggunakan
        1. Isi semua input di tab Prediksi
        2. Klik tombol **Prediksi**
        3. Lihat hasil dan probabilitasnya

        | Probabilitas Pindah | Interpretasi |
        |---|---|
        | 0% - 30% | Sangat likely bertahan |
        | 30% - 50% | Cenderung bertahan |
        | 50% - 70% | Cenderung pindah |
        | 70% - 100% | Sangat likely pindah |
        """)


# ══════════════════════════════════════════════════════════════
#  PAGE: KNN
# ══════════════════════════════════════════════════════════════
elif "KNN" in page:
    st.title("🔵 KNN — HR Job Change Prediction")
    st.markdown("---")

    if models['knn'] is None:
        st.error("❌ Model KNN tidak ditemukan di folder models/")
    else:
        tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Model Info", "📋 Panduan"])

        with tab1:
            st.subheader("Input Data Kandidat")
            input_data = input_form()
            st.markdown("---")

            if st.button("🔮 Prediksi", use_container_width=True, type="primary"):
                X_input = preprocess_knn(input_data)
                pred    = models['knn']['model'].predict(X_input)[0]
                proba   = models['knn']['model'].predict_proba(X_input)[0]
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                show_prediction(pred, proba)

        with tab2:
            st.subheader("📊 Informasi Model KNN")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(
                    models['knn']['params'].items(),
                    columns=['Parameter', 'Value']
                ), use_container_width=True)
            with col2:
                meta = models['knn']['metadata']
                st.markdown("### 📈 Performa")
                st.dataframe(pd.DataFrame({
                    'Metrik': ['ROC-AUC', 'Macro F1', 'Total Fitur'],
                    'Nilai' : [meta['auc'], meta['macro_f1'], meta['n_features']]
                }), use_container_width=True)

        with tab3:
            st.markdown("""
            ### Cara Menggunakan
            1. Isi semua input di tab Prediksi
            2. Klik tombol **Prediksi**
            3. Lihat hasil dan probabilitasnya
            """)


# ══════════════════════════════════════════════════════════════
#  PAGE: COMING SOON
# ══════════════════════════════════════════════════════════════
else:
    st.title("🚧 Coming Soon")
    st.info("Halaman ini akan diisi oleh anggota kelompok lain.")