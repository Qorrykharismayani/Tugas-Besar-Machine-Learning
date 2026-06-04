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
    .main { background-color: #f8f9fa; }
    
    /* Fix metric cards agar teks keliatan di dark mode */
    [data-testid="stMetric"] {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d3d5c;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0b0 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    .result-box { 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center; 
        font-size: 1.2em; 
        font-weight: bold; 
        margin-top: 20px; 
    }
    .stay { 
        background-color: #d4edda; 
        color: #155724; 
        border: 1px solid #c3e6cb; 
    }
    .leave { 
        background-color: #f8d7da; 
        color: #721c24; 
        border: 1px solid #f5c6cb; 
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load('models/random_forest_model.pkl')
    features = joblib.load('models/selected_features.pkl')
    with open('models/best_params_rf.json') as f:
        params = json.load(f)
    metadata = joblib.load('models/model_metadata.pkl')
    return model, features, params, metadata

model, feature_names, best_params, metadata = load_model()

# ── Preprocessing (sama persis dengan notebook) ────────────────
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Missing flags
    flag_cols = ['gender', 'company_size', 'company_type',
                 'major_discipline', 'last_new_job', 'enrolled_university']
    for col in flag_cols:
        df[f'{col}_missing'] = (df[col] == 'Unknown').astype(int)

    # Feature engineering
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

    # Ordinal encoding
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

    # Nominal encoding
    nominal_cols = ['gender', 'relevent_experience', 'enrolled_university',
                    'major_discipline', 'company_type']
    for col in nominal_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df[feature_names]

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("🤖 ML App")
    st.markdown("---")

    page = st.selectbox("📌 Pilih Model", [
        "🌲 Random Forest (HR Prediction)",
        "🔵 Model Lain (Coming Soon)",   # slot untuk teman-teman
    ])

    st.markdown("---")
    st.markdown("### 📊 Info Model RF")
    st.metric("ROC-AUC",   f"{metadata['auc']:.4f}")
    st.metric("Macro F1",  f"{metadata['macro_f1']}")
    st.metric("Features",  f"{metadata['n_features']}")
    st.metric("Version",   metadata['model_version'])

# ══════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════
if "Random Forest" in page:

    st.title("🌲 HR Analytics — Job Change Prediction")
    st.markdown("Prediksi apakah seorang kandidat data scientist akan **pindah kerja** atau tidak.")
    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Model Info", "📋 Panduan"])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Input Data Kandidat")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📍 Lokasi & Kota**")
            city_dev = st.slider(
                "City Development Index", 0.448, 0.949, 0.900,
                help="Indeks perkembangan kota (0=rendah, 1=tinggi)"
            )

            st.markdown("**🎓 Pendidikan**")
            education = st.selectbox("Education Level", [
                'Primary School','High School','Graduate','Masters','Phd'
            ])
            major = st.selectbox("Major Discipline", [
                'STEM','Business Degree','Arts','Humanities','No Major','Other','Unknown'
            ])

        with col2:
            st.markdown("**💼 Pengalaman Kerja**")
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
            st.markdown("**🏢 Informasi Perusahaan**")
            company_size = st.selectbox("Company Size", [
                '<10','10/49','50-99','100-500','500-999',
                '1000-4999','5000-9999','10000+','Unknown'
            ])
            company_type = st.selectbox("Company Type", [
                'Pvt Ltd','Funded Startup','Early Stage Startup',
                'Public Sector','NGO','Other','Unknown'
            ])

            st.markdown("**👤 Profil Kandidat**")
            gender = st.selectbox("Gender", ['Male','Female','Other','Unknown'])
            enrolled = st.selectbox("Enrolled University", [
                'no_enrollment','Part time course','Full time course','Unknown'
            ])
            training_hours = st.number_input(
                "Training Hours", min_value=1, max_value=336, value=50
            )

        st.markdown("---")
        predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary")

        if predict_btn:
            input_data = {
                'city_development_index': city_dev,
                'gender'               : gender,
                'relevent_experience'  : relevent_exp,
                'enrolled_university'  : enrolled,
                'education_level'      : education,
                'major_discipline'     : major,
                'experience'           : experience,
                'company_size'         : company_size,
                'company_type'         : company_type,
                'last_new_job'         : last_job,
                'training_hours'       : training_hours,
            }

            X_input = preprocess_input(input_data)
            pred    = model.predict(X_input)[0]
            proba   = model.predict_proba(X_input)[0]

            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Probabilitas Pindah",  f"{proba[1]*100:.1f}%")
            col_b.metric("Probabilitas Bertahan", f"{proba[0]*100:.1f}%")
            col_c.metric("Prediksi", "🚀 Pindah" if pred == 1 else "🏢 Bertahan")

            if pred == 1:
                st.markdown('<div class="result-box leave">⚠️ Kandidat ini diprediksi akan PINDAH KERJA</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box stay">✅ Kandidat ini diprediksi akan BERTAHAN</div>',
                            unsafe_allow_html=True)

            # Gauge chart probabilitas
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh([''], [proba[1]], color='tomato' if pred==1 else 'steelblue',
                    height=0.4)
            ax.barh([''], [proba[0]], left=[proba[1]],
                    color='#e9ecef', height=0.4)
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probabilitas')
            ax.set_title('Distribusi Probabilitas Prediksi')
            st.pyplot(fig)

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("📊 Informasi Model")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚙️ Hyperparameter Terbaik")
            params_df = pd.DataFrame(
                best_params.items(), columns=['Parameter', 'Value']
            )
            st.dataframe(params_df, use_container_width=True)

        with col2:
            st.markdown("### 📈 Performa Model")
            perf_df = pd.DataFrame({
                'Metrik' : ['ROC-AUC', 'Macro F1', 'F1 Class 0', 'F1 Class 1',
                            'Recall Class 1', 'Total Fitur'],
                'Nilai'  : ['0.7986', '0.73', '0.85', '0.61', '0.69', '25']
            })
            st.dataframe(perf_df, use_container_width=True)

        st.markdown("### 🔧 Teknik yang Digunakan")
        col3, col4, col5 = st.columns(3)
        col3.info("**Oversampling**\nSMOTE untuk handle class imbalance (75:25)")
        col4.info("**Feature Engineering**\n8 fitur baru dari domain knowledge HR")
        col5.info("**Hyperparameter Tuning**\nOptuna TPE Sampler, 50 trials")

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("📋 Panduan Penggunaan")
        st.markdown("""
        ### Cara Menggunakan Aplikasi

        1. **Isi semua input** di tab Prediksi sesuai profil kandidat
        2. Klik tombol **Prediksi Sekarang**
        3. Lihat hasil prediksi dan probabilitasnya

        ### Interpretasi Hasil

        | Probabilitas Pindah | Interpretasi |
        |---|---|
        | 0% - 30% | Kandidat sangat likely bertahan |
        | 30% - 50% | Kandidat cenderung bertahan |
        | 50% - 70% | Kandidat cenderung pindah |
        | 70% - 100% | Kandidat sangat likely pindah |

        ### Tentang Dataset
        - **Sumber**: HR Analytics Job Change of Data Scientists
        - **Total data**: 19.158 kandidat
        - **Target**: Prediksi apakah kandidat akan pindah kerja
        """)

# ══════════════════════════════════════════════════════════════
#  PAGE: COMING SOON (slot untuk teman-teman)
# ══════════════════════════════════════════════════════════════
else:
    st.title("🔵 Model Lain")
    st.info("Halaman ini akan diisi oleh anggota kelompok lain.")