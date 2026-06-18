import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import time

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS UTILITIES
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Modern Font & Base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Gradients and Backgrounds */
.stApp {
    background: radial-gradient(circle at top left, #121c2d 0%, #080c14 100%);
    color: #e2e8f0;
}

/* Card Styling */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(96, 165, 250, 0.5);
}

/* Metric Labels */
div[data-testid="stMetricLabel"] > div {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Headings */
h1, h2, h3 { color: #f8fafc; font-weight: 700; }
.hero-title { font-size: 2.5rem; font-weight: 800; background: linear-gradient(to right, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2; margin-bottom: 0.5rem;}
.hero-subtitle { color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin-bottom: 2rem;}

/* Status and Results */
.result-fail {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-left: 4px solid #ef4444;
    padding: 1rem 1.5rem; border-radius: 8px;
    color: #fca5a5; font-weight: 600; margin: 1rem 0;
}
.result-ok {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-left: 4px solid #22c55e;
    padding: 1rem 1.5rem; border-radius: 8px;
    color: #86efac; font-weight: 600; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA METRIK & PARAMETER
# ══════════════════════════════════════════════════════════════
RAW_COLS = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

PERF_DATA = {
    'rf':  {'name': 'Random Forest', 'color': '#f97316', 'icon': '🌲',
            'accuracy': None, 'precision': None, 'recall': 0.8382, 'f1_fail': None,
            'auc': 0.9847, 'macro_f1': 0.9163, 'threshold': 0.71, 'sampling': 'SMOTE',
            'params': {'n_estimators': 427, 'max_depth': 12, 'min_samples_split': 2,
                       'min_samples_leaf': 2, 'max_features': 'log2', 'class_weight': 'balanced'}},
    'dt':  {'name': 'Decision Tree', 'color': '#22c55e', 'icon': '🌳',
            'accuracy': 0.9855, 'precision': 0.8000, 'recall': 0.7647, 'f1_fail': 0.7820,
            'auc': None, 'macro_f1': None, 'threshold': 0.5, 'sampling': 'Asli (tanpa SMOTE)',
            'params': {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}},
    'knn': {'name': 'K-Nearest Neighbors', 'color': '#3b82f6', 'icon': '🔵',
            'accuracy': 0.9740, 'precision': 0.8333, 'recall': 0.2941, 'f1_fail': 0.4348,
            'auc': None, 'macro_f1': None, 'threshold': 0.5, 'sampling': 'Asli (tanpa SMOTE)',
            'params': {'n_neighbors': 5}},
    'svm': {'name': 'Support Vector Machine', 'color': '#eab308', 'icon': '⚡',
            'accuracy': 0.9790, 'precision': 0.7708, 'recall': 0.5441, 'f1_fail': 0.6379,
            'auc': None, 'macro_f1': None, 'threshold': 0.5, 'sampling': 'Asli (tanpa SMOTE)',
            'params': {'kernel': 'rbf', 'C': 10}},
}

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    m = {}
    def try_load(loaders):
        try: return {k: fn() for k, fn in loaders.items()}
        except Exception: return None

    m['rf'] = try_load({
        'model'   : lambda: joblib.load('models/rf_model.pkl'),
        'features': lambda: joblib.load('models/rf_selected_features.pkl'),
        'scaler'  : lambda: joblib.load('models/rf_scaler.pkl'),
        'le_dict' : lambda: joblib.load('models/rf_label_encoders.pkl'),
    })

    def load_generic(candidates):
        out = {}
        for key, fnames in candidates.items():
            for fn in fnames:
                try:
                    out[key] = joblib.load(f'models/{fn}')
                    break
                except Exception: continue
        return out if 'model' in out else None

    m['dt'] = load_generic({
        'model'       : ['dt_model_maintenance.pkl'],
        'preprocessor': ['dt_preprocessor_maintenance.pkl'],
        'features'    : ['dt_selected_features_maintenance.pkl'],
    })
    m['knn'] = load_generic({
        'model'       : ['knn_model_maintenance.pkl'],
        'preprocessor': ['knn_preprocessor_maintenance.pkl'],
        'features'    : ['knn_selected_features_maintenance.pkl'],
    })
    m['svm'] = load_generic({
        'model'       : ['svm_model_maintenance.pkl'],
        'preprocessor': ['svm_preprocessor_maintenance.pkl'],
        'features'    : ['svm_selected_features_maintenance.pkl'],
    })
    return m

models = load_all_models()

# ══════════════════════════════════════════════════════════════
#  CHART HELPERS (ALTAIR)
# ══════════════════════════════════════════════════════════════
ORDER = ['rf', 'dt', 'knn', 'svm']

def plot_metric_comparison_altair():
    data = []
    for k in ORDER:
        data.append({'Model': PERF_DATA[k]['name'], 'Metric': 'Accuracy', 'Score': PERF_DATA[k].get('accuracy') or 0})
        data.append({'Model': PERF_DATA[k]['name'], 'Metric': 'Recall (Failure)', 'Score': PERF_DATA[k].get('recall') or 0})
        f1 = PERF_DATA[k].get('f1_fail') if PERF_DATA[k].get('f1_fail') is not None else PERF_DATA[k].get('macro_f1')
        data.append({'Model': PERF_DATA[k]['name'], 'Metric': 'F1 (Failure)', 'Score': f1 or 0})
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Metric:N', title='', axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=['Accuracy', 'Recall (Failure)', 'F1 (Failure)'], range=['#f97316', '#3b82f6', '#a855f7']), legend=alt.Legend(title="Metrik", orient='bottom')),
        column=alt.Column('Model:N', header=alt.Header(labelOrient='bottom', titleOrient='top', title='Model', labelColor='white', titleColor='#94a3b8')),
        tooltip=['Model', 'Metric', 'Score']
    ).properties(width=120, height=300).configure_view(strokeOpacity=0).configure_axis(grid=True, gridColor='#1e293b', domainColor='#1e293b', labelColor='#94a3b8', titleColor='#94a3b8').configure_legend(labelColor='#e2e8f0', titleColor='#94a3b8')
    return chart

def plot_recall_focus_altair():
    data = [{'Model': PERF_DATA[k]['name'], 'Recall': PERF_DATA[k]['recall'], 'Color': PERF_DATA[k]['color']} for k in ORDER]
    df = pd.DataFrame(data).sort_values('Recall', ascending=False)
    
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('Model:N', sort='-x', title='', axis=alt.Axis(labelColor='#e2e8f0')),
        x=alt.X('Recall:Q', title='Recall (Failure)', scale=alt.Scale(domain=[0, 1.0]), axis=alt.Axis(labelColor='#94a3b8', titleColor='#94a3b8')),
        color=alt.Color('Color:N', scale=None),
        tooltip=['Model', 'Recall']
    ).properties(height=250, width=400).configure_view(strokeOpacity=0).configure_axis(grid=True, gridColor='#1e293b', domainColor='#1e293b')
    return chart

# ══════════════════════════════════════════════════════════════
#  SIMULATION & PREDICTION
# ══════════════════════════════════════════════════════════════
SAMPLE_ROWS = {
    'Normal (Sehat)': {'Type': 'L', 'Air temperature [K]': 298.1, 'Process temperature [K]': 308.6,
                       'Rotational speed [rpm]': 1551, 'Torque [Nm]': 42.8, 'Tool wear [min]': 0},
    'Berisiko Gagal (Tool Aus + Torsi Tinggi)': {'Type': 'H', 'Air temperature [K]': 302.3,
                       'Process temperature [K]': 311.5, 'Rotational speed [rpm]': 1320,
                       'Torque [Nm]': 64.5, 'Tool wear [min]': 215},
}

SIM_STEPS = {
    'rf': [
        ("Input Data Sensor", "Mengambil 6 nilai mentah: Type, suhu udara, suhu proses, rpm, torsi, tool wear."),
        ("Encoding Kategorikal", "Kolom Type (L/M/H) diubah menjadi angka memakai LabelEncoder."),
        ("Feature Engineering", "Membuat 3 fitur turunan: temp_diff (suhu selisih), power (torsi × rpm), strain (tool wear × torsi)."),
        ("Scaling", "Seluruh 8 fitur numerik distandarkan (StandardScaler)."),
        ("Voting Antar Pohon", "427 pohon keputusan memberikan prediksi, kemudian dirata-ratakan."),
        ("Thresholding", "Menerapkan ambang probabilitas 0.71 (hasil optimasi SMOTE).")
    ],
    'dt': [
        ("Input Data Sensor", "Mengambil 6 nilai mentah sensor."),
        ("Preprocessing Data", "Imputasi median & StandardScaler (Numerik), One-Hot Encoding (Type)."),
        ("Eksplorasi Aturan Pohon", "Data menelusuri cabang pohon berdasarkan aturan if-else (cth: torsi > 50)."),
        ("Mencapai Node Daun", "Node terakhir menentukan klasifikasi akhir."),
        ("Thresholding", "Menggunakan ambang probabilitas 0.50 (data asli).")
    ],
    'knn': [
        ("Input Data Sensor", "Mengambil 6 nilai mentah sensor."),
        ("Preprocessing Data", "Standardisasi sangat penting untuk mengukur jarak Euclid."),
        ("Pengukuran Jarak", "Menghitung jarak ke seluruh data training."),
        ("Pemilihan K-Terdekat", "Mengambil K=5 tetangga yang paling mirip dengan sampel."),
        ("Thresholding", "Voting mayoritas kelas dengan ambang probabilitas 0.50.")
    ],
    'svm': [
        ("Input Data Sensor", "Mengambil 6 nilai mentah sensor."),
        ("Preprocessing Data", "Standardisasi fitur dan One-Hot Encoding (Type)."),
        ("Proyeksi Kernel RBF", "Data dipetakan ke dimensi lebih tinggi (C=10) untuk mencari batas pisah."),
        ("Kalkulasi Hyperplane", "Menghitung posisi titik terhadap margin keputusan maksimal."),
        ("Thresholding", "Konversi jarak margin menjadi probabilitas (ambang 0.50).")
    ],
}

def preprocess_rf(data):
    df = pd.DataFrame([data])
    rename_map = {'Air temperature [K]': 'air_temp', 'Process temperature [K]': 'process_temp', 'Rotational speed [rpm]': 'rot_speed', 'Torque [Nm]': 'torque', 'Tool wear [min]': 'tool_wear'}
    df = df.rename(columns=rename_map)
    for col, le in models['rf']['le_dict'].items():
        if col in df.columns: df[col] = le.transform(df[col].astype(str))
    df['temp_diff'] = df['process_temp'] - df['air_temp']
    df['power'] = df['torque'] * df['rot_speed']
    df['strain'] = df['tool_wear'] * df['torque']
    num_cols = ['air_temp', 'process_temp', 'rot_speed', 'torque', 'tool_wear', 'temp_diff', 'power', 'strain']
    df[num_cols] = models['rf']['scaler'].transform(df[num_cols])
    for f in models['rf']['features']:
        if f not in df.columns: df[f] = 0
    return df[models['rf']['features']]

def preprocess_generic(key, data):
    df = pd.DataFrame([data])[RAW_COLS]
    pre = models[key].get('preprocessor')
    return pre.transform(df) if pre is not None else df

def predict_any(key, data):
    X = preprocess_rf(data) if key == 'rf' else preprocess_generic(key, data)
    proba = models[key]['model'].predict_proba(X)[0]
    th = PERF_DATA[key]['threshold']
    return int(proba[1] >= th), proba, th

def run_simulation(key, sample):
    st.write("")
    with st.status(f"🚀 Memulai Simulasi Pipeline: {PERF_DATA[key]['name']}", expanded=True) as status:
        for i, (judul, desc) in enumerate(SIM_STEPS[key]):
            time.sleep(0.7)
            st.markdown(f"**{i+1}. {judul}**")
            st.caption(f"↳ {desc}")
        time.sleep(0.5)
        status.update(label="✅ Simulasi Selesai", state="complete", expanded=False)
        
    try:
        pred, proba, th = predict_any(key, sample)
        show_prediction_result(pred, proba, th)
    except Exception as e:
        st.error(f"Gagal menjalankan prediksi: {e}")

def show_prediction_result(pred, proba, threshold):
    st.markdown("### 🎯 Hasil Klasifikasi")
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Prob. Failure", f"{proba[1]*100:.1f}%")
    m2.metric("🟢 Prob. Normal", f"{proba[0]*100:.1f}%")
    m3.metric("📌 Threshold", f"{threshold:.2f}")

    if pred == 1:
        st.markdown('<div class="result-fail">🚨 Mesin diprediksi akan <strong>GAGAL (FAILURE)</strong> — Segera lakukan inspeksi & maintenance!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-ok">✅ Mesin diprediksi <strong>NORMAL</strong> — Aman untuk dioperasikan.</div>', unsafe_allow_html=True)
    
    # Progress bar style visualization
    st.write("**Visualisasi Probabilitas**")
    progress_html = f"""
    <div style='width:100%; background-color:#1e293b; border-radius:8px; overflow:hidden; display:flex; height:24px;'>
        <div style='width:{proba[1]*100}%; background-color:#ef4444; display:flex; align-items:center; justify-content:center; color:white; font-size:0.75rem; font-weight:bold;'>{proba[1]*100:.1f}% Fail</div>
        <div style='width:{proba[0]*100}%; background-color:#22c55e; display:flex; align-items:center; justify-content:center; color:white; font-size:0.75rem; font-weight:bold;'>{proba[0]*100:.1f}% Normal</div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
NAV = {"🏠 Beranda": 'home', "🌲 Random Forest": 'rf', "🌳 Decision Tree": 'dt', "🔵 KNN": 'knn', "⚡ SVM": 'svm'}

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding-top: 1rem; padding-bottom: 2rem;'>
        <div style='font-size:3rem; margin-bottom:0.5rem;'>⚙️</div>
        <div style='font-size:1.25rem; font-weight:800; color:white;'>Predictive Maintenance</div>
        <div style='color:#94a3b8; font-size:0.8rem; margin-top:0.25rem;'>AI-Driven Anomaly Detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("Navigasi Utama", list(NAV.keys()))
    active = NAV[page]
    
    st.divider()
    st.caption("STATUS MODEL")
    for label, key in NAV.items():
        if key == 'home': continue
        status_icon = "🟢" if models.get(key) else "🔴"
        st.markdown(f"`{status_icon}` {label}")

# ══════════════════════════════════════════════════════════════
#  PAGE ROUTING
# ══════════════════════════════════════════════════════════════
if active == 'home':
    st.markdown("<div class='hero-title'>Prediksi Kegagalan Mesin dengan Machine Learning</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Analisis performa model klasifikasi (Random Forest, Decision Tree, KNN, dan SVM) untuk mendeteksi dini risiko kerusakan mesin berdasarkan data sensor real-time.</div>", unsafe_allow_html=True)
    
    st.divider()

    # --- Section: Info Dataset ---
    st.markdown("### 📦 Overview Dataset")
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Data", "10.000 Baris")
        m2.metric("Fitur Input", "6 Kolom")
        m3.metric("Kelas Normal", "9.661 (96.6%)")
        m4.metric("Kelas Failure", "339 (3.4%)")
        
        st.info("**Imbalance Data Extreme:** Karena hanya 3.4% mesin yang mengalami kerusakan, **Recall Failure** jauh lebih krusial dibandingkan sekadar Akurasi Global. Gagal mendeteksi mesin rusak (False Negative) bisa mengakibatkan downtime industri yang mahal.")
        
        with st.expander("Lihat Detail Fitur / Kolom", expanded=False):
            st.dataframe(pd.DataFrame({
                'Fitur': ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'],
                'Tipe': ['Kategorikal', 'Numerik', 'Numerik', 'Numerik', 'Numerik', 'Numerik'],
                'Deskripsi': ['Kualitas produk (L/M/H)', 'Suhu udara sekitar', 'Suhu proses mesin', 'Kecepatan putaran', 'Torsi yang bekerja', 'Akumulasi keausan alat']
            }), use_container_width=True, hide_index=True)

    st.write("")
    
    # --- Section: Simulasi Testing ---
    st.markdown("### 🧪 Simulasi Prediksi Interaktif")
    st.write("Jalankan simulasi langkah demi langkah untuk melihat bagaimana model memproses data sensor.")
    
    with st.container(border=True):
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            sim_model_label = st.selectbox("1️⃣ Pilih Model AI", ['🌲 Random Forest', '🌳 Decision Tree', '🔵 KNN', '⚡ SVM'])
            sim_key = {'🌲 Random Forest': 'rf', '🌳 Decision Tree': 'dt', '🔵 KNN': 'knn', '⚡ SVM': 'svm'}[sim_model_label]
        with col_sim2:
            sim_sample_label = st.selectbox("2️⃣ Pilih Skenario Data Sensor", list(SAMPLE_ROWS.keys()))
            sim_sample = SAMPLE_ROWS[sim_sample_label]

        st.caption("Data Sensor yang diuji:")
        st.dataframe(pd.DataFrame([sim_sample]), use_container_width=True, hide_index=True)
        
        if not models.get(sim_key):
            st.error(f"Model {sim_model_label} gagal dimuat!")
        else:
            if st.button("▶️ Mulai Simulasi Testing", type="primary", use_container_width=True):
                run_simulation(sim_key, sim_sample)

    st.write("")
    
    # --- Section: Perbandingan Performa ---
    st.markdown("### 📊 Perbandingan Performa Model")
    col_chart1, col_chart2 = st.columns([1.5, 1])
    with col_chart1:
        st.altair_chart(plot_metric_comparison_altair(), use_container_width=True)
    with col_chart2:
        st.altair_chart(plot_recall_focus_altair(), use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 🏆 Kesimpulan & Rekomendasi")
        c1, c2 = st.columns(2)
        with c1:
            st.success("**Random Forest adalah model terbaik.** Dengan kombinasi SMOTE, feature engineering, dan optimasi threshold, model ini mencetak **Recall Failure tertinggi (83.8%)**. Artinya, model ini paling tangguh mendeteksi mesin bermasalah.")
        with c2:
            st.warning("**Hindari Model dengan Recall Rendah.** Meskipun KNN dan SVM memiliki akurasi di atas 97%, recall mereka sangat buruk (hanya ~30-50%). Mereka cenderung buta terhadap mesin yang rusak karena dominasi kelas mayoritas (Normal).")


else:
    # ══════════════════════════════════════════════════════════════
    #  PAGE: INDIVIDUAL MODEL DETAIL
    # ══════════════════════════════════════════════════════════════
    d = PERF_DATA[active]
    st.markdown(f"<div class='hero-title' style='font-size:2rem;'>{d['icon']} {d['name']}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Detail arsitektur, parameter, dan uji prediksi mandiri.</div>", unsafe_allow_html=True)
    
    if not models.get(active):
        st.error(f"⚠️ Model {d['name']} gagal dimuat. Pastikan file model tersedia.")
    else:
        t_info, t_pred = st.tabs(["📊 Performa & Parameter", "🔮 Prediksi Kustom"])
        
        with t_info:
            # Metrics
            st.subheader("Key Performance Indicators")
            m1, m2, m3, m4 = st.columns(4)
            if active == 'rf':
                m1.metric("ROC-AUC", f"{d['auc']:.4f}")
                m2.metric("Macro F1", f"{d['macro_f1']:.4f}")
                m3.metric("Recall Failure", f"{d['recall']*100:.1f}%")
                m4.metric("Threshold Evaluasi", f"{d['threshold']}")
            else:
                m1.metric("Accuracy", f"{d['accuracy']*100:.2f}%")
                m2.metric("Recall Failure", f"{d['recall']*100:.1f}%")
                m3.metric("F1 Failure", f"{d['f1_fail']:.4f}")
                m4.metric("Threshold Evaluasi", f"{d['threshold']}")

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("#### ⚙️ Konfigurasi Hyperparameter")
                    st.dataframe(pd.DataFrame(d['params'].items(), columns=['Parameter', 'Nilai Config']), use_container_width=True, hide_index=True)
            with c2:
                with st.container(border=True):
                    st.markdown("#### 🛠️ Metode Pelatihan")
                    st.markdown(f"- **Metode Sampling:** {d['sampling']}")
                    if active == 'rf':
                        st.markdown("- **Feature Engineering:** `temp_diff`, `power`, `strain`")
                    else:
                        st.markdown("- **Preprocessing:** `StandardScaler`, `One-Hot Encoding`")
        
        with t_pred:
            st.subheader("Input Spesifikasi Mesin")
            with st.form("custom_predict_form"):
                fc1, fc2 = st.columns(2)
                with fc1:
                    t_val = st.selectbox("Kualitas Mesin (Type)", ['L', 'M', 'H'])
                    a_val = st.number_input("Air temperature [K]", 290.0, 320.0, 298.0, 0.1)
                    p_val = st.number_input("Process temperature [K]", 300.0, 320.0, 308.0, 0.1)
                with fc2:
                    r_val = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500, 10)
                    tq_val = st.number_input("Torque [Nm]", 3.0, 80.0, 40.0, 0.1)
                    w_val = st.number_input("Tool wear [min]", 0, 300, 50, 1)
                
                submitted = st.form_submit_button("🔮 Eksekusi Prediksi", type="primary", use_container_width=True)
                
                if submitted:
                    custom_data = {
                        'Type': t_val, 'Air temperature [K]': a_val, 'Process temperature [K]': p_val,
                        'Rotational speed [rpm]': r_val, 'Torque [Nm]': tq_val, 'Tool wear [min]': w_val
                    }
                    try:
                        c_pred, c_proba, c_th = predict_any(active, custom_data)
                        st.divider()
                        show_prediction_result(c_pred, c_proba, c_th)
                    except Exception as e:
                        st.error(f"Error prediksi: {e}")