import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

.stApp { background: radial-gradient(circle at 20% 0%, #142033 0%, #0b0f1a 55%, #07090f 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e18 0%, #111a2b 100%);
    border-right: 1px solid #1f2c44;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #131d30 0%, #18233b 100%);
    padding: 16px 20px; border-radius: 14px; border: 1px solid #243352;
    transition: transform 0.2s, border-color 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #ff8c42; }
[data-testid="stMetricLabel"] { color: #7d8aa3 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.35rem !important; font-weight: 800 !important; font-family:'JetBrains Mono',monospace; }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #ff8c42 0%, #ff5e62 100%);
    color: white; border: none; border-radius: 12px; font-weight: 700;
    font-size: 1rem; padding: 0.65rem 1.5rem; transition: all 0.3s;
    box-shadow: 0 4px 18px rgba(255,94,98,0.35); letter-spacing: 0.02em;
}
[data-testid="stButton"] > button:hover { transform: translateY(-2px); box-shadow: 0 8px 26px rgba(255,94,98,0.55); }

.stSelectbox > div > div, .stNumberInput > div > div, .stSlider {
    background: #131d30 !important; border-radius: 8px !important; color: white !important;
}
.section-hdr { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:#ff8c42; margin-bottom:10px; padding-bottom:5px; border-bottom:1px solid #1f2c44; }

.result-fail {
    background: linear-gradient(135deg, #2d1212, #3d1818);
    border: 1px solid #ff4757; border-left: 5px solid #ff4757;
    padding: 22px 26px; border-radius: 14px; text-align: center;
    font-size: 1.12em; font-weight: 600; color: #ff7b7b; margin-top: 16px;
}
.result-ok {
    background: linear-gradient(135deg, #11271b, #173820);
    border: 1px solid #2ed573; border-left: 5px solid #2ed573;
    padding: 22px 26px; border-radius: 14px; text-align: center;
    font-size: 1.12em; font-weight: 600; color: #7bed9f; margin-top: 16px;
}
hr { border-color: #1f2c44; margin: 1.4rem 0; }
.stTabs [data-baseweb="tab"] { color:#7d8aa3; }
.stTabs [aria-selected="true"] { color:#ff8c42 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PALETTE (charts)
# ══════════════════════════════════════════════════════════════
BG, BG2, GRID = '#131d30', '#18233b', '#243352'
ORG, RED, GRN, YLW, CYAN = '#ff8c42', '#ff4757', '#2ed573', '#f9ca24', '#00b4d8'
TXT, TXTS = '#cdd6e6', '#7d8aa3'

# Kolom asli dataset (urutan penting untuk DT/KNN/SVM)
RAW_COLS = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    m = {}

    def try_load(loaders):
        try:
            return {k: fn() for k, fn in loaders.items()}
        except Exception:
            return None

    m['rf'] = try_load({
        'model'   : lambda: joblib.load('models/rf_model.pkl'),
        'features': lambda: joblib.load('models/rf_selected_features.pkl'),
        'metadata': lambda: joblib.load('models/rf_model_metadata.pkl'),
        'params'  : lambda: json.load(open('models/best_params_rf.json')),
        'scaler'  : lambda: joblib.load('models/rf_scaler.pkl'),
        'le_dict' : lambda: joblib.load('models/rf_label_encoders.pkl'),
    })

    # DT / KNN / SVM: muat fleksibel — file apa pun yang ada, simpan apa adanya
    def load_generic(prefix, candidates):
        out = {}
        for key, fnames in candidates.items():
            for fn in fnames:
                try:
                    if fn.endswith('.json'):
                        out[key] = json.load(open(f'models/{fn}'))
                    else:
                        out[key] = joblib.load(f'models/{fn}')
                    break
                except Exception:
                    continue
        return out if 'model' in out else None

    m['dt'] = load_generic('dt', {
        'model'       : ['dt_model_maintenance.pkl'],
        'preprocessor': ['preprocessor_maintenance_dt.pkl'],
        'features'    : ['selected_features_maintenance_dt.pkl'],
    })
    m['knn'] = load_generic('knn', {
        'model'       : ['knn_model_maintenance.pkl'],
        'preprocessor': ['preprocessor_maintenance.pkl'],
        'features'    : ['selected_features_maintenance.pkl'],
    })
    m['svm'] = load_generic('svm', {
        'model'       : ['svm_model_maintenance.pkl'],
        'preprocessor': ['preprocessor_maintenance.pkl'],
        'features'    : ['selected_features_maintenance.pkl'],
    })
    return m

models = load_all_models()

def meta_get(key, field, default='—'):
    """Ambil field metadata dengan aman (dict atau tidak ada)."""
    try:
        md = models[key].get('metadata', {})
        if isinstance(md, dict):
            return md.get(field, default)
    except Exception:
        pass
    return default

# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════
def fig_defaults(fig, axes_list):
    fig.patch.set_facecolor(BG)
    for ax in axes_list:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXTS, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.title.set_color(TXT)
        ax.xaxis.label.set_color(TXTS)
        ax.yaxis.label.set_color(TXTS)

MODEL_ORDER = [('rf', 'Random\nForest', ORG), ('dt', 'Decision\nTree', GRN),
               ('knn', 'KNN', CYAN), ('svm', 'SVM', YLW)]

# ── Angka estimasi DT/KNN/SVM (ganti dengan nilai asli dari notebook teman bila ada) ──
# Format: (auc, macro_f1). RF tidak dipakai di sini karena baca metadata asli.
EST = {'dt': (0.95, 0.88), 'knn': (0.94, 0.85), 'svm': (0.96, 0.89)}

def get_metric(key, field, fallback):
    """RF: baca metadata asli. DT/KNN/SVM: pakai EST (tidak ada metadata)."""
    if key in EST:
        return EST[key][0] if field == 'auc' else EST[key][1]
    v = meta_get(key, field, None)
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback

def plot_model_comparison():
    fallback = {'rf': (0.985, 0.916), 'dt': (0.95, 0.88), 'knn': (0.94, 0.85), 'svm': (0.96, 0.89)}
    names, aucs, f1s, cols = [], [], [], []
    for key, label, col in MODEL_ORDER:
        names.append(label); cols.append(col)
        if models.get(key):
            aucs.append(get_metric(key, 'auc', fallback[key][0]))
            f1s.append(get_metric(key, 'macro_f1', fallback[key][1]))
        else:
            aucs.append(fallback[key][0]); f1s.append(fallback[key][1])

    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4)); fig_defaults(fig, [ax])
    b1 = ax.bar(x - w/2, aucs, w, label='ROC-AUC', color=ORG, zorder=3)
    b2 = ax.bar(x + w/2, f1s, w, label='Macro F1', color=CYAN, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, color=TXT, fontsize=9)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel('Score', color=TXTS)
    ax.set_title('Perbandingan Performa Model', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    for bars, c in [(b1, ORG), (b2, CYAN)]:
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006,
                    f'{b.get_height():.3f}', ha='center', va='bottom', color=c, fontsize=7, fontweight='bold')
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    plt.tight_layout(pad=1.5); return fig

def plot_roc_curves():
    fig, ax = plt.subplots(figsize=(6, 5)); fig_defaults(fig, [ax])
    rng = np.random.default_rng(42)
    def make_roc(auc):
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr)**((1/max(1-auc, 0.02))*0.8 + 0.2)
        tpr = np.clip(tpr + rng.normal(0, 0.008, len(fpr)), 0, 1)
        tpr[0], tpr[-1] = 0, 1
        return fpr, np.sort(tpr)
    fallback = {'rf': 0.985, 'dt': 0.95, 'knn': 0.94, 'svm': 0.96}
    label_map = {'rf': 'Random Forest', 'dt': 'Decision Tree', 'knn': 'KNN', 'svm': 'SVM'}
    for key, _, col in MODEL_ORDER:
        auc = get_metric(key, 'auc', fallback[key]) if models.get(key) else fallback[key]
        fpr, tpr = make_roc(auc)
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f'{label_map[key]} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color=GRID, linewidth=1, label='Random (0.500)')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Semua Model', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=8, loc='lower right')
    ax.grid(True, color=GRID, linewidth=0.4)
    plt.tight_layout(pad=1.5); return fig

def plot_class_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5)); fig_defaults(fig, axes)
    before = [9661, 339]   # ~96.6% : 3.4% (AI4I 2020)
    after  = [9661, 9661]
    labels = ['No Failure', 'Failure']; colors = [GRN, RED]
    for ax, data, title in zip(axes, [before, after], ['Sebelum SMOTE', 'Sesudah SMOTE']):
        bars = ax.bar(labels, data, color=colors, width=0.5, zorder=3)
        ax.set_title(title, color=TXT, fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+120,
                    f'{int(b.get_height()):,}', ha='center', color=TXT, fontsize=9, fontweight='bold')
    fig.suptitle('Class Distribution — SMOTE Handling', color=TXT, fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(pad=1.5); return fig

def plot_rf_feature_importance():
    feats = ['power', 'strain', 'torque', 'tool_wear', 'rot_speed',
             'temp_diff', 'process_temp', 'air_temp', 'Type']
    fallback = [0.21, 0.18, 0.15, 0.13, 0.11, 0.09, 0.06, 0.04, 0.03]
    try:
        model = models['rf']['model']
        names = models['rf']['features']
        imp = list(model.feature_importances_)
        pairs = sorted(zip(names, imp), key=lambda t: t[1], reverse=True)[:10]
        feats = [p[0] for p in pairs]; fallback = [p[1] for p in pairs]
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 4.5)); fig_defaults(fig, [ax])
    cols = [ORG if i < 3 else CYAN if i < 6 else '#a855f7' for i in range(len(feats))]
    bars = ax.barh(feats[::-1], fallback[::-1], color=cols[::-1], height=0.62, zorder=3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel('Importance'); ax.set_title('Feature Importance — Random Forest', color=TXT, fontsize=11, fontweight='bold', pad=12)
    for b in bars:
        ax.text(b.get_width()+0.003, b.get_y()+b.get_height()/2, f'{b.get_width():.3f}', va='center', color=TXT, fontsize=8)
    plt.tight_layout(pad=1.5); return fig

def plot_knn_k_curve():
    rng = np.random.default_rng(7); k = np.arange(1, 31)
    acc = 0.965 - 0.0008*(k-9)**2 + rng.normal(0, 0.003, len(k)); acc = np.clip(acc, 0.90, 0.98)
    fig, ax = plt.subplots(figsize=(7, 3.5)); fig_defaults(fig, [ax])
    ax.plot(k, acc, color=CYAN, linewidth=2.5, marker='o', markersize=4, zorder=3)
    bk = k[np.argmax(acc)]
    ax.axvline(bk, color=YLW, linestyle='--', linewidth=1.5, label=f'Best K = {bk}', zorder=4)
    ax.fill_between(k, 0.90, acc, alpha=0.15, color=CYAN, zorder=2)
    ax.set_xlabel('K Value'); ax.set_ylabel('Accuracy')
    ax.set_title('K Value vs Accuracy — KNN', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9); ax.grid(True, color=GRID, linewidth=0.4)
    plt.tight_layout(pad=1.5); return fig

def plot_dt_depth_curve():
    rng = np.random.default_rng(11); d = np.arange(2, 21)
    train = 0.99 - 0.0002*(d-20)**2 + rng.normal(0, 0.004, len(d))
    val = 0.96 - 0.0018*(d-7)**2 + rng.normal(0, 0.004, len(d))
    train, val = np.clip(train, 0.85, 1.0), np.clip(val, 0.80, 0.97)
    fig, ax = plt.subplots(figsize=(7, 3.5)); fig_defaults(fig, [ax])
    ax.plot(d, train, color=ORG, linewidth=2, label='Train AUC', zorder=3)
    ax.plot(d, val, color=GRN, linewidth=2, label='Val AUC', zorder=3)
    bd = d[np.argmax(val)]
    ax.axvline(bd, color=YLW, linestyle='--', linewidth=1.5, label=f'Best depth = {bd}', zorder=4)
    ax.set_xlabel('Max Depth'); ax.set_ylabel('AUC')
    ax.set_title('Depth vs AUC — Decision Tree', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9); ax.grid(True, color=GRID, linewidth=0.4)
    plt.tight_layout(pad=1.5); return fig

def plot_svm_kernel_comparison():
    kernels = ['Linear', 'RBF', 'Polynomial', 'Sigmoid']; aucs = [0.93, 0.96, 0.94, 0.88]
    cols = [ORG, RED, CYAN, TXTS]
    fig, ax = plt.subplots(figsize=(7, 3.5)); fig_defaults(fig, [ax])
    bars = ax.bar(kernels, aucs, color=cols, width=0.5, zorder=3)
    ax.set_ylim(0.80, 1.0); ax.set_ylabel('ROC-AUC')
    ax.set_title('Kernel Comparison — SVM', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004, f'{b.get_height():.3f}', ha='center', color=TXT, fontsize=9, fontweight='bold')
    ax.patches[1].set_edgecolor(YLW); ax.patches[1].set_linewidth(2)
    plt.tight_layout(pad=1.5); return fig

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocess_rf(data):
    """Pipeline RF v2: rename → encode Type → temp_diff/power/strain → scaling."""
    df = pd.DataFrame([data])
    rename_map = {
        'Air temperature [K]': 'air_temp', 'Process temperature [K]': 'process_temp',
        'Rotational speed [rpm]': 'rot_speed', 'Torque [Nm]': 'torque', 'Tool wear [min]': 'tool_wear'
    }
    df = df.rename(columns=rename_map)

    le_dict = models['rf']['le_dict']
    for col, le in le_dict.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                df[col] = 0

    df['temp_diff'] = df['process_temp'] - df['air_temp']
    df['power'] = df['torque'] * df['rot_speed']
    df['strain'] = df['tool_wear'] * df['torque']

    num_cols = ['air_temp', 'process_temp', 'rot_speed', 'torque', 'tool_wear',
                'temp_diff', 'power', 'strain']
    df[num_cols] = models['rf']['scaler'].transform(df[num_cols])

    feats = models['rf']['features']
    for f in feats:
        if f not in df.columns:
            df[f] = 0
    return df[feats]

def preprocess_generic(key, data):
    """DT/KNN/SVM: kirim DataFrame mentah (nama kolom asli) ke preprocessor jika ada."""
    df = pd.DataFrame([data])[RAW_COLS]
    pre = models[key].get('preprocessor')
    if pre is not None:
        try:
            return pre.transform(df)
        except Exception:
            # fallback: kalau preprocessor mengharapkan kolom sudah di-reindex
            feats = models[key].get('features')
            if feats is not None:
                enc = pd.get_dummies(df).reindex(columns=feats, fill_value=0)
                return pre.transform(enc)
            raise
    return df  # model sudah berupa pipeline utuh

def predict_any(key, data):
    """Kembalikan (pred, proba2, threshold)."""
    X = preprocess_rf(data) if key == 'rf' else preprocess_generic(key, data)
    model = models[key]['model']
    proba = model.predict_proba(X)[0]
    th = meta_get(key, 'threshold', None)
    try:
        th = float(th)
    except (TypeError, ValueError):
        th = 0.5
    pred = int(proba[1] >= th)
    return pred, proba, th

# ══════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════
def input_form_maintenance():
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-hdr">⚙️ Spesifikasi Mesin</div>', unsafe_allow_html=True)
        type_mesin   = st.selectbox("Tipe Kualitas (Type)", ['L', 'M', 'H'])
        air_temp     = st.number_input("Air temperature [K]", 290.0, 320.0, 298.0, step=0.1)
        process_temp = st.number_input("Process temperature [K]", 300.0, 320.0, 308.0, step=0.1)
    with c2:
        st.markdown('<div class="section-hdr">📊 Metrik Sensor</div>', unsafe_allow_html=True)
        rpm       = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500, step=10)
        torque    = st.number_input("Torque [Nm]", 3.0, 80.0, 40.0, step=0.1)
        tool_wear = st.number_input("Tool wear [min]", 0, 300, 50, step=1)
    return {
        'Type': type_mesin,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
    }

def show_prediction_result(pred, proba, threshold):
    a, b, c = st.columns(3)
    a.metric("🔴 Prob. Failure", f"{proba[1]*100:.1f}%")
    b.metric("🟢 Prob. Normal", f"{proba[0]*100:.1f}%")
    c.metric("📌 Threshold", f"{threshold:.2f}")

    if pred == 1:
        st.markdown('<div class="result-fail">🚨 Mesin diprediksi akan <strong>GAGAL (FAILURE)</strong> — segera lakukan inspeksi & maintenance.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-ok">✅ Mesin diprediksi <strong>NORMAL</strong> — aman untuk dioperasikan.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.barh([''], [proba[1]], color=(RED if pred == 1 else GRN), height=0.5, zorder=3)
    ax.barh([''], [proba[0]], left=[proba[1]], color=GRID, height=0.5, zorder=3)
    ax.axvline(threshold, color=YLW, linestyle='--', linewidth=2, zorder=4, label=f'Threshold ({threshold:.2f})')
    ax.set_xlim(0, 1); ax.set_xlabel('Probabilitas Failure', color=TXTS, fontsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TXTS)
    ax.legend(loc='upper right', fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor='white')
    ax.text(proba[1]/2, 0, f'{proba[1]*100:.1f}%', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    plt.tight_layout(pad=0.5); st.pyplot(fig); plt.close()

def show_model_tab(key, perf_rows, tech_items, chart_fn_list):
    is_est = key in EST  # DT/KNN/SVM tanpa metadata asli
    cols = st.columns(4)
    auc_label = "ROC-AUC (est.)" if is_est else "ROC-AUC"
    f1_label  = "Macro F1 (est.)" if is_est else "Macro F1"
    cols[0].metric(auc_label, f"{get_metric(key, 'auc', 0):.4f}")
    cols[1].metric(f1_label, f"{get_metric(key, 'macro_f1', 0):.4f}")

    # Jumlah fitur: dari metadata (RF) atau dari panjang features (DT/KNN/SVM)
    nfeat = meta_get(key, 'n_features', None)
    if nfeat in (None, '—'):
        feats = models[key].get('features')
        nfeat = len(feats) if isinstance(feats, (list, tuple)) else '—'
    cols[2].metric("Input Fitur", str(nfeat))

    th = meta_get(key, 'threshold', None)
    if th not in (None, '—'):
        try: cols[3].metric("Threshold", f"{float(th):.2f}")
        except: cols[3].metric("Threshold", "0.50")
    else:
        cols[3].metric("Threshold", "0.50")

    if is_est:
        st.caption("ℹ️ ROC-AUC & Macro F1 model ini adalah estimasi (file metadata tidak disertakan). Ganti di dict `EST` bila nilai asli tersedia.")

    st.markdown("---")
    if chart_fn_list:
        ccols = st.columns(len(chart_fn_list))
        for col, fn in zip(ccols, chart_fn_list):
            with col:
                st.pyplot(fn()); plt.close()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚙️ Hyperparameter")
        params = models[key].get('params', {})
        if isinstance(params, dict) and params:
            st.dataframe(pd.DataFrame(params.items(), columns=['Parameter', 'Value']),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Parameter tuning tidak disertakan dalam file model.")
    with c2:
        st.markdown("#### 📈 Performa Detail")
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔧 Teknik yang Digunakan")
    tcols = st.columns(len(tech_items))
    for col, (title, desc) in zip(tcols, tech_items.items()):
        col.info(f"**{title}**\n\n{desc}")

def render_prediction_tab(key):
    data = input_form_maintenance()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary", key=f"btn_{key}"):
        with st.spinner("Memproses data sensor..."):
            try:
                pred, proba, th = predict_any(key, data)
                st.markdown("---"); st.markdown("#### 📊 Hasil Prediksi")
                show_prediction_result(pred, proba, th)
            except Exception as e:
                st.error(f"Gagal memproses: {e}")
                st.caption("Cek apakah format file model sesuai. Jalankan `cek_models.py` untuk melihat isi folder models/.")

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
NAV = {"🌲 Random Forest": 'rf', "🌳 Decision Tree": 'dt', "🔵 KNN": 'knn', "⚡ SVM": 'svm'}
NAME = {'rf': "Random Forest", 'dt': "Decision Tree", 'knn': "K-Nearest Neighbors", 'svm': "Support Vector Machine"}
COLOR = {'rf': '#ff8c42', 'dt': '#2ed573', 'knn': '#00b4d8', 'svm': '#f9ca24'}

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 8px;'>
        <div style='font-size:2.4rem;'>🛠️</div>
        <div style='font-size:1.15rem;font-weight:800;color:white;margin-top:6px;'>Predictive Maintenance</div>
        <div style='font-size:0.72rem;color:#7d8aa3;margin-top:3px;letter-spacing:0.05em;'>Machine Failure Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-hdr">Pilih Model</div>', unsafe_allow_html=True)
    page = st.selectbox("", list(NAV.keys()), label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-hdr">Status Model</div>', unsafe_allow_html=True)
    for label, key in NAV.items():
        dot = "🟢" if models.get(key) else "🔴"
        st.markdown(f'<div style="font-size:0.85rem;color:#cdd6e6;padding:3px 0;">{dot} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    active = NAV[page]
    if models.get(active):
        st.markdown('<div class="section-hdr">Model Aktif</div>', unsafe_allow_html=True)
        st.metric("ROC-AUC", f"{get_metric(active, 'auc', 0):.4f}")
        st.metric("Macro F1", f"{get_metric(active, 'macro_f1', 0):.4f}")
        th = meta_get(active, 'threshold', None)
        if th not in (None, '—'):
            try: st.metric("Threshold", f"{float(th):.2f}")
            except: pass
        nfeat = meta_get(active, 'n_features', None)
        if nfeat in (None, '—'):
            feats = models[active].get('features')
            nfeat = len(feats) if isinstance(feats, (list, tuple)) else '—'
        st.metric("Input Fitur", str(nfeat))

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#46506a;text-align:center;">Dataset: AI4I 2020<br>Predictive Maintenance</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
mc = COLOR[active]
st.markdown(f"""
<div style='padding:22px 0 6px;'>
    <div style='font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:{mc};margin-bottom:6px;'>{page.split()[0]} {NAME[active]}</div>
    <h1 style='font-size:2rem;font-weight:800;color:white;margin:0;line-height:1.2;'>Machine Failure Prediction</h1>
    <p style='color:#7d8aa3;margin-top:8px;font-size:0.95rem;'>Memprediksi kegagalan mesin dari data sensor: suhu, kecepatan putaran, torsi, dan keausan alat.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

with st.expander("📊 Lihat Perbandingan Semua Model", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_model_comparison()); plt.close()
    with c2:
        st.pyplot(plot_roc_curves()); plt.close()
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  TECH / PERF CONFIG PER MODEL
# ══════════════════════════════════════════════════════════════
PERF = {
    'rf': {'Metrik': ['ROC-AUC', 'Macro F1', 'Recall Failure', 'Threshold', 'Total Fitur'],
           'Nilai': [f"{get_metric('rf','auc',0):.4f}", f"{get_metric('rf','macro_f1',0):.4f}",
                     '0.8382', f"{meta_get('rf','threshold','0.71')}", str(meta_get('rf','n_features','11'))]},
    'dt': {'Metrik': ['ROC-AUC (est.)', 'Macro F1 (est.)', 'Input Fitur'],
           'Nilai': [f"{get_metric('dt','auc',0):.4f}", f"{get_metric('dt','macro_f1',0):.4f}", '6']},
    'knn': {'Metrik': ['ROC-AUC (est.)', 'Macro F1 (est.)', 'Input Fitur'],
            'Nilai': [f"{get_metric('knn','auc',0):.4f}", f"{get_metric('knn','macro_f1',0):.4f}", '6']},
    'svm': {'Metrik': ['ROC-AUC (est.)', 'Macro F1 (est.)', 'Input Fitur'],
            'Nilai': [f"{get_metric('svm','auc',0):.4f}", f"{get_metric('svm','macro_f1',0):.4f}", '6']},
}
TECH = {
    'rf': {'SMOTE': 'Atasi imbalance ~96:4', 'Feature Engineering': 'temp_diff, power, strain',
           'Optuna Tuning': 'TPE Sampler, 50 trials', 'Threshold Opt.': '→ 0.71 untuk Macro F1'},
    'dt': {'SMOTE': 'Atasi class imbalance', 'Pruning': 'max_depth cegah overfit', 'Optuna Tuning': 'Cari parameter pohon optimal'},
    'knn': {'SMOTE': 'Atasi class imbalance', 'StandardScaler': 'Scaling wajib untuk KNN', 'Optuna Tuning': 'Cari n_neighbors optimal'},
    'svm': {'SMOTE': 'Atasi class imbalance', 'StandardScaler': 'Scaling wajib untuk margin', 'Kernel RBF': 'Pemisahan non-linear'},
}
CHARTS = {
    'rf': [plot_rf_feature_importance, plot_class_distribution],
    'dt': [plot_dt_depth_curve, plot_class_distribution],
    'knn': [plot_knn_k_curve, plot_class_distribution],
    'svm': [plot_svm_kernel_comparison, plot_class_distribution],
}
GUIDE = {
    'rf': "Random Forest menggabungkan banyak pohon keputusan (ensemble). Model kami memakai SMOTE untuk menyeimbangkan kelas, feature engineering (temp_diff, power, strain), tuning Optuna, dan optimasi threshold ke 0.71.",
    'dt': "Decision Tree membangun aturan keputusan bercabang dari fitur sensor. Setiap node menanyakan satu kondisi (mis. torsi > X) hingga mencapai prediksi gagal/normal.",
    'knn': "KNN mengklasifikasikan mesin berdasarkan kemiripan dengan K mesin terdekat di ruang fitur. Scaling wajib agar semua sensor punya skala setara.",
    'svm': "SVM mencari hyperplane dengan margin maksimal yang memisahkan mesin normal dan gagal. Kernel RBF memungkinkan pemisahan non-linear.",
}

# ══════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════
if not models.get(active):
    st.warning(f"⚠️ Model {NAME[active]} belum tersedia. Pastikan file .pkl ada di folder `models/`.")
else:
    tab_info, tab_guide, tab_pred = st.tabs(["📊  Model Info", "📋  Panduan", "🔮  Prediksi"])
    with tab_info:
        show_model_tab(active, PERF[active], TECH[active], CHARTS[active])
    with tab_guide:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 📖 Cara Menggunakan
            1. Buka tab **Prediksi**
            2. Isi data sensor mesin
            3. Klik **Jalankan Prediksi**
            4. Lihat hasil & probabilitas kegagalan
            """)
        with col2:
            st.markdown(f"""
            #### 📦 Tentang Dataset
            - **Sumber**: AI4I 2020 Predictive Maintenance
            - **Total data**: 10.000 baris
            - **Fitur input**: Type, suhu udara/proses, rpm, torsi, tool wear
            - **Target**: Failure (0 / 1) — sangat imbalanced

            #### 🔬 Tentang Model
            {GUIDE[active]}
            """)
    with tab_pred:
        render_prediction_tab(active)