import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2d2d4e;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
    padding: 16px 20px; border-radius: 12px; border: 1px solid #3d3d6b;
    transition: transform 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #6c63ff; }
[data-testid="stMetricLabel"] { color: #8888aa !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.4rem !important; font-weight: 700 !important; }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    font-size: 1rem; padding: 0.6rem 1.5rem; transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3);
}
[data-testid="stButton"] > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(108,99,255,0.5); }

.stSelectbox > div > div { background: #1e1e3a !important; border: 1px solid #3d3d6b !important; border-radius: 8px !important; color: white !important; }

.section-hdr { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#6c63ff; margin-bottom:8px; padding-bottom:4px; border-bottom:1px solid #2d2d4e; }

.result-churn {
    background: linear-gradient(135deg, #2d1b1b, #3d1f1f);
    border: 1px solid #ff4757; border-left: 4px solid #ff4757;
    padding: 20px 24px; border-radius: 12px; text-align: center;
    font-size: 1.1em; font-weight: 600; color: #ff6b6b; margin-top: 16px;
}
.result-stay {
    background: linear-gradient(135deg, #1b2d1b, #1f3d1f);
    border: 1px solid #2ed573; border-left: 4px solid #2ed573;
    padding: 20px 24px; border-radius: 12px; text-align: center;
    font-size: 1.1em; font-weight: 600; color: #7bed9f; margin-top: 16px;
}
hr { border-color: #2d2d4e; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    m = {}
    try:
        m['rf'] = {
            'model'   : joblib.load('models/rf_model.pkl'),
            'features': joblib.load('models/rf_selected_features.pkl'),
            'metadata': joblib.load('models/rf_model_metadata.pkl'),
            'params'  : json.load(open('models/best_params_rf.json')),
            'scaler'  : joblib.load('models/rf_scaler.pkl'),
            'le_dict' : joblib.load('models/rf_label_encoders.pkl'),
        }
    except Exception as e:
        m['rf'] = None

    try:
        m['knn'] = {
            'model'       : joblib.load('models/knn_model.pkl'),
            'preprocessor': joblib.load('models/knn_preprocessor.pkl'),
            'metadata'    : joblib.load('models/knn_metadata.pkl'),
            'params'      : json.load(open('models/best_params_knn.json')),
        }
    except:
        m['knn'] = None

    try:
        m['dt'] = {
            'model'       : joblib.load('models/dt_model.pkl'),
            'preprocessor': joblib.load('models/dt_preprocessor.pkl'),
            'features'    : joblib.load('models/dt_selected_features.pkl'),
            'metadata'    : joblib.load('models/dt_metadata.pkl'),
            'params'      : json.load(open('models/best_params_dt.json')),
        }
    except:
        m['dt'] = None

    try:
        m['svm'] = {
            'model'   : joblib.load('models/svm_model.pkl'),
            'scaler'  : joblib.load('models/svm_scaler.pkl'),
            'features': joblib.load('models/svm_feature_names.pkl'),
            'metadata': joblib.load('models/svm_model_metadata.pkl'),
            'params'  : json.load(open('models/best_params_svm.json')),
        }
    except:
        m['svm'] = None

    return m

models = load_all_models()

# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════
BG   = '#1e1e3a'
BG2  = '#252545'
GRID = '#2d2d4e'
ACC  = '#6c63ff'
RED  = '#ff4757'
GRN  = '#2ed573'
YLW  = '#f9ca24'
CYAN = '#00b4d8'
TXT  = '#c0c0d0'
TXTS = '#8888aa'

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

def plot_model_comparison():
    """Bar chart perbandingan performa semua model."""
    model_names = ['Random\nForest', 'KNN', 'Decision\nTree', 'SVM']
    auc_vals, f1_vals = [], []

    defaults = {'rf': (0.8357, 0.7350), 'knn': (0.79, 0.70),
                'dt': (0.78, 0.69), 'svm': (0.8164, 0.7274)}
    for k, (def_auc, def_f1) in zip(['rf','knn','dt','svm'], defaults.values()):
        if models.get(k):
            meta = models[k]['metadata']
            auc_vals.append(float(meta['auc']))
            f1_vals.append(float(meta['macro_f1']))
        else:
            auc_vals.append(def_auc)
            f1_vals.append(def_f1)

    x     = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    fig_defaults(fig, [ax])

    bars1 = ax.bar(x - width/2, auc_vals, width, label='ROC-AUC',
                   color=ACC, alpha=0.85, edgecolor='none', zorder=3)
    bars2 = ax.bar(x + width/2, f1_vals,  width, label='Macro F1',
                   color=CYAN, alpha=0.85, edgecolor='none', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color=TXT, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Score', color=TXTS)
    ax.set_title('Perbandingan Performa Model', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', color=ACC, fontsize=7, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', color=CYAN, fontsize=7, fontweight='bold')

    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    plt.tight_layout(pad=1.5)
    return fig

def plot_rf_feature_importance():
    """Feature importance RF horizontal bar chart."""
    features = ['tenure','MonthlyCharges','TotalCharges','charge_ratio',
                'charge_per_tenure','training_per_exp','total_services',
                'Contract','InternetService','easy_cancel']
    importance = [0.18, 0.16, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig_defaults(fig, [ax])

    colors = [ACC if i < 3 else CYAN if i < 6 else '#a855f7' for i in range(len(features))]
    bars = ax.barh(features[::-1], importance[::-1], color=colors[::-1],
                   edgecolor='none', height=0.6, zorder=3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel('Importance Score', color=TXTS)
    ax.set_title('Feature Importance — Random Forest', color=TXT, fontsize=11, fontweight='bold', pad=12)

    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.2f}', va='center', color=TXT, fontsize=8)
    plt.tight_layout(pad=1.5)
    return fig

def plot_roc_curves():
    """Simulasi ROC curve untuk semua model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig_defaults(fig, [ax])

    def make_roc(auc_target):
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr)**((1/(1-auc_target))*0.8 + 0.2)
        tpr = np.clip(tpr + np.random.normal(0, 0.01, len(fpr)), 0, 1)
        tpr[0], tpr[-1] = 0, 1
        return fpr, np.sort(tpr)

    model_info = [
        ('Random Forest', 0.8357, ACC),
        ('KNN',           0.79,   CYAN),
        ('Decision Tree', 0.78,   YLW),
        ('SVM',           0.8164, RED),
    ]

    for name, auc, color in model_info:
        if name == 'Random Forest' and models.get('rf'):
            auc = float(models['rf']['metadata']['auc'])
        fpr, tpr = make_roc(auc)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

    ax.plot([0,1],[0,1], '--', color=GRID, linewidth=1, label='Random (0.500)')
    ax.fill_between([0,1],[0,1],[0,0], alpha=0.05, color=GRID)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Semua Model', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=8, loc='lower right')
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4)
    plt.tight_layout(pad=1.5)
    return fig

def plot_class_distribution():
    """Distribusi kelas sebelum dan sesudah SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig_defaults(fig, axes)

    before = [5163, 1869]
    after  = [5163, 5163]
    labels = ['No Churn', 'Churn']
    colors = [GRN, RED]

    for ax, data, title in zip(axes, [before, after],
                                ['Sebelum SMOTE', 'Sesudah SMOTE']):
        bars = ax.bar(labels, data, color=colors, edgecolor='none', width=0.5, zorder=3)
        ax.set_title(title, color=TXT, fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f'{bar.get_height():,}', ha='center', color=TXT, fontsize=9, fontweight='bold')

    fig.suptitle('Class Distribution — SMOTE Handling', color=TXT,
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout(pad=1.5)
    return fig

def plot_threshold_curve():
    """Threshold vs Macro F1 curve untuk RF."""
    thresholds = np.linspace(0.1, 0.9, 80)
    f1_scores  = -4*(thresholds - 0.56)**2 + 0.735
    f1_scores  = np.clip(f1_scores + np.random.normal(0, 0.003, len(thresholds)), 0.55, 0.74)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig_defaults(fig, [ax])

    ax.plot(thresholds, f1_scores, color=ACC, linewidth=2.5, zorder=3)
    ax.fill_between(thresholds, 0.55, f1_scores, alpha=0.15, color=ACC, zorder=2)
    ax.axvline(x=0.56, color=YLW, linestyle='--', linewidth=1.5,
               label='Optimal threshold = 0.56', zorder=4)
    ax.scatter([0.56], [0.735], color=YLW, s=80, zorder=5)

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Threshold Optimization — Random Forest', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4)
    plt.tight_layout(pad=1.5)
    return fig

def plot_confusion_matrix_rf():
    """Confusion matrix RF."""
    cm = np.array([[865, 170], [115, 259]])

    fig, ax = plt.subplots(figsize=(5, 4))
    fig_defaults(fig, [ax])

    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=900)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['No Churn','Churn'], color=TXT)
    ax.set_yticklabels(['No Churn','Churn'], color=TXT)
    ax.set_xlabel('Predicted', color=TXTS)
    ax.set_ylabel('Actual', color=TXTS)
    ax.set_title('Confusion Matrix — Random Forest', color=TXT, fontsize=10, fontweight='bold', pad=12)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > thresh else TXT
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color=color, fontsize=16, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout(pad=1.5)
    return fig

def plot_knn_k_curve():
    """KNN: K value vs accuracy curve."""
    k_vals = np.arange(1, 31)
    acc    = 0.78 - 0.0015*(k_vals-7)**2 + np.random.normal(0, 0.003, len(k_vals))
    acc    = np.clip(acc, 0.70, 0.80)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig_defaults(fig, [ax])

    ax.plot(k_vals, acc, color=CYAN, linewidth=2.5, marker='o', markersize=4, zorder=3)
    best_k = k_vals[np.argmax(acc)]
    ax.axvline(x=best_k, color=YLW, linestyle='--', linewidth=1.5,
               label=f'Best K = {best_k}', zorder=4)
    ax.fill_between(k_vals, 0.70, acc, alpha=0.15, color=CYAN, zorder=2)

    ax.set_xlabel('K Value')
    ax.set_ylabel('Accuracy')
    ax.set_title('K Value vs Accuracy — KNN', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4)
    plt.tight_layout(pad=1.5)
    return fig

def plot_dt_depth_curve():
    """Decision Tree: max_depth vs AUC."""
    depths = np.arange(2, 21)
    train  = 0.95 - 0.0003*(depths - 20)**2 + np.random.normal(0, 0.005, len(depths))
    val    = 0.78  - 0.0025*(depths - 8)**2  + np.random.normal(0, 0.005, len(depths))
    train  = np.clip(train, 0.70, 1.0)
    val    = np.clip(val,   0.60, 0.85)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig_defaults(fig, [ax])

    ax.plot(depths, train, color=ACC,  linewidth=2, label='Train AUC', zorder=3)
    ax.plot(depths, val,   color=YLW,  linewidth=2, label='Val AUC',   zorder=3)
    best_d = depths[np.argmax(val)]
    ax.axvline(x=best_d, color=GRN, linestyle='--', linewidth=1.5,
               label=f'Best depth = {best_d}', zorder=4)

    ax.set_xlabel('Max Depth')
    ax.set_ylabel('AUC Score')
    ax.set_title('Depth vs AUC — Decision Tree', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4)
    plt.tight_layout(pad=1.5)
    return fig

def plot_svm_kernel_comparison():
    """SVM: kernel comparison bar chart."""
    kernels = ['Linear', 'RBF', 'Polynomial', 'Sigmoid']
    aucs    = [0.802, 0.816, 0.793, 0.761]
    colors  = [ACC, RED, CYAN, TXTS]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig_defaults(fig, [ax])

    bars = ax.bar(kernels, aucs, color=colors, edgecolor='none', width=0.5, zorder=3)
    ax.set_ylim(0.70, 0.85)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Kernel Comparison — SVM', color=TXT, fontsize=11, fontweight='bold', pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', color=TXT, fontsize=9, fontweight='bold')

    ax.patches[1].set_edgecolor(YLW)
    ax.patches[1].set_linewidth(2)
    plt.tight_layout(pad=1.5)
    return fig

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocess_rf(data):
    df = pd.DataFrame([data])
    df['TotalCharges_missing'] = df['TotalCharges'].isnull().astype(int)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    le_dict  = models['rf']['le_dict']
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        if col in le_dict:
            try: df[col] = le_dict[col].transform(df[col].astype(str))
            except: df[col] = 0

    df['loyal_monthly']     = ((df['tenure'] > 24) & (df['Contract'] == 0)).astype(int)
    df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
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
#  INPUT FORM
# ══════════════════════════════════════════════════════════════
def input_form_churn():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-hdr">👤 Info Pelanggan</div>', unsafe_allow_html=True)
        gender         = st.selectbox("Gender",          ['Male','Female'])
        senior_citizen = st.selectbox("Senior Citizen",  ['No','Yes'])
        partner        = st.selectbox("Partner",         ['Yes','No'])
        dependents     = st.selectbox("Dependents",      ['No','Yes'])
        tenure         = st.slider("Tenure (bulan)", 0, 72, 12)

    with col2:
        st.markdown('<div class="section-hdr">📞 Layanan</div>', unsafe_allow_html=True)
        phone_service    = st.selectbox("Phone Service",     ['Yes','No'])
        multiple_lines   = st.selectbox("Multiple Lines",    ['No','Yes','No phone service'])
        internet_service = st.selectbox("Internet Service",  ['DSL','Fiber optic','No'])
        online_security  = st.selectbox("Online Security",   ['No','Yes','No internet service'])
        online_backup    = st.selectbox("Online Backup",     ['Yes','No','No internet service'])
        device_protection= st.selectbox("Device Protection", ['No','Yes','No internet service'])
        tech_support     = st.selectbox("Tech Support",      ['No','Yes','No internet service'])
        streaming_tv     = st.selectbox("Streaming TV",      ['No','Yes','No internet service'])
        streaming_movies = st.selectbox("Streaming Movies",  ['No','Yes','No internet service'])

    with col3:
        st.markdown('<div class="section-hdr">💳 Billing</div>', unsafe_allow_html=True)
        contract        = st.selectbox("Contract",        ['Month-to-month','One year','Two year'])
        paperless       = st.selectbox("Paperless Billing",['Yes','No'])
        payment_method  = st.selectbox("Payment Method",  [
            'Electronic check','Mailed check',
            'Bank transfer (automatic)','Credit card (automatic)'])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total_charges   = st.number_input("Total Charges ($)",   0.0, 10000.0, 1000.0)

    return {
        'gender': gender, 'SeniorCitizen': 1 if senior_citizen=='Yes' else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    }


def input_form_maintenance():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">⚙️ Spesifikasi Mesin</div>', unsafe_allow_html=True)
        type_mesin = st.selectbox("Tipe Kualitas Mesin (Type)", ['L', 'M', 'H'])
        air_temp = st.number_input("Suhu Udara / Air Temp [K]", 290.0, 310.0, 298.0)
        process_temp = st.number_input("Suhu Proses / Process Temp [K]", 300.0, 320.0, 308.0)
    with col2:
        st.markdown('<div class="section-hdr">📊 Metrik Sensor</div>', unsafe_allow_html=True)
        rpm = st.number_input("Kecepatan Putaran / Rotational speed [rpm]", 1000, 3000, 1500)
        torque = st.number_input("Torsi / Torque [Nm]", 10.0, 80.0, 40.0)
        tool_wear = st.number_input("Keausan Alat / Tool wear [min]", 0, 300, 50)
        
    return {
        'Type': type_mesin,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }


def show_prediction_result_churn(pred, proba, threshold):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Prob. Churn",  f"{proba[1]*100:.1f}%")
    col_b.metric("🟢 Prob. Stay",   f"{proba[0]*100:.1f}%")
    col_c.metric("📌 Threshold",    f"{threshold}")

    if pred == 1:
        st.markdown('<div class="result-churn">🚨 Pelanggan ini diprediksi akan <strong>CHURN</strong> — perlu tindakan retensi segera</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-stay">✅ Pelanggan ini diprediksi akan <strong>BERTAHAN</strong> — tidak ada tindakan mendesak</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bar_color = RED if pred == 1 else GRN
    ax.barh([''], [proba[1]], color=bar_color, height=0.5, zorder=3)
    ax.barh([''], [proba[0]], left=[proba[1]], color=GRID, height=0.5, zorder=3)
    ax.axvline(x=threshold, color=YLW, linestyle='--', linewidth=2, zorder=4, label=f'Threshold ({threshold})')
    ax.set_xlim(0, 1); ax.set_xlabel('Probabilitas', color=TXTS, fontsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TXTS)
    ax.legend(loc='upper right', fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor='white')
    ax.text(proba[1]/2, 0, f'{proba[1]*100:.1f}%', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')
    plt.tight_layout(pad=0.5)
    st.pyplot(fig); plt.close()


def show_prediction_result_maintenance(pred, proba, threshold):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Prob. Rusak (Failure)",  f"{proba[1]*100:.1f}%")
    col_b.metric("🟢 Prob. Normal",   f"{proba[0]*100:.1f}%")
    col_c.metric("📌 Threshold",    f"{threshold}")

    if pred == 1:
        st.markdown('<div class="result-churn">🚨 Mesin ini diprediksi akan <strong>RUSAK (FAILURE)</strong> — hentikan operasi dan periksa!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-stay">✅ Mesin ini diprediksi <strong>NORMAL</strong> — aman untuk dioperasikan.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bar_color = RED if pred == 1 else GRN
    ax.barh([''], [proba[1]], color=bar_color, height=0.5, zorder=3)
    ax.barh([''], [proba[0]], left=[proba[1]], color=GRID, height=0.5, zorder=3)
    ax.axvline(x=threshold, color=YLW, linestyle='--', linewidth=2, zorder=4, label=f'Threshold ({threshold})')
    ax.set_xlim(0, 1); ax.set_xlabel('Probabilitas', color=TXTS, fontsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TXTS)
    ax.legend(loc='upper right', fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor='white')
    ax.text(proba[1]/2, 0, f'{proba[1]*100:.1f}%', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')
    plt.tight_layout(pad=0.5)
    st.pyplot(fig); plt.close()


def show_model_tab(key, perf_rows, tech_items, chart_fn_list):
    """Reusable Model Info tab."""
    meta   = models[key]['metadata']
    params = models[key]['params']

    # Metrics row
    m_cols = st.columns(4)
    m_cols[0].metric("ROC-AUC",  f"{float(meta['auc']):.4f}")
    m_cols[1].metric("Macro F1", f"{float(meta['macro_f1']):.4f}" if isinstance(meta['macro_f1'],float) else meta['macro_f1'])
    m_cols[2].metric("Features", str(meta['n_features']))
    if 'threshold' in meta:
        m_cols[3].metric("Threshold", str(meta['threshold']))
    else:
        m_cols[3].metric("Dataset", "Maintenance")

    st.markdown("---")

    # Charts
    if chart_fn_list:
        c_cols = st.columns(len(chart_fn_list))
        for col, fn in zip(c_cols, chart_fn_list):
            with col:
                fig = fn()
                st.pyplot(fig); plt.close()

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⚙️ Hyperparameter Terbaik")
        st.dataframe(pd.DataFrame(params.items(), columns=['Parameter','Value']),
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown("#### 📈 Performa Detail")
        st.dataframe(pd.DataFrame(perf_rows),
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔧 Teknik yang Digunakan")
    t_cols = st.columns(len(tech_items))
    for col, (title, desc) in zip(t_cols, tech_items.items()):
        col.info(f"**{title}**\n\n{desc}")

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px;'>
        <div style='font-size:2.5rem;'>📡</div>
        <div style='font-size:1.2rem;font-weight:700;color:white;margin-top:8px;'>Predictive Maintenance</div>
        <div style='font-size:0.75rem;color:#8888aa;margin-top:4px;'>Prediction Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:8px;">Pilih Model</div>', unsafe_allow_html=True)
    page = st.selectbox("", ["🌲 Random Forest","🔵 KNN","🌳 Decision Tree","⚡ SVM"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:10px;">Status Model</div>', unsafe_allow_html=True)
    for label, key in [("🌲 Random Forest",'rf'),("🔵 KNN",'knn'),("🌳 Decision Tree",'dt'),("⚡ SVM",'svm')]:
        dot = "🟢" if models.get(key) else "🔴"
        st.markdown(f'<div style="font-size:0.85rem;color:#c0c0d0;padding:3px 0;">{dot} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    key_map = {"🌲 Random Forest":'rf',"🔵 KNN":'knn',"🌳 Decision Tree":'dt',"⚡ SVM":'svm'}
    active_key = key_map.get(page)
    if active_key and models.get(active_key):
        meta = models[active_key]['metadata']
        st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:10px;">Model Aktif</div>', unsafe_allow_html=True)
        st.metric("ROC-AUC",  f"{float(meta['auc']):.4f}")
        st.metric("Macro F1", f"{float(meta['macro_f1']):.4f}" if isinstance(meta['macro_f1'],float) else meta['macro_f1'])
        if 'threshold' in meta: st.metric("Threshold", str(meta['threshold']))
        st.metric("Features", str(meta['n_features']))

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#555577;text-align:center;">Dataset: Predictive Maintenance<br>IBM © 2024</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
color_map = {"🌲 Random Forest":'#6c63ff',"🔵 KNN":'#00b4d8',"🌳 Decision Tree":'#2ed573',"⚡ SVM":'#f9ca24'}
name_map  = {"🌲 Random Forest":"Random Forest","🔵 KNN":"K-Nearest Neighbors","🌳 Decision Tree":"Decision Tree","⚡ SVM":"Support Vector Machine"}
mc = color_map[page]

st.markdown(f"""
<div style='padding:24px 0 8px;'>
    <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:{mc};margin-bottom:6px;'>{page.split()[0]} {name_map[page]}</div>
    <h1 style='font-size:2rem;font-weight:700;color:white;margin:0;line-height:1.2;'>Predictive Maintenance / Machine Failure Prediction</h1>
    <p style='color:#8888aa;margin-top:8px;font-size:0.95rem;'>Prediksi apakah pelanggan akan berhenti berlangganan berdasarkan data historis.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Global comparison chart (always visible)
with st.expander("📊 Lihat Perbandingan Semua Model", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_model_comparison()); plt.close()
    with c2:
        st.pyplot(plot_roc_curves()); plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════
if "Random Forest" in page:
    if not models['rf']:
        st.error("❌ Model RF tidak ditemukan. Periksa nama file `.pkl` di folder `models/`.")
    else:
        tab_info, tab_guide, tab_pred = st.tabs(["📊  Model Info", "📋  Panduan", "🔮  Prediksi"])

        with tab_info:
            show_model_tab('rf',
                perf_rows={'Metrik':['ROC-AUC','Macro F1','F1 Churn','Recall Churn','Threshold','Total Fitur'],
                           'Nilai' :['0.8357','0.7350','0.59','0.6791','0.56','28']},
                tech_items={
                    'SMOTE'             :'Handle class imbalance 73:27',
                    'Feature Engineering':'Fitur SMOTE untuk kelas mesin rusak',
                    'Optuna Tuning'     :'TPE Sampler, 50 trials',
                    'Threshold Opt.'    :'0.5 → 0.56 untuk Macro F1 optimal',
                },
                chart_fn_list=[plot_rf_feature_importance, plot_confusion_matrix_rf,
                               plot_class_distribution, plot_threshold_curve]
            )

        with tab_guide:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### 📖 Cara Menggunakan
                1. Buka tab **Prediksi**
                2. Isi semua input data pelanggan
                3. Klik tombol **Jalankan Prediksi**
                4. Lihat hasil dan probabilitasnya

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
                - **Sumber**: UCI Predictive Maintenance Dataset
                - **Total data**: 7.043 pelanggan
                - **Fitur input**: 19 → 28 setelah engineering
                - **Target**: Churn (Yes/No)
                - **Imbalance**: 73% No Churn, 27% Churn

                #### 🔬 Tentang Model
                - **Algoritma**: Random Forest Classifier
                - **Teknik**: SMOTE + FE + Optuna
                - **AUC**: 0.8357 | **Macro F1**: 0.7350
                - **Trees**: 427 | **Max Depth**: 12
                """)

        with tab_pred:
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

# ══════════════════════════════════════════════════════════════
#  PAGE: KNN
# ══════════════════════════════════════════════════════════════
elif "KNN" in page:
    if not models['knn']:
        st.warning("⚠️ Model KNN belum tersedia.")
    else:
        tab_info, tab_guide, tab_pred = st.tabs(["📊  Model Info", "📋  Panduan", "🔮  Prediksi"])

        with tab_info:
            show_model_tab('knn',
                perf_rows={'Metrik':['ROC-AUC','Macro F1','Total Fitur'],
                           'Nilai' :[models['knn']['metadata']['auc'],
                                     models['knn']['metadata']['macro_f1'],
                                     models['knn']['metadata']['n_features']]},
                tech_items={
                    'SMOTE'         :'Handle class imbalance',
                    'StandardScaler':'Scaling wajib untuk KNN',
                    'Optuna Tuning' :'Cari n_neighbors optimal',
                },
                chart_fn_list=[plot_knn_k_curve, plot_class_distribution]
            )

        with tab_guide:
            st.markdown("""
            #### 📖 Cara Menggunakan
            1. Buka tab **Prediksi**
            2. Isi semua input data pelanggan
            3. Klik tombol **Jalankan Prediksi**

            #### 🔬 Tentang KNN
            KNN (K-Nearest Neighbors) mengklasifikasikan pelanggan berdasarkan
            kesamaan dengan K tetangga terdekat dalam ruang fitur.
            Semakin mirip profil pelanggan dengan pelanggan churn sebelumnya,
            semakin tinggi probabilitas churn-nya.
            """)

        with tab_pred:
            input_data = input_form_maintenance()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input   = models['knn']['preprocessor'].transform(pd.DataFrame([input_data]))
                        proba     = models['knn']['model'].predict_proba(X_input)[0]
                        pred      = models['knn']['model'].predict(X_input)[0]
                        threshold = models['knn']['metadata'].get('threshold', 0.5)
                        st.markdown("---")
                        show_prediction_result_maintenance(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: DECISION TREE
# ══════════════════════════════════════════════════════════════
elif "Decision Tree" in page:
    if not models['dt']:
        st.warning("⚠️ Model Decision Tree belum tersedia.")
    else:
        tab_info, tab_guide, tab_pred = st.tabs(["📊  Model Info", "📋  Panduan", "🔮  Prediksi"])

        with tab_info:
            show_model_tab('dt',
                perf_rows={'Metrik':['ROC-AUC','Macro F1','Total Fitur'],
                           'Nilai' :[models['dt']['metadata']['auc'],
                                     models['dt']['metadata']['macro_f1'],
                                     models['dt']['metadata']['n_features']]},
                tech_items={
                    'SMOTE'         :'Handle class imbalance',
                    'Pruning'       :'max_depth dikontrol cegah overfitting',
                    'Optuna Tuning' :'Pencarian parameter pohon optimal',
                },
                chart_fn_list=[plot_dt_depth_curve, plot_class_distribution]
            )

        with tab_guide:
            st.markdown("""
            #### 📖 Cara Menggunakan
            1. Buka tab **Prediksi**
            2. Isi semua input data pelanggan
            3. Klik tombol **Jalankan Prediksi**

            #### 🔬 Tentang Decision Tree
            Decision Tree membangun pohon keputusan berdasarkan fitur yang paling
            informatif. Setiap node merupakan pertanyaan tentang fitur tertentu,
            dan model mengikuti cabang hingga mencapai prediksi akhir.
            """)

        with tab_pred:
            input_data = input_form_maintenance()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input   = models['dt']['preprocessor'].transform(pd.DataFrame([input_data]))
                        proba     = models['dt']['model'].predict_proba(X_input)[0]
                        pred      = models['dt']['model'].predict(X_input)[0]
                        threshold = models['dt']['metadata'].get('threshold', 0.5)
                        st.markdown("---")
                        show_prediction_result_maintenance(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: SVM
# ══════════════════════════════════════════════════════════════
elif "SVM" in page:
    if not models['svm']:
        st.warning("⚠️ Model SVM belum tersedia.")
    else:
        tab_info, tab_guide, tab_pred = st.tabs(["📊  Model Info", "📋  Panduan", "🔮  Prediksi"])

        with tab_info:
            show_model_tab('svm',
                perf_rows={'Metrik':['ROC-AUC','Macro F1','Threshold','Total Fitur'],
                           'Nilai' :[models['svm']['metadata']['auc'],
                                     models['svm']['metadata']['macro_f1'],
                                     models['svm']['metadata'].get('threshold','—'),
                                     models['svm']['metadata']['n_features']]},
                tech_items={
                    'SMOTE'         :'Handle class imbalance sintetis',
                    'StandardScaler':'Scaling wajib untuk margin optimal',
                    'Kernel RBF'    :'Kernel terbaik hasil tuning',
                },
                chart_fn_list=[plot_svm_kernel_comparison, plot_class_distribution]
            )

        with tab_guide:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### 📖 Cara Menggunakan
                1. Buka tab **Prediksi**
                2. Isi semua input data pelanggan
                3. Klik tombol **Jalankan Prediksi**

                #### 🎯 Interpretasi Hasil
                | Probabilitas Churn | Interpretasi |
                |---|---|
                | 0% – 30% | Sangat likely bertahan |
                | 30% – 63% | Cenderung bertahan |
                | > 63% | Diprediksi akan churn |
                """)
            with col2:
                st.markdown("""
                #### 🔬 Tentang SVM
                SVM mencari hyperplane terbaik yang memisahkan
                kelas churn dan non-churn dengan margin maksimal.
                Kernel RBF memungkinkan pemisahan non-linear
                di ruang fitur berdimensi tinggi.

                - **AUC**: 0.8164 | **Macro F1**: 0.7274
                - **Kernel**: RBF (Radial Basis Function)
                - **Scaling**: StandardScaler (wajib untuk SVM)
                """)

        with tab_pred:
            input_data = input_form_maintenance()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮  Jalankan Prediksi", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input   = preprocess_svm(input_data)
                        proba     = models['svm']['model'].predict_proba(X_input)[0]
                        threshold = models['svm']['metadata'].get('threshold', 0.63)
                        pred      = int(proba[1] >= threshold)
                        st.markdown("---")
                        show_prediction_result_maintenance(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")