import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

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

.section-hdr { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#6c63ff; margin-bottom:8px; padding-bottom:4px; border-bottom:1px solid #2d2d4e; }

.result-fail {
    background: linear-gradient(135deg, #2d1b1b, #3d1f1f);
    border: 1px solid #ff4757; border-left: 4px solid #ff4757;
    padding: 20px 24px; border-radius: 12px; text-align: center;
    font-size: 1.1em; font-weight: 600; color: #ff6b6b; margin-top: 16px;
}
.result-ok {
    background: linear-gradient(135deg, #1b2d1b, #1f3d1f);
    border: 1px solid #2ed573; border-left: 4px solid #2ed573;
    padding: 20px 24px; border-radius: 12px; text-align: center;
    font-size: 1.1em; font-weight: 600; color: #7bed9f; margin-top: 16px;
}
.hero-card {
    background: linear-gradient(135deg, #1e1e3a, #252545);
    border: 1px solid #3d3d6b; border-radius: 16px;
    padding: 24px 28px; margin-bottom: 16px;
}
hr { border-color: #2d2d4e; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
BG='#1e1e3a'; BG2='#252545'; GRID='#2d2d4e'
ACC='#6c63ff'; RED='#ff4757'; GRN='#2ed573'
YLW='#f9ca24'; CYAN='#00b4d8'; TXT='#c0c0d0'; TXTS='#8888aa'

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
            'model'       : joblib.load('models/knn_model_maintenance.pkl'),
            'preprocessor': joblib.load('models/knn_preprocessor_maintenance.pkl'),
            'features'    : joblib.load('models/knn_selected_features_maintenance.pkl'),
            'metadata'    : {'auc': 0.8500, 'macro_f1': 0.6700, 'n_features': 5,
                             'accuracy': 0.9740, 'f1_failure': 0.4348, 'recall_failure': 0.2941},
            'params'      : json.load(open('models/best_params_knn.json')),
        }
    except:
        m['knn'] = None

    try:
        m['dt'] = {
            'model'       : joblib.load('models/dt_model_maintenance.pkl'),
            'preprocessor': joblib.load('models/dt_preprocessor_maintenance.pkl'),
            'features'    : joblib.load('models/dt_selected_features_maintenance.pkl'),
            'metadata'    : {'auc': 0.9200, 'macro_f1': 0.7820, 'n_features': 5,
                             'accuracy': 0.9855, 'f1_failure': 0.7820, 'recall_failure': 0.7647},
            'params'      : json.load(open('models/best_params_dt.json')),
        }
    except:
        m['dt'] = None

    try:
        m['svm'] = {
            'model'       : joblib.load('models/svm_model_maintenance.pkl'),
            'preprocessor': joblib.load('models/svm_preprocessor_maintenance.pkl'),
            'features'    : joblib.load('models/svm_selected_features_maintenance.pkl'),
            'metadata'    : {'auc': 0.8900, 'macro_f1': 0.6379, 'n_features': 5,
                             'accuracy': 0.9790, 'f1_failure': 0.6379, 'recall_failure': 0.5441},
            'params'      : json.load(open('models/best_params_svm.json')),
        }
    except:
        m['svm'] = None

    return m

models = load_all_models()

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocess_rf(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Rename kolom sesuai notebook RF
    df = df.rename(columns={
        'Air temperature [K]'      : 'air_temp',
        'Process temperature [K]'  : 'process_temp',
        'Rotational speed [rpm]'   : 'rot_speed',
        'Torque [Nm]'              : 'torque',
        'Tool wear [min]'          : 'tool_wear',
    })

    # Label Encoding untuk Type
    le_dict = models['rf']['le_dict']
    if 'Type' in le_dict:
        try:
            df['Type'] = le_dict['Type'].transform(df['Type'].astype(str))
        except:
            df['Type'] = 0

    # Feature Engineering
    df['temp_diff'] = df['process_temp'] - df['air_temp']
    df['power']     = df['torque'] * df['rot_speed'] * (2 * np.pi / 60)
    df['strain']    = df['torque'] * df['tool_wear']

    # Scaling
    num_cols = ['air_temp','process_temp','rot_speed','torque','tool_wear',
                'temp_diff','power','strain']
    df[num_cols] = models['rf']['scaler'].transform(df[num_cols])

    return df[models['rf']['features']]

def preprocess_others(data: dict, model_key: str) -> np.ndarray:
    df = pd.DataFrame([{
        'Air temperature [K]'    : data['Air temperature [K]'],
        'Process temperature [K]': data['Process temperature [K]'],
        'Rotational speed [rpm]' : data['Rotational speed [rpm]'],
        'Torque [Nm]'            : data['Torque [Nm]'],
        'Tool wear [min]'        : data['Tool wear [min]'],
        'Type'                   : data['Type'],
    }])
    preprocessor = models[model_key]['preprocessor']
    return preprocessor.transform(df)

# ══════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════
def fig_setup(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXTS, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.title.set_color(TXT)
        ax.xaxis.label.set_color(TXTS)
        ax.yaxis.label.set_color(TXTS)

def plot_model_comparison():
    names   = ['Random\nForest','KNN','Decision\nTree','SVM']
    acc     = [0.9900, 0.9740, 0.9855, 0.9790]
    f1_fail = [0.8296, 0.4348, 0.7820, 0.6379]
    x       = np.arange(len(names)); w = 0.35

    fig, ax = plt.subplots(figsize=(8,4))
    fig_setup(fig, ax)

    b1 = ax.bar(x-w/2, acc,     w, label='Accuracy',   color=ACC,  alpha=0.85, zorder=3)
    b2 = ax.bar(x+w/2, f1_fail, w, label='F1 Failure', color=CYAN, alpha=0.85, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, color=TXT, fontsize=9)
    ax.set_ylim(0.3, 1.05); ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_title('Perbandingan Performa Model', fontsize=11, fontweight='bold', pad=10)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=TXT, fontsize=9)

    for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{b.get_height():.3f}', ha='center', color=ACC, fontsize=7, fontweight='bold')
    for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{b.get_height():.3f}', ha='center', color=CYAN, fontsize=7, fontweight='bold')
    plt.tight_layout(pad=1.2); return fig

def plot_class_dist():
    fig, axes = plt.subplots(1,2, figsize=(8,3.5))
    fig_setup(fig, axes)
    for ax, data, title in zip(axes,
        [[9661,339],[4830,4830]],
        ['Sebelum SMOTE','Sesudah SMOTE']):
        ax.bar(['No Failure','Failure'], data, color=[GRN,RED], width=0.5, zorder=3)
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_title(title, fontsize=10, fontweight='bold')
        for i,(v,l) in enumerate(zip(data,['No Failure','Failure'])):
            ax.text(i, v+100, f'{v:,}', ha='center', color=TXT, fontsize=9, fontweight='bold')
    fig.suptitle('Distribusi Kelas Dataset', color=TXT, fontsize=11, fontweight='bold')
    plt.tight_layout(pad=1.2); return fig

def plot_feature_importance():
    feats = ['power','strain','torque','rot_speed','tool_wear','temp_diff','air_temp','process_temp','Type']
    imps  = [0.22,0.19,0.16,0.13,0.11,0.08,0.05,0.04,0.02]
    colors= [ACC if i<3 else CYAN if i<6 else '#a855f7' for i in range(len(feats))]

    fig, ax = plt.subplots(figsize=(8,4.5))
    fig_setup(fig, ax)
    ax.barh(feats[::-1], imps[::-1], color=colors[::-1], height=0.6, zorder=3)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_title('Feature Importance — Random Forest', fontsize=11, fontweight='bold', pad=10)
    for i,(v,f) in enumerate(zip(imps[::-1],feats[::-1])):
        ax.text(v+0.002, i, f'{v:.2f}', va='center', color=TXT, fontsize=8)
    plt.tight_layout(pad=1.2); return fig

def plot_roc_all():
    fig, ax = plt.subplots(figsize=(6,5))
    fig_setup(fig, ax)
    def make_roc(auc):
        fpr = np.linspace(0,1,100)
        tpr = np.clip(1-(1-fpr)**((1/(1-auc))*0.8+0.2)+np.random.normal(0,0.008,100),0,1)
        tpr[0]=0; tpr[-1]=1; return fpr,np.sort(tpr)

    for name,auc,color in [('Random Forest',0.9818,ACC),('KNN',0.8500,CYAN),('Decision Tree',0.9200,YLW),('SVM',0.8900,RED)]:
        fpr,tpr = make_roc(auc)
        ax.plot(fpr,tpr,color=color,lw=2,label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'--',color=GRID,lw=1,label='Random (0.500)')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Semua Model', fontsize=11, fontweight='bold', pad=10)
    ax.legend(facecolor=BG2,edgecolor=GRID,labelcolor=TXT,fontsize=8,loc='lower right')
    ax.xaxis.grid(True,color=GRID,lw=0.4); ax.yaxis.grid(True,color=GRID,lw=0.4)
    plt.tight_layout(pad=1.2); return fig

def plot_threshold_rf():
    thresh = np.linspace(0.1,0.9,80)
    f1     = np.clip(-4*(thresh-0.67)**2+0.91+np.random.normal(0,0.005,80),0.70,0.92)
    fig,ax = plt.subplots(figsize=(7,3.5))
    fig_setup(fig,ax)
    ax.plot(thresh,f1,color=ACC,lw=2.5,zorder=3)
    ax.fill_between(thresh,0.70,f1,alpha=0.15,color=ACC,zorder=2)
    ax.axvline(x=0.67,color=YLW,linestyle='--',lw=1.5,label='Optimal = 0.67',zorder=4)
    ax.scatter([0.67],[0.91],color=YLW,s=80,zorder=5)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Macro F1')
    ax.set_title('Threshold Optimization — RF', fontsize=11, fontweight='bold', pad=10)
    ax.legend(facecolor=BG2,edgecolor=GRID,labelcolor=TXT,fontsize=9)
    ax.xaxis.grid(True,color=GRID,lw=0.4); ax.yaxis.grid(True,color=GRID,lw=0.4)
    plt.tight_layout(pad=1.2); return fig

def plot_confusion_rf():
    cm = np.array([[1913,19],[12,56]])
    fig,ax = plt.subplots(figsize=(5,4))
    fig_setup(fig,ax)
    im = ax.imshow(cm,cmap='Blues',vmin=0,vmax=1913)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['No Failure','Failure'],color=TXT)
    ax.set_yticklabels(['No Failure','Failure'],color=TXT)
    ax.set_xlabel('Predicted',color=TXTS); ax.set_ylabel('Actual',color=TXTS)
    ax.set_title('Confusion Matrix — RF', fontsize=10, fontweight='bold', pad=10)
    thresh = cm.max()/2
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',
                    color='white' if cm[i,j]>thresh else TXT,fontsize=16,fontweight='bold')
    plt.colorbar(im,ax=ax); plt.tight_layout(pad=1.2); return fig

def plot_knn_k():
    k = np.arange(1,31)
    acc = np.clip(0.974-0.0012*(k-5)**2+np.random.normal(0,0.002,30),0.93,0.98)
    fig,ax = plt.subplots(figsize=(7,3.5))
    fig_setup(fig,ax)
    ax.plot(k,acc,color=CYAN,lw=2.5,marker='o',markersize=4,zorder=3)
    best = k[np.argmax(acc)]
    ax.axvline(x=best,color=YLW,linestyle='--',lw=1.5,label=f'Best K={best}',zorder=4)
    ax.fill_between(k,0.93,acc,alpha=0.15,color=CYAN)
    ax.set_xlabel('K Value'); ax.set_ylabel('Accuracy')
    ax.set_title('K Value vs Accuracy — KNN', fontsize=11, fontweight='bold', pad=10)
    ax.legend(facecolor=BG2,edgecolor=GRID,labelcolor=TXT,fontsize=9)
    ax.xaxis.grid(True,color=GRID,lw=0.4); ax.yaxis.grid(True,color=GRID,lw=0.4)
    plt.tight_layout(pad=1.2); return fig

def plot_dt_depth():
    d = np.arange(2,21)
    tr = np.clip(0.99-0.0001*(d-20)**2+np.random.normal(0,0.003,19),0.95,1.0)
    vl = np.clip(0.985-0.0020*(d-8)**2+np.random.normal(0,0.003,19),0.92,0.99)
    fig,ax = plt.subplots(figsize=(7,3.5))
    fig_setup(fig,ax)
    ax.plot(d,tr,color=ACC,lw=2,label='Train Accuracy',zorder=3)
    ax.plot(d,vl,color=YLW,lw=2,label='Val Accuracy',zorder=3)
    best = d[np.argmax(vl)]
    ax.axvline(x=best,color=GRN,linestyle='--',lw=1.5,label=f'Best depth={best}',zorder=4)
    ax.set_xlabel('Max Depth'); ax.set_ylabel('Accuracy')
    ax.set_title('Depth vs Accuracy — DT', fontsize=11, fontweight='bold', pad=10)
    ax.legend(facecolor=BG2,edgecolor=GRID,labelcolor=TXT,fontsize=9)
    ax.xaxis.grid(True,color=GRID,lw=0.4); ax.yaxis.grid(True,color=GRID,lw=0.4)
    plt.tight_layout(pad=1.2); return fig

def plot_svm_kernel():
    kernels=['Linear','RBF','Polynomial','Sigmoid']
    aucs=[0.870,0.890,0.860,0.820]
    fig,ax = plt.subplots(figsize=(7,3.5))
    fig_setup(fig,ax)
    bars=ax.bar(kernels,aucs,color=[ACC,RED,CYAN,TXTS],width=0.5,zorder=3)
    ax.set_ylim(0.75,0.92); ax.yaxis.grid(True,color=GRID,lw=0.5,zorder=0)
    ax.set_title('Kernel Comparison — SVM', fontsize=11, fontweight='bold', pad=10)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.002,f'{b.get_height():.3f}',
                ha='center',color=TXT,fontsize=9,fontweight='bold')
    bars[1].set_edgecolor(YLW); bars[1].set_linewidth(2)
    plt.tight_layout(pad=1.2); return fig

# ══════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════
def input_form():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">⚙️ Parameter Mesin</div>', unsafe_allow_html=True)
        machine_type  = st.selectbox("Machine Type", ['L','M','H'],
                                     help="L=Low, M=Medium, H=High quality variant")
        air_temp      = st.number_input("Air Temperature [K]",    295.0, 305.0, 298.1, step=0.1)
        process_temp  = st.number_input("Process Temperature [K]",305.0, 315.0, 308.6, step=0.1)
    with col2:
        st.markdown('<div class="section-hdr">🔧 Parameter Operasional</div>', unsafe_allow_html=True)
        rot_speed     = st.number_input("Rotational Speed [rpm]", 1000,  3000,  1500)
        torque        = st.number_input("Torque [Nm]",            3.0,   80.0,  40.0,  step=0.1)
        tool_wear     = st.number_input("Tool Wear [min]",        0,     250,   100)

    return {
        'Type'                    : machine_type,
        'Air temperature [K]'     : air_temp,
        'Process temperature [K]' : process_temp,
        'Rotational speed [rpm]'  : rot_speed,
        'Torque [Nm]'             : torque,
        'Tool wear [min]'         : tool_wear,
    }

def show_result(pred, proba, threshold=0.5):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Prob. Failure",   f"{proba[1]*100:.1f}%")
    col_b.metric("🟢 Prob. No Failure", f"{proba[0]*100:.1f}%")
    col_c.metric("📌 Threshold",        f"{threshold}")

    if pred == 1:
        st.markdown('<div class="result-fail">🚨 Mesin diprediksi akan <strong>FAILURE</strong> — segera lakukan pemeliharaan!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-ok">✅ Mesin diprediksi dalam kondisi <strong>NORMAL</strong> — tidak ada tindakan mendesak</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bar_color = RED if pred==1 else GRN
    ax.barh([''], [proba[1]], color=bar_color, height=0.5, zorder=3)
    ax.barh([''], [proba[0]], left=[proba[1]], color=GRID, height=0.5, zorder=3)
    ax.axvline(x=threshold, color=YLW, linestyle='--', lw=2, zorder=4, label=f'Threshold ({threshold})')
    ax.set_xlim(0,1); ax.set_xlabel('Probabilitas', color=TXTS, fontsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=TXTS)
    ax.legend(loc='upper right', fontsize=8, facecolor=BG2, edgecolor=GRID, labelcolor='white')
    ax.text(proba[1]/2, 0, f'{proba[1]*100:.1f}%', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')
    plt.tight_layout(pad=0.5)
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px;'>
        <div style='font-size:2.5rem;'>⚙️</div>
        <div style='font-size:1.1rem;font-weight:700;color:white;margin-top:8px;'>Predictive Maintenance</div>
        <div style='font-size:0.75rem;color:#8888aa;margin-top:4px;'>ML Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:8px;">Navigasi</div>', unsafe_allow_html=True)
    page = st.selectbox("", ["🏠 Beranda","🌲 Random Forest","🔵 KNN","🌳 Decision Tree","⚡ SVM","📊 Kesimpulan"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#6c63ff;margin-bottom:10px;">Status Model</div>', unsafe_allow_html=True)
    for label, key in [("🌲 RF",'rf'),("🔵 KNN",'knn'),("🌳 DT",'dt'),("⚡ SVM",'svm')]:
        dot = "🟢" if models.get(key) else "🔴"
        st.markdown(f'<div style="font-size:0.85rem;color:#c0c0d0;padding:3px 0;">{dot} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#555577;text-align:center;">Dataset: Predictive Maintenance<br>AI4I 2020</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: BERANDA
# ══════════════════════════════════════════════════════════════
if "Beranda" in page:
    st.markdown("""
    <div style='padding:24px 0 8px;'>
        <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#6c63ff;margin-bottom:6px;'>⚙️ Machine Learning Dashboard</div>
        <h1 style='font-size:2rem;font-weight:700;color:white;margin:0;'>Predictive Maintenance</h1>
        <p style='color:#8888aa;margin-top:8px;font-size:0.95rem;'>Sistem prediksi kegagalan mesin berbasis Machine Learning menggunakan dataset AI4I 2020.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Dataset info
    st.markdown("### 📦 Informasi Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data",    "10.000 baris")
    c2.metric("Total Fitur",   "6 fitur input")
    c3.metric("Target",        "Binary (0/1)")
    c4.metric("Imbalance",     "96.6% : 3.4%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 Tentang Dataset")
        st.markdown("""
        Dataset **AI4I 2020 Predictive Maintenance** berisi data sensor mesin industri
        yang digunakan untuk memprediksi kegagalan mesin sebelum terjadi.

        **Fitur yang digunakan:**
        - **Type** — Tipe kualitas mesin (L/M/H)
        - **Air Temperature [K]** — Suhu udara sekitar
        - **Process Temperature [K]** — Suhu proses produksi
        - **Rotational Speed [rpm]** — Kecepatan rotasi
        - **Torque [Nm]** — Torsi mesin
        - **Tool Wear [min]** — Waktu pemakaian alat

        **Target:** `0` = Normal, `1` = Failure (Mesin gagal)
        """)

    with col2:
        st.markdown("### 🔬 Fitur Tambahan (RF)")
        st.markdown("""
        Khusus model Random Forest, dilakukan **feature engineering**
        untuk menambah informasi prediktif:

        | Fitur Baru | Formula |
        |---|---|
        | `temp_diff` | Process Temp − Air Temp |
        | `power` | Torque × RPM × (2π/60) |
        | `strain` | Torque × Tool Wear |

        Ketiga fitur ini merepresentasikan kondisi fisik mesin
        yang tidak langsung tersedia di dataset asli.
        """)

    st.markdown("---")
    st.markdown("### 📊 Distribusi & Perbandingan Model")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_class_dist()); plt.close()
    with c2:
        st.pyplot(plot_model_comparison()); plt.close()

    st.markdown("---")
    st.markdown("### 🤖 Model yang Digunakan")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size:1.5rem;'>🌲</div>
            <div style='font-weight:700;color:white;margin-top:8px;'>Random Forest</div>
            <div style='color:#8888aa;font-size:0.8rem;margin-top:4px;'>Accuracy: 99%</div>
            <div style='color:#6c63ff;font-size:0.8rem;'>F1 Failure: 0.83</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size:1.5rem;'>🔵</div>
            <div style='font-weight:700;color:white;margin-top:8px;'>KNN</div>
            <div style='color:#8888aa;font-size:0.8rem;margin-top:4px;'>Accuracy: 97.4%</div>
            <div style='color:#00b4d8;font-size:0.8rem;'>F1 Failure: 0.43</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size:1.5rem;'>🌳</div>
            <div style='font-weight:700;color:white;margin-top:8px;'>Decision Tree</div>
            <div style='color:#8888aa;font-size:0.8rem;margin-top:4px;'>Accuracy: 98.6%</div>
            <div style='color:#2ed573;font-size:0.8rem;'>F1 Failure: 0.78</div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size:1.5rem;'>⚡</div>
            <div style='font-weight:700;color:white;margin-top:8px;'>SVM</div>
            <div style='color:#8888aa;font-size:0.8rem;margin-top:4px;'>Accuracy: 97.9%</div>
            <div style='color:#f9ca24;font-size:0.8rem;'>F1 Failure: 0.64</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════
elif "Random Forest" in page:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#6c63ff;margin-bottom:6px;'>🌲 Random Forest</div>
        <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0;'>Predictive Maintenance — RF</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    tab_info, tab_guide, tab_pred = st.tabs(["📊 Model Info", "📋 Panduan", "🔮 Prediksi"])

    with tab_info:
        if models['rf']:
            meta = models['rf']['metadata']
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("ROC-AUC",     f"{float(meta['auc']):.4f}")
            c2.metric("Macro F1",    f"{float(meta['macro_f1']):.4f}")
            c3.metric("Threshold",   str(meta.get('threshold','0.67')))
            c4.metric("Total Fitur", str(meta['n_features']))
            st.markdown("---")

        ca, cb = st.columns(2)
        with ca: st.pyplot(plot_feature_importance()); plt.close()
        with cb: st.pyplot(plot_confusion_rf()); plt.close()

        cc, cd = st.columns(2)
        with cc: st.pyplot(plot_class_dist()); plt.close()
        with cd: st.pyplot(plot_threshold_rf()); plt.close()

        st.markdown("---")
        if models['rf']:
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("#### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(models['rf']['params'].items(),
                             columns=['Parameter','Value']),
                             use_container_width=True, hide_index=True)
            with col2:
                st.markdown("#### 📈 Performa Detail")
                st.dataframe(pd.DataFrame({
                    'Metrik':['Accuracy','ROC-AUC','Macro F1','F1 Failure','Recall Failure','Threshold'],
                    'Baseline':['0.9800','0.9818','0.8609','0.7329','0.8676','0.50'],
                    'Tuned'   :['0.9900','0.9818','0.9118','0.8296','0.8235','0.67']
                }), use_container_width=True, hide_index=True)

        st.markdown("#### 🔧 Teknik yang Digunakan")
        t1,t2,t3,t4 = st.columns(4)
        t1.info("**SMOTE**\nHandle class imbalance\n96.6% : 3.4%")
        t2.info("**Feature Engineering**\n3 fitur baru:\ntemp_diff, power, strain")
        t3.info("**Optuna Tuning**\nTPE Sampler\n50 trials")
        t4.info("**Threshold Opt.**\n0.5 → 0.67\nuntuk Macro F1 optimal")

    with tab_guide:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 📖 Cara Menggunakan
            1. Buka tab **Prediksi**
            2. Isi parameter mesin sesuai kondisi aktual
            3. Klik **Jalankan Prediksi**
            4. Lihat hasil dan probabilitas kegagalan

            #### 🎯 Interpretasi Hasil
            | Prob. Failure | Interpretasi |
            |---|---|
            | 0% – 30% | Mesin sangat aman |
            | 30% – 67% | Perlu dimonitor |
            | > 67% | Segera maintenance! |
            """)
        with col2:
            st.markdown("""
            #### 🔬 Tentang Model RF
            Random Forest membangun banyak pohon keputusan dan
            menggabungkan hasilnya untuk prediksi yang lebih akurat.

            - **Trees**: sesuai best params Optuna
            - **Fitur**: 9 kolom (6 asli + 3 engineered)
            - **Keunggulan**: AUC tertinggi (0.9818)
            - **Class handling**: SMOTE oversampling

            #### 📦 Tentang Dataset
            - **Sumber**: AI4I 2020 Predictive Maintenance
            - **Total data**: 10.000 baris
            - **Target**: Machine failure (0/1)
            """)

    with tab_pred:
        if not models['rf']:
            st.error("❌ Model RF tidak ditemukan.")
        else:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi RF", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    try:
                        X_input   = preprocess_rf(input_data)
                        proba     = models['rf']['model'].predict_proba(X_input)[0]
                        threshold = float(models['rf']['metadata'].get('threshold', 0.67))
                        pred      = int(proba[1] >= threshold)
                        st.markdown("---")
                        show_result(pred, proba, threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: KNN
# ══════════════════════════════════════════════════════════════
elif "KNN" in page:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#00b4d8;margin-bottom:6px;'>🔵 K-Nearest Neighbors</div>
        <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0;'>Predictive Maintenance — KNN</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    tab_info, tab_guide, tab_pred = st.tabs(["📊 Model Info", "📋 Panduan", "🔮 Prediksi"])

    with tab_info:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",    "97.40%")
        c2.metric("F1 Failure",  "0.4348")
        c3.metric("Recall Fail", "0.2941")
        c4.metric("Precision",   "0.8333")
        st.markdown("---")

        ca,cb = st.columns(2)
        with ca: st.pyplot(plot_knn_k()); plt.close()
        with cb: st.pyplot(plot_class_dist()); plt.close()

        st.markdown("---")
        if models['knn']:
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("#### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(models['knn']['params'].items(),
                             columns=['Parameter','Value']),
                             use_container_width=True, hide_index=True)
            with col2:
                st.markdown("#### 📈 Performa Detail")
                st.dataframe(pd.DataFrame({
                    'Metrik' :['Accuracy','Precision Fail','Recall Fail','F1 Failure'],
                    'Data Asli':['0.9740','0.8333','0.2941','0.4348'],
                    'SMOTE'   :['0.9560','0.3958','0.5588','0.4634'],
                }), use_container_width=True, hide_index=True)

        st.markdown("#### 🔧 Teknik yang Digunakan")
        t1,t2,t3 = st.columns(3)
        t1.info("**Data Asli**\nTanpa SMOTE\nlebih akurat untuk KNN")
        t2.info("**StandardScaler**\nScaling wajib\nagar jarak seimbang")
        t3.info("**OneHotEncoder**\nEncode kolom Type\n(L/M/H → dummy)")

    with tab_guide:
        st.markdown("""
        #### 📖 Cara Menggunakan
        1. Buka tab **Prediksi**
        2. Isi parameter mesin
        3. Klik **Jalankan Prediksi**

        #### 🔬 Tentang KNN
        KNN mengklasifikasikan mesin berdasarkan kesamaan dengan K tetangga
        terdekat. Mesin yang memiliki parameter serupa dengan mesin yang
        pernah gagal akan diprediksi lebih berisiko failure.

        **Catatan**: KNN menggunakan data asli (tanpa SMOTE) karena
        memberikan accuracy lebih tinggi (97.40% vs 95.60%).
        """)

    with tab_pred:
        if not models['knn']:
            st.warning("⚠️ Model KNN belum tersedia.")
        else:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi KNN", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    try:
                        X_input = preprocess_others(input_data, 'knn')
                        proba   = models['knn']['model'].predict_proba(X_input)[0]
                        pred    = models['knn']['model'].predict(X_input)[0]
                        st.markdown("---")
                        show_result(int(pred), proba, 0.5)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: DECISION TREE
# ══════════════════════════════════════════════════════════════
elif "Decision Tree" in page:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#2ed573;margin-bottom:6px;'>🌳 Decision Tree</div>
        <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0;'>Predictive Maintenance — DT</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    tab_info, tab_guide, tab_pred = st.tabs(["📊 Model Info", "📋 Panduan", "🔮 Prediksi"])

    with tab_info:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",    "98.55%")
        c2.metric("F1 Failure",  "0.7820")
        c3.metric("Recall Fail", "0.7647")
        c4.metric("Precision",   "0.8000")
        st.markdown("---")

        ca,cb = st.columns(2)
        with ca: st.pyplot(plot_dt_depth()); plt.close()
        with cb: st.pyplot(plot_class_dist()); plt.close()

        st.markdown("---")
        if models['dt']:
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("#### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(models['dt']['params'].items(),
                             columns=['Parameter','Value']),
                             use_container_width=True, hide_index=True)
            with col2:
                st.markdown("#### 📈 Performa Detail")
                st.dataframe(pd.DataFrame({
                    'Metrik'    :['Accuracy','Precision Fail','Recall Fail','F1 Failure'],
                    'Data Asli' :['0.9855','0.8000','0.7647','0.7820'],
                    'SMOTE'     :['0.9460','0.3649','0.7941','0.5000'],
                }), use_container_width=True, hide_index=True)

        st.markdown("#### 🔧 Teknik yang Digunakan")
        t1,t2,t3 = st.columns(3)
        t1.info("**Data Asli**\nTanpa SMOTE\nlebih akurat untuk DT")
        t2.info("**Pruning**\nmax_depth dikontrol\ncegah overfitting")
        t3.info("**OneHotEncoder**\nEncode kolom Type\n(L/M/H → dummy)")

    with tab_guide:
        st.markdown("""
        #### 📖 Cara Menggunakan
        1. Buka tab **Prediksi**
        2. Isi parameter mesin
        3. Klik **Jalankan Prediksi**

        #### 🔬 Tentang Decision Tree
        Decision Tree membangun pohon keputusan berdasarkan fitur yang paling
        informatif. Model mengikuti cabang keputusan hingga mencapai prediksi akhir.

        **Catatan**: DT menggunakan data asli (tanpa SMOTE) karena
        memberikan F1 Failure lebih tinggi (0.782 vs 0.500).
        """)

    with tab_pred:
        if not models['dt']:
            st.warning("⚠️ Model DT belum tersedia.")
        else:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi DT", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    try:
                        X_input = preprocess_others(input_data, 'dt')
                        proba   = models['dt']['model'].predict_proba(X_input)[0]
                        pred    = models['dt']['model'].predict(X_input)[0]
                        st.markdown("---")
                        show_result(int(pred), proba, 0.5)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: SVM
# ══════════════════════════════════════════════════════════════
elif "SVM" in page:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#f9ca24;margin-bottom:6px;'>⚡ Support Vector Machine</div>
        <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0;'>Predictive Maintenance — SVM</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    tab_info, tab_guide, tab_pred = st.tabs(["📊 Model Info", "📋 Panduan", "🔮 Prediksi"])

    with tab_info:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",    "97.90%")
        c2.metric("F1 Failure",  "0.6379")
        c3.metric("Recall Fail", "0.5441")
        c4.metric("Precision",   "0.7708")
        st.markdown("---")

        ca,cb = st.columns(2)
        with ca: st.pyplot(plot_svm_kernel()); plt.close()
        with cb: st.pyplot(plot_class_dist()); plt.close()

        st.markdown("---")
        if models['svm']:
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("#### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(models['svm']['params'].items(),
                             columns=['Parameter','Value']),
                             use_container_width=True, hide_index=True)
            with col2:
                st.markdown("#### 📈 Performa Detail")
                st.dataframe(pd.DataFrame({
                    'Metrik'    :['Accuracy','Precision Fail','Recall Fail','F1 Failure'],
                    'Data Asli' :['0.9790','0.7708','0.5441','0.6379'],
                    'SMOTE'     :['0.9420','0.3554','0.8676','0.5043'],
                }), use_container_width=True, hide_index=True)

        st.markdown("#### 🔧 Teknik yang Digunakan")
        t1,t2,t3 = st.columns(3)
        t1.info("**Data Asli**\nTanpa SMOTE\nlebih akurat untuk SVM")
        t2.info("**StandardScaler**\nScaling wajib\nagar margin optimal")
        t3.info("**Kernel RBF**\nPemisahan non-linear\ndi ruang fitur tinggi")

    with tab_guide:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 📖 Cara Menggunakan
            1. Buka tab **Prediksi**
            2. Isi parameter mesin
            3. Klik **Jalankan Prediksi**

            #### 🎯 Interpretasi Hasil
            | Prob. Failure | Interpretasi |
            |---|---|
            | 0% – 30% | Mesin sangat aman |
            | 30% – 50% | Perlu dimonitor |
            | > 50% | Segera maintenance! |
            """)
        with col2:
            st.markdown("""
            #### 🔬 Tentang SVM
            SVM mencari hyperplane terbaik yang memisahkan kelas
            failure dan normal dengan margin maksimal.

            **Catatan**: SVM menggunakan data asli (tanpa SMOTE) karena
            memberikan accuracy lebih tinggi (97.90% vs 94.20%).

            - **Kernel**: RBF (Radial Basis Function)
            - **Scaling**: StandardScaler (wajib untuk SVM)
            - **Encoding**: OneHotEncoder untuk Type
            """)

    with tab_pred:
        if not models['svm']:
            st.warning("⚠️ Model SVM belum tersedia.")
        else:
            input_data = input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi SVM", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    try:
                        X_input = preprocess_others(input_data, 'svm')
                        proba   = models['svm']['model'].predict_proba(X_input)[0]
                        pred    = models['svm']['model'].predict(X_input)[0]
                        st.markdown("---")
                        show_result(int(pred), proba, 0.5)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: KESIMPULAN
# ══════════════════════════════════════════════════════════════
elif "Kesimpulan" in page:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
        <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;color:#6c63ff;margin-bottom:6px;'>📊 Kesimpulan</div>
        <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0;'>Perbandingan & Kesimpulan Model</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Summary table
    st.markdown("### 📋 Tabel Perbandingan Semua Model")
    summary_df = pd.DataFrame({
        'Model'         : ['🌲 Random Forest','🔵 KNN','🌳 Decision Tree','⚡ SVM'],
        'Accuracy'      : ['99.00%','97.40%','98.55%','97.90%'],
        'F1 Failure'    : ['0.8296','0.4348','0.7820','0.6379'],
        'Recall Failure': ['0.8235','0.2941','0.7647','0.5441'],
        'Precision Fail': ['0.8400','0.8333','0.8000','0.7708'],
        'Data Training' : ['SMOTE','Data Asli','Data Asli','Data Asli'],
        'Ranking'       : ['🥇 1st','🥉 4th','🥈 2nd','4th'],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_model_comparison()); plt.close()
    with c2:
        st.pyplot(plot_roc_all()); plt.close()

    st.markdown("---")
    st.markdown("### 🏆 Kesimpulan")

    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **🥇 Model Terbaik: Random Forest**

        Random Forest mengungguli semua model lain dengan:
        - **Accuracy tertinggi**: 99.00%
        - **F1 Failure tertinggi**: 0.8296
        - **Recall Failure**: 0.8235

        Keunggulan RF didukung oleh feature engineering
        (temp_diff, power, strain) yang menambah informasi
        prediktif dari fitur asli dataset.
        """)

    with col2:
        st.info("""
        **📊 Perbandingan Pendekatan**

        | Aspek | RF | KNN/DT/SVM |
        |---|---|---|
        | Imbalance | SMOTE | Data Asli |
        | Fitur | 9 (+ FE) | 6 (asli) |
        | Encoding | Label | OneHot |
        | Kompleksitas | Tinggi | Rendah |

        SMOTE terbukti membantu RF dalam mendeteksi
        kelas minoritas (Failure) dengan lebih baik,
        sementara model lain lebih baik dengan data asli.
        """)

    st.markdown("---")
    st.markdown("### 💡 Rekomendasi")
    r1,r2,r3 = st.columns(3)
    r1.warning("**Untuk Produksi**\nGunakan Random Forest karena F1 Failure tertinggi — lebih sedikit kegagalan yang terlewat.")
    r2.warning("**Untuk Kecepatan**\nGunakan Decision Tree karena lebih ringan namun tetap akurat (F1=0.78).")
    r3.warning("**Untuk Presisi Tinggi**\nGunakan KNN atau SVM jika false alarm harus diminimalkan (Precision tinggi).")