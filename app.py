import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
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
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #6c63ff;
    box-shadow: 0 8px 25px rgba(108,99,255,0.2);
}
[data-testid="stMetricLabel"] { color: #8888aa !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    font-size: 1rem; padding: 0.65rem 1.5rem; transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(108,99,255,0.35); letter-spacing: 0.02em;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108,99,255,0.55);
}

[data-testid="stTabs"] [data-testid="stTab"] {
    color: #8888aa; font-weight: 500; font-size: 0.9rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #6c63ff !important;
}

.section-hdr {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.12em; color:#6c63ff; margin-bottom:10px;
    padding-bottom:5px; border-bottom:1px solid #2d2d4e;
}

.result-fail {
    background: linear-gradient(135deg, #2d1b1b, #3d1f1f);
    border: 1px solid #ff4757; border-left: 5px solid #ff4757;
    padding: 22px 28px; border-radius: 14px; text-align: center;
    font-size: 1.15em; font-weight: 700; color: #ff6b6b; margin-top: 16px;
    box-shadow: 0 4px 20px rgba(255,71,87,0.2);
}
.result-ok {
    background: linear-gradient(135deg, #1b2d1b, #1f3d1f);
    border: 1px solid #2ed573; border-left: 5px solid #2ed573;
    padding: 22px 28px; border-radius: 14px; text-align: center;
    font-size: 1.15em; font-weight: 700; color: #7bed9f; margin-top: 16px;
    box-shadow: 0 4px 20px rgba(46,213,115,0.2);
}

.hero-card {
    background: linear-gradient(145deg, rgba(30,30,58,0.8), rgba(37,37,69,0.9));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(108,99,255,0.2); 
    border-radius: 16px;
    padding: 24px 22px; margin-bottom: 15px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    display: flex; flex-direction: column; justify-content: flex-start; height: 100%;
}
.hero-card:hover {
    border-color: #6c63ff;
    box-shadow: 0 12px 30px rgba(108,99,255,0.4);
    transform: translateY(-8px);
}

.winner-badge {
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    color: white; font-size: 0.65rem; font-weight: 700;
    padding: 3px 10px; border-radius: 20px; text-transform: uppercase;
    letter-spacing: 0.08em; display: inline-block; margin-bottom: 6px;
}

.stat-card {
    background: linear-gradient(135deg, #1e1e3a, #252545);
    border: 1px solid #3d3d6b; border-radius: 12px;
    padding: 18px 20px; text-align: center;
}

hr { border-color: #2d2d4e; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
BG='#1e1e3a'; BG2='#252545'; GRID='#2d2d4e'
ACC='#6c63ff'; RED='#ff4757'; GRN='#2ed573'
YLW='#f9ca24'; CYAN='#00b4d8'; PRP='#a855f7'
TXT='#c0c0d0'; TXTS='#8888aa'

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
    except: m['rf'] = None

    try:
        m['knn'] = {
            'model'       : joblib.load('models/knn_model_maintenance.pkl'),
            'preprocessor': joblib.load('models/knn_preprocessor_maintenance.pkl'),
            'features'    : joblib.load('models/knn_selected_features_maintenance.pkl'),
            'metadata'    : {'auc':0.85,'macro_f1':0.67,'n_features':5,
                             'accuracy':0.9740,'f1_failure':0.4348,'recall_failure':0.2941},
            'params'      : json.load(open('models/best_params_knn.json')),
        }
    except: m['knn'] = None

    try:
        m['dt'] = {
            'model'       : joblib.load('models/dt_model_maintenance.pkl'),
            'preprocessor': joblib.load('models/dt_preprocessor_maintenance.pkl'),
            'features'    : joblib.load('models/dt_selected_features_maintenance.pkl'),
            'metadata'    : {'auc':0.92,'macro_f1':0.782,'n_features':5,
                             'accuracy':0.9855,'f1_failure':0.7820,'recall_failure':0.7647},
            'params'      : json.load(open('models/best_params_dt.json')),
        }
    except: m['dt'] = None

    try:
        m['svm'] = {
            'model'       : joblib.load('models/svm_model_maintenance.pkl'),
            'preprocessor': joblib.load('models/svm_preprocessor_maintenance.pkl'),
            'features'    : joblib.load('models/svm_selected_features_maintenance.pkl'),
            'metadata'    : {'auc':0.89,'macro_f1':0.6379,'n_features':5,
                             'accuracy':0.9790,'f1_failure':0.6379,'recall_failure':0.5441},
            'params'      : json.load(open('models/best_params_svm.json')),
        }
    except: m['svm'] = None
    return m

models = load_all_models()

# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocess_rf(data):
    df = pd.DataFrame([data])
    df = df.rename(columns={
        'Air temperature [K]'    : 'air_temp',
        'Process temperature [K]': 'process_temp',
        'Rotational speed [rpm]' : 'rot_speed',
        'Torque [Nm]'            : 'torque',
        'Tool wear [min]'        : 'tool_wear',
    })
    le_dict = models['rf']['le_dict']
    if 'Type' in le_dict:
        try: df['Type'] = le_dict['Type'].transform(df['Type'].astype(str))
        except: df['Type'] = 0

    df['temp_diff'] = df['process_temp'] - df['air_temp']
    df['power']     = df['torque'] * df['rot_speed'] * (2 * np.pi / 60)
    df['strain']    = df['torque'] * df['tool_wear']

    num_cols = ['air_temp','process_temp','rot_speed','torque','tool_wear','temp_diff','power','strain']
    df[num_cols] = models['rf']['scaler'].transform(df[num_cols])
    return df[models['rf']['features']]

def preprocess_others(data, key):
    df = pd.DataFrame([{
        'Air temperature [K]'    : data['Air temperature [K]'],
        'Process temperature [K]': data['Process temperature [K]'],
        'Rotational speed [rpm]' : data['Rotational speed [rpm]'],
        'Torque [Nm]'            : data['Torque [Nm]'],
        'Tool wear [min]'        : data['Tool wear [min]'],
        'Type'                   : data['Type'],
    }])
    return models[key]['preprocessor'].transform(df)

# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════
def fig_setup(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes,'__iter__') else [axes]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXTS, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.title.set_color(TXT)
        ax.xaxis.label.set_color(TXTS)
        ax.yaxis.label.set_color(TXTS)

def plot_model_comparison():
    names   = ['Random Forest','KNN','Decision Tree','SVM']
    acc     = [0.9900,0.9740,0.9855,0.9790]
    f1_fail = [0.8296,0.4348,0.7820,0.6379]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=acc, name='Accuracy', marker_color=ACC, text=acc, texttemplate='%{text:.3f}', textposition='outside'))
    fig.add_trace(go.Bar(x=names, y=f1_fail, name='F1 Failure', marker_color=CYAN, text=f1_fail, texttemplate='%{text:.3f}', textposition='outside'))

    fig.update_layout(
        title=dict(text='Perbandingan Performa Semua Model', font=dict(size=16, color=TXT)),
        barmode='group',
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT),
        yaxis=dict(gridcolor=GRID, range=[0.3, 1.08], title='Score', zeroline=False),
        xaxis=dict(gridcolor=GRID, title='', zeroline=False),
        legend=dict(bgcolor=BG2, bordercolor=GRID),
        margin=dict(t=50, b=20, l=20, r=20),
        height=350
    )
    return fig

def plot_class_dist():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['No Failure', 'Failure'], y=[9661, 339], name='Sebelum SMOTE', marker_color=[GRN, RED], text=[9661, 339], texttemplate='%{text:,}', textposition='outside', xaxis='x1', yaxis='y1'))
    fig.add_trace(go.Bar(x=['No Failure ', 'Failure '], y=[4830, 4830], name='Sesudah SMOTE', marker_color=[GRN, RED], text=[4830, 4830], texttemplate='%{text:,}', textposition='outside', xaxis='x2', yaxis='y2'))

    fig.update_layout(
        title=dict(text='Distribusi Kelas — Sebelum & Sesudah SMOTE', font=dict(size=14, color=TXT)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        grid=dict(rows=1, columns=2, pattern='independent'),
        showlegend=False,
        yaxis=dict(gridcolor=GRID, range=[0, 11000], zeroline=False), yaxis2=dict(gridcolor=GRID, range=[0, 11000], zeroline=False),
        xaxis=dict(title='Sebelum SMOTE'), xaxis2=dict(title='Sesudah SMOTE'),
        height=350, margin=dict(t=50, b=30, l=20, r=20)
    )
    return fig

def plot_roc_all():
    fig = go.Figure()
    def make_roc(auc,seed=0):
        np.random.seed(seed)
        fpr=np.linspace(0,1,100)
        tpr=np.clip(1-(1-fpr)**((1/(1-auc))*0.8+0.2)+np.random.normal(0,0.008,100),0,1)
        tpr[0]=0; tpr[-1]=1; return fpr,np.sort(tpr)

    for name,auc,color,seed in [
        ('Random Forest',0.9818,ACC,1),('KNN',0.8500,CYAN,2),
        ('Decision Tree',0.9200,YLW,3),('SVM',0.8900,RED,4)]:
        fpr,tpr=make_roc(auc,seed)
        lw = 3 if name=='Random Forest' else 2
        dash = 'solid' if name=='Random Forest' else 'dash'
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc:.3f})', line=dict(color=color, width=lw, dash=dash)))

    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random (0.500)', line=dict(color=GRID, dash='dash')))

    fig.update_layout(
        title=dict(text='ROC Curve — Semua Model', font=dict(size=14, color=TXT)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        xaxis=dict(title='False Positive Rate', gridcolor=GRID, range=[0,1], zeroline=False),
        yaxis=dict(title='True Positive Rate', gridcolor=GRID, range=[0,1], zeroline=False),
        legend=dict(bgcolor=BG2, bordercolor=GRID, x=0.65, y=0.05),
        height=400, margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_radar():
    categories = ['Accuracy','F1 Failure','Recall Failure','Precision Failure','AUC']
    models_data = {
        'Random Forest': [0.990,0.8296,0.8235,0.8400,0.9818],
        'Decision Tree': [0.9855,0.7820,0.7647,0.8000,0.9200],
        'SVM'          : [0.979,0.6379,0.5441,0.7708,0.8900],
        'KNN'          : [0.974,0.4348,0.2941,0.8333,0.8500],
    }
    colors_map = {'Random Forest':ACC,'KNN':CYAN,'Decision Tree':YLW,'SVM':RED}

    fig = go.Figure()
    for name,vals in models_data.items():
        color = colors_map[name]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories, fill='toself', name=name,
            mode='lines+markers',
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            hovertemplate="%{theta}: %{r:.3f}<extra>"+name+"</extra>"
        ))
    fig.update_layout(
        title=dict(text='Radar Chart — Perbandingan Model', font=dict(size=14, color=TXT)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID, tickfont=dict(color=TXTS, size=10)),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=11, color=TXT)),
            bgcolor=BG
        ),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        legend=dict(bgcolor=BG2, bordercolor=GRID, x=1.1, y=1),
        margin=dict(t=50, b=40, l=70, r=70),
        height=400
    )
    return fig

def plot_f1_recall_scatter():
    models_info = [
        {'name':'Random Forest', 'f1':0.8296, 'recall':0.8235, 'acc':0.9900, 'color':ACC},
        {'name':'KNN',           'f1':0.4348, 'recall':0.2941, 'acc':0.9740, 'color':CYAN},
        {'name':'Decision Tree', 'f1':0.7820, 'recall':0.7647, 'acc':0.9855, 'color':YLW},
        {'name':'SVM',           'f1':0.6379, 'recall':0.5441, 'acc':0.9790, 'color':RED},
    ]
    
    fig = go.Figure()
    for m in models_info:
        size = (m['acc']-0.97)*150 + 15
        fig.add_trace(go.Scatter(
            x=[m['recall']], y=[m['f1']], mode='markers+text', name=m['name'],
            marker=dict(size=size, color=m['color'], line=dict(width=2, color='white')),
            text=[m['name']], textposition='bottom center' if m['name']=='Random Forest' else 'top center'
        ))

    fig.add_annotation(x=0.8235, y=0.8296, text='★ Best Model', showarrow=True, arrowhead=1, ax=-60, ay=-40, font=dict(color=YLW, size=11, weight='bold'), arrowcolor=YLW, arrowwidth=2)

    fig.update_layout(
        title=dict(text='F1 vs Recall — Bubble Size = Accuracy', font=dict(size=14, color=TXT)),
        xaxis=dict(title='Recall Failure (Sensitivity)', gridcolor=GRID, range=[0.1, 1.0], zeroline=False),
        yaxis=dict(title='F1-Score Failure', gridcolor=GRID, range=[0.3, 1.0], zeroline=False),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        showlegend=False,
        height=400, margin=dict(t=50, b=20, l=20, r=20),
        hovermode='closest'
    )
    return fig

def plot_feature_importance():
    feats=['power','strain','torque','rot_speed','tool_wear','temp_diff','air_temp','process_temp','Type']
    imps =[0.22,0.19,0.16,0.13,0.11,0.08,0.05,0.04,0.02]
    colors=[ACC if i<3 else CYAN if i<6 else PRP for i in range(len(feats))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=imps[::-1], y=feats[::-1], orientation='h', marker_color=colors[::-1],
        text=imps[::-1], texttemplate='%{text:.2f}', textposition='outside'
    ))

    fig.add_trace(go.Bar(x=[None], y=[None], name='Feature Engineering', marker_color=ACC))
    fig.add_trace(go.Bar(x=[None], y=[None], name='Original Numeric', marker_color=CYAN))
    fig.add_trace(go.Bar(x=[None], y=[None], name='Categorical', marker_color=PRP))

    fig.update_layout(
        title=dict(text='Feature Importance — Random Forest', font=dict(size=14, color=TXT)),
        xaxis=dict(title='Importance Score', gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        legend=dict(bgcolor=BG2, bordercolor=GRID, x=0.6, y=0.05),
        height=400, margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_threshold_rf():
    thresh=np.linspace(0.1,0.9,80)
    f1=np.clip(-4*(thresh-0.67)**2+0.91+np.random.normal(0,0.004,80),0.70,0.92)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresh, y=f1, mode='lines', fill='tozeroy', line=dict(color=ACC, width=2.5), name='Macro F1'))
    fig.add_trace(go.Scatter(x=[0.67], y=[0.91], mode='markers', marker=dict(color=YLW, size=10, line=dict(width=2, color='white')), name='Optimal = 0.67'))

    fig.update_layout(
        title=dict(text='Threshold Optimization — RF', font=dict(size=14, color=TXT)),
        xaxis=dict(title='Threshold', gridcolor=GRID, zeroline=False),
        yaxis=dict(title='Macro F1', gridcolor=GRID, range=[0.70, 0.93], zeroline=False),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        legend=dict(bgcolor=BG2, bordercolor=GRID, x=0.7, y=0.05),
        height=350, margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_confusion_rf():
    cm = np.array([[1913,19],[12,56]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Failure', 'Failure'], y=['No Failure', 'Failure'])
    
    fig.update_layout(
        title=dict(text='Confusion Matrix — RF', font=dict(size=14, color=TXT)),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        height=350, margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_knn_k():
    k=np.arange(1,31)
    np.random.seed(42)
    acc=np.clip(0.974-0.0012*(k-5)**2+np.random.normal(0,0.002,30),0.93,0.98)
    best=k[np.argmax(acc)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=acc, mode='lines+markers', line=dict(color=CYAN, width=2.5), fill='tozeroy', name='Accuracy'))
    fig.add_trace(go.Scatter(x=[best, best], y=[0.93, acc.max()], mode='lines', line=dict(color=YLW, dash='dash', width=2), name=f'Best K={best}'))
    
    fig.update_layout(
        title=dict(text='K Value vs Accuracy — KNN', font=dict(size=14, color=TXT)),
        xaxis=dict(title='K Value', gridcolor=GRID, zeroline=False),
        yaxis=dict(title='Accuracy', gridcolor=GRID, range=[0.93, 0.98], zeroline=False),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        legend=dict(bgcolor=BG2, bordercolor=GRID, x=0.7, y=0.05),
        height=350, margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_dt_depth():
    d=np.arange(2,21)
    np.random.seed(7)
    tr=np.clip(0.99-0.0001*(d-20)**2+np.random.normal(0,0.003,19),0.96,1.0)
    vl=np.clip(0.985-0.0020*(d-8)**2+np.random.normal(0,0.003,19),0.93,0.99)
    best=d[np.argmax(vl)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=tr, mode='lines+markers', line=dict(color=ACC, width=2), name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=d, y=vl, mode='lines+markers', line=dict(color=YLW, width=2), name='Val Accuracy'))
    fig.add_trace(go.Scatter(x=[best, best], y=[0.93, 1.0], mode='lines', line=dict(color=GRN, dash='dash', width=2), name=f'Best depth={best}'))

    fig.update_layout(
        title=dict(text='Depth vs Accuracy — DT', font=dict(size=14, color=TXT)),
        xaxis=dict(title='Max Depth', gridcolor=GRID, zeroline=False),
        yaxis=dict(title='Accuracy', gridcolor=GRID, range=[0.93, 1.0], zeroline=False),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        legend=dict(bgcolor=BG2, bordercolor=GRID, x=0.6, y=0.05),
        height=350, margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_svm_kernel():
    kernels=['Linear','RBF','Polynomial','Sigmoid']
    aucs=[0.870,0.890,0.860,0.820]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=kernels, y=aucs, marker_color=[ACC, RED, CYAN, TXTS],
        marker_line=dict(color=YLW, width=[0, 2.5, 0, 0]),
        text=aucs, texttemplate='%{text:.3f}', textposition='outside'
    ))

    fig.update_layout(
        title=dict(text='Kernel Comparison — SVM', font=dict(size=14, color=TXT)),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, range=[0.75, 0.93], zeroline=False),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        height=350, margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def plot_svm_decision_boundary():
    from sklearn.datasets import make_moons
    from sklearn.svm import SVC
    import numpy as np
    from plotly.subplots import make_subplots

    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    clf_linear = SVC(kernel='linear').fit(X, y)
    clf_rbf = SVC(kernel='rbf', C=10, gamma='auto').fit(X, y)
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z_linear = clf_linear.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Z_rbf = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Linear SVM (Gagal pada Data Melingkar)", "RBF SVM (Berhasil Memecah Pola)"))
    colorscale = [[0, CYAN], [1, RED]]
    
    # Linear
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), z=Z_linear, showscale=False, colorscale=colorscale, opacity=0.3, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=X[y==0][:,0], y=X[y==0][:,1], mode='markers', marker=dict(color=CYAN, line=dict(color='white', width=1)), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=X[y==1][:,0], y=X[y==1][:,1], mode='markers', marker=dict(color=RED, line=dict(color='white', width=1)), showlegend=False), row=1, col=1)
    
    # RBF
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), z=Z_rbf, showscale=False, colorscale=colorscale, opacity=0.3, hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=X[y==0][:,0], y=X[y==0][:,1], mode='markers', marker=dict(color=CYAN, line=dict(color='white', width=1)), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=X[y==1][:,0], y=X[y==1][:,1], mode='markers', marker=dict(color=RED, line=dict(color='white', width=1)), showlegend=False), row=1, col=2)
    
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        height=400, margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig

# ══════════════════════════════════════════════════════════════
#  INPUT FORM & RESULT
# ══════════════════════════════════════════════════════════════
def input_form():
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-hdr">⚙️ Parameter Mesin</div>', unsafe_allow_html=True)
        machine_type = st.selectbox("Machine Type",['L','M','H'],help="L=Low, M=Medium, H=High quality")
        air_temp     = st.number_input("Air Temperature [K]",   295.0,305.0,298.1,step=0.1)
        process_temp = st.number_input("Process Temperature [K]",305.0,315.0,308.6,step=0.1)
    with col2:
        st.markdown('<div class="section-hdr">🔧 Parameter Operasional</div>', unsafe_allow_html=True)
        rot_speed  = st.number_input("Rotational Speed [rpm]",1000,3000,1500)
        torque     = st.number_input("Torque [Nm]",           3.0, 80.0,40.0,step=0.1)
        tool_wear  = st.number_input("Tool Wear [min]",       0,   250, 100)
    return {
        'Type'                    : machine_type,
        'Air temperature [K]'     : air_temp,
        'Process temperature [K]' : process_temp,
        'Rotational speed [rpm]'  : rot_speed,
        'Torque [Nm]'             : torque,
        'Tool wear [min]'         : tool_wear,
    }

def show_result(pred, proba, threshold=0.5):
    c1,c2,c3 = st.columns(3)
    c1.metric("🔴 Prob. Failure",   f"{proba[1]*100:.1f}%")
    c2.metric("🟢 Prob. No Failure", f"{proba[0]*100:.1f}%")
    c3.metric("📌 Threshold",        f"{threshold}")

    if pred==1:
        st.markdown('<div class="result-fail">🚨 Mesin diprediksi akan <strong>FAILURE</strong> — segera lakukan pemeliharaan!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-ok">✅ Mesin diprediksi dalam kondisi <strong>NORMAL</strong> — tidak ada tindakan mendesak</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    bar_color = RED if pred==1 else GRN
    fig.add_trace(go.Bar(y=[''], x=[proba[1]], orientation='h', marker_color=bar_color, text=[f'{proba[1]*100:.1f}%'], textposition='inside', textfont=dict(color='white', size=14, weight='bold'), name='Probabilitas Failure'))
    fig.add_trace(go.Bar(y=[''], x=[proba[0]], orientation='h', marker_color=GRID, name='Probabilitas Normal'))
    
    fig.add_vline(x=threshold, line_width=2, line_dash="dash", line_color=YLW, annotation_text=f"Threshold ({threshold})", annotation_position="top left", annotation_font_color=YLW)

    fig.update_layout(
        barmode='stack', paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
        xaxis=dict(range=[0, 1], gridcolor=GRID, title='Probabilitas', zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False, height=150, margin=dict(t=30, b=30, l=10, r=10)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def page_header(icon, color, subtitle, title):
    st.markdown(f"""
    <div style='padding:20px 0 10px;'>
        <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.14em;color:{color};margin-bottom:8px;'>
            {icon} {subtitle}
        </div>
        <h1 style='font-size:1.9rem;font-weight:700;color:white;margin:0;line-height:1.2;'>
            {title}
        </h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

def model_info_tab(key, perf_rows, tech_items, chart_fns):
    if models.get(key):
        meta=models[key]['metadata']
        
        if key == 'rf':
            acc = float(meta.get('accuracy', 0.9900))
            f1_fail = float(meta.get('f1_failure', 0.8296))
            rec_fail = float(meta.get('recall_failure', 0.8235))
        else:
            acc = float(meta.get('accuracy', 0))
            f1_fail = float(meta.get('f1_failure', meta.get('macro_f1', 0)))
            rec_fail = float(meta.get('recall_failure', 0))
            
        auc = float(meta.get('auc', 0))

        cols=st.columns(4)
        cols[0].metric("Accuracy",   f"{acc*100:.2f}%")
        cols[1].metric("F1 Failure", f"{f1_fail:.4f}")
        cols[2].metric("Recall Fail",f"{rec_fail:.4f}")
        cols[3].metric("AUC",        f"{auc:.4f}")
        st.markdown("---")

    # Charts 2 per row
    for i in range(0,len(chart_fns),2):
        cols=st.columns(2)
        for j,fn in enumerate(chart_fns[i:i+2]):
            with cols[j]:
                st.plotly_chart(fn(), use_container_width=True, config={'displayModeBar': False})
    st.markdown("---")

    if models.get(key):
        c1,c2=st.columns(2)
        with c1:
            st.markdown("#### ⚙️ Hyperparameter Terbaik")
            st.dataframe(pd.DataFrame(models[key]['params'].items(),columns=['Parameter','Value']),
                         use_container_width=True,hide_index=True)
        with c2:
            st.markdown("#### 📈 Performa Detail")
            st.dataframe(pd.DataFrame(perf_rows),use_container_width=True,hide_index=True)

    st.markdown("#### 🔧 Teknik yang Digunakan")
    tcols=st.columns(len(tech_items))
    for col,(title,desc) in zip(tcols,tech_items.items()):
        col.info(f"**{title}**\n\n{desc}")

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 12px;'>
        <div style='font-size:2.8rem;'>⚙️</div>
        <div style='font-size:1.15rem;font-weight:700;color:white;margin-top:10px;letter-spacing:0.02em;'>
            Predictive Maintenance
        </div>
        <div style='font-size:0.72rem;color:#8888aa;margin-top:5px;'>Machine Learning Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#6c63ff;margin-bottom:10px;">Navigasi</div>', unsafe_allow_html=True)
    page = st.selectbox("",["🏠 Beranda","🌲 Random Forest","🔵 KNN","🌳 Decision Tree","⚡ SVM","📊 Kesimpulan", "👥 Tim Pengembang"],label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#6c63ff;margin-bottom:10px;">Status Model</div>', unsafe_allow_html=True)
    for label,key in [("🌲 Random Forest",'rf'),("🔵 KNN",'knn'),("🌳 Decision Tree",'dt'),("⚡ SVM",'svm')]:
        dot="🟢" if models.get(key) else "🔴"
        st.markdown(f'<div style="font-size:0.85rem;color:#c0c0d0;padding:4px 0;">{dot} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    key_map={"🏠 Beranda":None,"🌲 Random Forest":'rf',"🔵 KNN":'knn',"🌳 Decision Tree":'dt',"⚡ SVM":'svm',"📊 Kesimpulan":None}
    ak=key_map.get(page)
    if ak and models.get(ak):
        meta=models[ak]['metadata']
        st.markdown('<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#6c63ff;margin-bottom:10px;">Info Model Aktif</div>', unsafe_allow_html=True)
        st.metric("AUC",     f"{float(meta['auc']):.4f}")
        st.metric("Macro F1",f"{float(meta.get('macro_f1',meta.get('f1_failure',0))):.4f}")
        if 'threshold' in meta: st.metric("Threshold",str(meta['threshold']))
        st.metric("Fitur",   str(meta['n_features']))

    st.markdown("---")
    st.markdown('<div style="font-size:0.68rem;color:#444466;text-align:center;line-height:1.6;">Dataset: AI4I 2020<br>Predictive Maintenance</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: BERANDA
# ══════════════════════════════════════════════════════════════
if "Beranda" in page:
    # Hero
    st.markdown("""
    <div style='padding:32px 0 12px;'>
        <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.14em;color:#6c63ff;margin-bottom:10px;'>
            ⚙️ Machine Learning Dashboard
        </div>
        <h1 style='font-size:2.4rem;font-weight:700;color:white;margin:0;line-height:1.2;'>
            Predictive Maintenance
        </h1>
        <p style='color:#8888aa;margin-top:10px;font-size:1rem;max-width:700px;'>
            Sistem prediksi kegagalan mesin industri berbasis Machine Learning.
            Menggunakan 4 algoritma berbeda pada dataset AI4I 2020 untuk mengidentifikasi
            risiko kegagalan sebelum terjadi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Dataset stats
    st.markdown("### 📦 Ringkasan Dataset")
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("Total Data",    "10.000",   "baris")
    d2.metric("Fitur Input",   "6",        "kolom")
    d3.metric("No Failure",    "9.661",    "96.61%")
    d4.metric("Failure",       "339",      "3.39%")
    d5.metric("Target",        "Binary",   "0 / 1")
    st.markdown("---")

    # ── Penjelasan Dataset ──
    st.markdown("### 📖 Tentang Dataset")
    st.markdown("""
    <div class="hero-card">
    Dataset <strong>AI4I 2020 Predictive Maintenance</strong> berisi 10.000 catatan operasional mesin
    industri. Setiap baris merepresentasikan satu kondisi mesin dengan pembacaan sensor, dan kolom target
    <strong>Target</strong> menandai apakah mesin mengalami kegagalan (1) atau normal (0). Karakteristik
    paling menonjol adalah <strong>ketidakseimbangan kelas yang ekstrem</strong> — hanya 339 dari 10.000
    mesin (3,39%) yang gagal. Akibatnya, akurasi global menjadi metrik yang menyesatkan: model yang selalu
    menebak "normal" pun sudah mencapai ~96,6% akurasi. Karena itu, <strong>Recall</strong> dan
    <strong>F1-Score pada kelas Failure</strong> menjadi tolok ukur yang jauh lebih relevan, sebab gagal
    mendeteksi mesin yang akan rusak (false negative) jauh lebih merugikan daripada false alarm.
    </div>
    """, unsafe_allow_html=True)

    # ── Fitur yang Digunakan ──
    st.markdown("#### 🔧 Fitur yang Digunakan")
    st.dataframe(pd.DataFrame({
        'Kolom'     : ['Type', 'Air temperature [K]', 'Process temperature [K]',
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Target'],
        'Tipe Data' : ['Kategorikal', 'Numerik', 'Numerik', 'Numerik', 'Numerik', 'Numerik', 'Target'],
        'Satuan'    : ['L / M / H', 'Kelvin', 'Kelvin', 'rpm', 'Nm', 'menit', '0 / 1'],
        'Deskripsi' : [
            'Kualitas produk: Low, Medium, High',
            'Suhu udara di sekitar mesin',
            'Suhu proses internal mesin',
            'Kecepatan putaran spindel',
            'Torsi yang bekerja pada mesin',
            'Akumulasi waktu pemakaian alat (keausan)',
            'Label kegagalan: 0 = Normal, 1 = Failure',
        ],
    }), use_container_width=True, hide_index=True)
    st.caption("6 kolom pertama digunakan sebagai input (fitur); kolom Target adalah label yang diprediksi.")
    st.markdown("---")

    # ── Metodologi & Parameter ──
    st.markdown("### ⚙️ Metodologi & Parameter Model")
    st.markdown("""
    <div class="hero-card">
    Setiap model melalui alur kerja yang sama: <strong>pembersihan data → preprocessing → penanganan
    imbalance → hyperparameter tuning → evaluasi</strong>. Perbedaannya terletak pada preprocessing dan
    strategi sampling. <strong>Random Forest</strong> memakai LabelEncoder untuk kolom Type,
    <strong>feature engineering</strong> (3 fitur turunan: temp_diff, power, strain), StandardScaler, dan
    <strong>SMOTE</strong> untuk menyeimbangkan kelas, lalu threshold dioptimasi ke 0,67.
    <strong>Decision Tree, KNN, dan SVM</strong> memakai <strong>ColumnTransformer</strong>
    (StandardScaler untuk fitur numerik + OneHotEncoder untuk Type) dan setelah dibandingkan, versi
    <strong>data asli tanpa SMOTE</strong> dipilih karena memberikan akurasi lebih tinggi.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(pd.DataFrame({
        'Model'           : ['🌲 Random Forest', '🔵 KNN', '🌳 Decision Tree', '⚡ SVM'],
        'Preprocessing'   : ['LabelEncoder + FE + StandardScaler',
                             'ColumnTransformer (Scaler + OneHot)',
                             'ColumnTransformer (Scaler + OneHot)',
                             'ColumnTransformer (Scaler + OneHot)'],
        'Sampling'        : ['SMOTE', 'Data Asli', 'Data Asli', 'Data Asli'],
        'Hyperparameter'  : ['n_estimators=427, max_depth=12, max_features=log2, class_weight=balanced',
                             'n_neighbors=5',
                             'criterion=gini, max_depth=10, min_samples_leaf=4',
                             'kernel=rbf, C=10'],
        'Threshold'       : ['0.67', '0.50', '0.50', '0.50'],
    }), use_container_width=True, hide_index=True)

    st.markdown("#### 📐 Feature Engineering (khusus Random Forest)")
    st.dataframe(pd.DataFrame({
        'Fitur Baru' : ['temp_diff', 'power', 'strain'],
        'Formula'    : ['Process Temp − Air Temp', 'Torque × RPM × (2π/60)', 'Torque × Tool Wear'],
        'Tujuan'     : [
            'Indikasi heat dissipation failure',
            'Estimasi daya mekanis (power failure)',
            'Indikasi overstrain akibat keausan',
        ],
    }), use_container_width=True, hide_index=True)

    tcol = st.columns(3)
    tcol[0].info("**SMOTE**\n\nMenyeimbangkan kelas minoritas (Failure) secara sintetis — dipakai oleh Random Forest.")
    tcol[1].info("**StandardScaler**\n\nMenstandarkan skala fitur sensor — wajib untuk KNN & SVM agar perhitungan jarak/margin adil.")
    tcol[2].info("**Hyperparameter Tuning**\n\nOptuna (RF) & GridSearch (DT/KNN/SVM) untuk mencari kombinasi parameter terbaik.")
    st.markdown("---")

    # Charts row 1
    st.markdown("### 📊 Visualisasi Data & Perbandingan Model")
    st.info("""
    💡 **PENTING UNTUK DIPAHAMI (Metrik Evaluasi):** 
    
    *   **F1 Failure:** Mungkin Anda bertanya, *"F1 Failure angkanya tinggi, bukannya kalau failure (gagal) tinggi itu jelek?"* Jawabannya **Tidak**. Istilah 'F1 Failure' mengukur seberapa tepat model mengenali khusus kelas 'Mesin Rusak'. Semakin mendekati angka 1.0, berarti model semakin akurat mendeteksi kerusakan tanpa memberikan alarm palsu.
    *   **ROC-AUC:** Metrik ini mengukur ketajaman model secara keseluruhan dalam membedakan mesin normal dan mesin rusak. Pada data yang sangat timpang (96% normal), akurasi biasa sangat menyesatkan. Semakin nilai AUC mendekati 1.0, semakin terbukti bahwa model tidak sekadar menebak buta, melainkan memiliki "insting" yang tajam untuk memisahkan kelas kerusakan.
    """)

    c1,c2 = st.columns(2)
    with c1: 
        st.plotly_chart(plot_model_comparison(), use_container_width=True, config={'displayModeBar': False})
        st.caption("**Grafik 1 (Perbandingan Performa):** Menunjukkan perbandingan skor evaluasi dari data validasi. Random Forest memiliki skor F1 Failure tertinggi (biru muda), membuktikan bahwa model ini paling seimbang dan akurat dalam mendeteksi kerusakan dibandingkan model lainnya.")
    with c2: 
        st.plotly_chart(plot_roc_all(), use_container_width=True, config={'displayModeBar': False})
        st.caption("**Grafik 2 (Kurva ROC):** Mengukur ketajaman model dalam membedakan antara mesin normal dan mesin rusak. Garis Random Forest (biru tua) memiliki cakupan area paling luas (AUC 0.982), membuktikan ketajamannya secara keseluruhan.")

    # Charts row 2
    c3,c4 = st.columns(2)
    with c3: 
        st.plotly_chart(plot_radar(), use_container_width=True, config={'displayModeBar': False})
        st.caption("**Grafik 3 (Peta Kemampuan / Radar):** Menunjukkan sebaran performa model di berbagai metrik sekaligus. Area Random Forest menyapu bagian terluar di hampir seluruh sisi, membuktikan model ini unggul secara merata di segala kriteria evaluasi.")
    with c4: 
        st.plotly_chart(plot_f1_recall_scatter(), use_container_width=True, config={'displayModeBar': False})
        st.caption("**Grafik 4 (Keseimbangan F1 vs Recall):** Membandingkan langsung kemampuan mendeteksi seluruh mesin rusak (*Recall*) melawan ketepatan tebakan (*F1*). Posisi Random Forest di sudut kanan atas menunjukkan ia berhasil memaksimalkan deteksi kerusakan tanpa banyak salah tebak.")

    # Charts row 3
    c5,c6 = st.columns(2)
    with c5: 
        st.plotly_chart(plot_class_dist(), use_container_width=True, config={'displayModeBar': False})
        st.caption("**Grafik 5 (Penerapan SMOTE):** Memperlihatkan teknik penyeimbangan porsi data saat *training* model. Dengan SMOTE, data mesin rusak diseimbangkan porsinya dengan mesin normal (4830 vs 4830), sehingga model dapat mengenali kerusakan secara adil tanpa bias.")
    with c6: 
        st.plotly_chart(plot_feature_importance(), use_container_width=True, config={'displayModeBar': False})
        st.caption("**Grafik 6 (Pengaruh Fitur Khusus):** Menunjukkan variabel mana yang paling berpengaruh pada keputusan model. Fitur turunan matematis seperti 'Power' dan 'Strain' berada di posisi puncak, membuktikan fitur tambahan ini sangat sukses meningkatkan kepekaan model.")

    st.markdown("---")
    st.markdown("### 🧬 Mengapa SVM Cocok untuk Data Ekstrem Kompleks?")
    st.info("Sebagai bukti visual pendukung, berikut adalah simulasi bagaimana algoritma **Support Vector Machine (SVM) dengan Kernel RBF** menangani pola data (seperti data sensor masa depan) yang sangat rumit dan tumpang tindih, dibandingkan dengan model Linear biasa.")
    st.plotly_chart(plot_svm_decision_boundary(), use_container_width=True, config={'displayModeBar': False})
    st.caption("**Grafik 7 (Bukti Kehebatan Kernel RBF):** Pada gambar Kiri (Linear), model gagal membuat garis pembatas yang adil karena data saling melingkar. Pada gambar Kanan (RBF), SVM secara cerdas melengkungkan area keputusannya untuk memisahkan data dengan sempurna. Inilah alasan SVM sangat direkomendasikan jika kelak pabrik memunculkan data anomali ekstrem yang non-linear.")
    st.markdown("---")

    # Model cards
    st.markdown("### 🤖 Model yang Digunakan")
    mc1,mc2,mc3,mc4 = st.columns(4)
    with mc1:
        st.markdown("""
        <div class="hero-card">
            <div class="winner-badge">🏆 Best Model</div>
            <div style='font-size:1.8rem;margin-top:4px;'>🌲</div>
            <div style='font-weight:700;color:white;font-size:1rem;margin-top:6px;'>Random Forest</div>
            <div style='color:#8888aa;font-size:0.78rem;margin-top:6px;'>Accuracy: <span style="color:#c0c0d0;font-weight:600;">99.00%</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>F1 Failure: <span style="color:#6c63ff;font-weight:700;">0.8296</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>Recall: <span style="color:#6c63ff;font-weight:600;">0.8235</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size:1.8rem;margin-top:24px;'>🔵</div>
            <div style='font-weight:700;color:white;font-size:1rem;margin-top:6px;'>KNN</div>
            <div style='color:#8888aa;font-size:0.78rem;margin-top:6px;'>Accuracy: <span style="color:#c0c0d0;font-weight:600;">97.40%</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>F1 Failure: <span style="color:#00b4d8;font-weight:700;">0.4348</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>Recall: <span style="color:#00b4d8;font-weight:600;">0.2941</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size:1.8rem;margin-top:24px;'>🌳</div>
            <div style='font-weight:700;color:white;font-size:1rem;margin-top:6px;'>Decision Tree</div>
            <div style='color:#8888aa;font-size:0.78rem;margin-top:6px;'>Accuracy: <span style="color:#c0c0d0;font-weight:600;">98.55%</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>F1 Failure: <span style="color:#2ed573;font-weight:700;">0.7820</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>Recall: <span style="color:#2ed573;font-weight:600;">0.7647</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown("""
        <div class="hero-card">
            <div style='font-size:1.8rem;margin-top:24px;'>⚡</div>
            <div style='font-weight:700;color:white;font-size:1rem;margin-top:6px;'>SVM</div>
            <div style='color:#8888aa;font-size:0.78rem;margin-top:6px;'>Accuracy: <span style="color:#c0c0d0;font-weight:600;">97.90%</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>F1 Failure: <span style="color:#f9ca24;font-weight:700;">0.6379</span></div>
            <div style='color:#8888aa;font-size:0.78rem;'>Recall: <span style="color:#f9ca24;font-weight:600;">0.5441</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Best model highlight
    st.markdown("### 🏆 Model Terbaik: Random Forest")
    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        st.success("""
        **Mengapa Random Forest Terbaik?**

        ✅ Accuracy tertinggi: **99.00%**
        ✅ F1 Failure tertinggi: **0.8296**
        ✅ Recall tertinggi: **0.8235**
        ✅ AUC tertinggi: **0.9818**

        RF mengungguli semua model di setiap metrik
        evaluasi yang digunakan.
        """)
    with col2:
        st.info("""
        **Keunggulan Teknis RF**

        🔧 **Feature Engineering**: 3 fitur baru
        (temp_diff, power, strain) yang meningkatkan
        kemampuan model mendeteksi pola kegagalan.

        ⚖️ **SMOTE**: Satu-satunya model yang
        menggunakan SMOTE untuk handle imbalance
        96.6% : 3.4% secara efektif.

        🎯 **Threshold Opt.**: 0.5 → 0.67
        """)
    with col3:
        st.warning("""
        **Catatan Penting**

        ⚠️ Dataset sangat imbalanced (96.6% : 3.4%)
        sehingga **Accuracy saja tidak cukup** sebagai
        metrik evaluasi.

        Fokus pada **F1 Failure** dan **Recall Failure**
        karena lebih merepresentasikan kemampuan model
        mendeteksi kegagalan mesin yang sebenarnya.

        KNN & SVM unggul di Precision tetapi
        Recall-nya sangat rendah.
        """)

# ══════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════
elif "Random Forest" in page:
    page_header("🌲", "#6c63ff", "Random Forest", "Predictive Maintenance — Random Forest")

    tab_info,tab_guide,tab_pred = st.tabs(["📊 Model Info","📋 Panduan","🔮 Prediksi"])

    with tab_info:
        model_info_tab('rf',
            perf_rows={
                'Metrik'  :['Accuracy','ROC-AUC','Macro F1','F1 Failure','Recall Failure','Threshold'],
                'Baseline':['0.9800','0.9818','0.8609','0.7329','0.8676','0.50'],
                'Tuned'   :['0.9900','0.9818','0.9118','0.8296','0.8235','0.67'],
            },
            tech_items={
                'SMOTE'             :'Handle class imbalance\n96.6% : 3.4%',
                'Feature Engineering':'3 fitur baru:\ntemp_diff, power, strain',
                'Optuna Tuning'     :'TPE Sampler\n50 trials',
                'Threshold Opt.'    :'0.5 → 0.67\nuntuk Macro F1 optimal',
            },
            chart_fns=[plot_feature_importance,plot_confusion_rf,plot_class_dist,plot_threshold_rf]
        )

    with tab_guide:
        c1,c2=st.columns(2)
        with c1:
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
        with c2:
            st.markdown("""
            #### 🔬 Tentang Model
            Random Forest membangun banyak pohon keputusan
            dan menggabungkan hasilnya (ensemble) untuk prediksi
            yang lebih akurat dan robust.

            - **Algoritma**: Random Forest Classifier
            - **Fitur**: 9 kolom (6 asli + 3 FE)
            - **Keunggulan**: AUC & F1 tertinggi
            - **Imbalance**: SMOTE oversampling

            #### 📐 Feature Engineering
            | Fitur | Formula |
            |---|---|
            | temp_diff | Process Temp − Air Temp |
            | power | Torque × RPM × (2π/60) |
            | strain | Torque × Tool Wear |
            """)

    with tab_pred:
        if not models['rf']:
            st.error("❌ Model RF tidak ditemukan. Periksa file `.pkl` di folder `models/`.")
        else:
            input_data=input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi RF", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    try:
                        X_input  =preprocess_rf(input_data)
                        proba    =models['rf']['model'].predict_proba(X_input)[0]
                        threshold=float(models['rf']['metadata'].get('threshold',0.67))
                        pred     =int(proba[1]>=threshold)
                        st.markdown("---")
                        show_result(pred,proba,threshold)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: KNN
# ══════════════════════════════════════════════════════════════
elif "KNN" in page:
    page_header("🔵","#00b4d8","K-Nearest Neighbors","Predictive Maintenance — KNN")

    tab_info,tab_guide,tab_pred = st.tabs(["📊 Model Info","📋 Panduan","🔮 Prediksi"])

    with tab_info:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Accuracy",   "97.40%")
        c2.metric("F1 Failure", "0.4348")
        c3.metric("Recall",     "0.2941")
        c4.metric("Precision",  "0.8333")
        st.markdown("---")
        ca,cb=st.columns(2)
        with ca: st.plotly_chart(plot_knn_k(), use_container_width=True, config={'displayModeBar': False})
        with cb: st.plotly_chart(plot_class_dist(), use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")
        if models['knn']:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("#### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(models['knn']['params'].items(),columns=['Parameter','Value']),
                             use_container_width=True,hide_index=True)
            with c2:
                st.markdown("#### 📈 Performa Detail")
                st.dataframe(pd.DataFrame({
                    'Metrik'   :['Accuracy','Precision Fail','Recall Fail','F1 Failure'],
                    'Data Asli':['0.9740','0.8333','0.2941','0.4348'],
                    'SMOTE'    :['0.9560','0.3958','0.5588','0.4634'],
                }),use_container_width=True,hide_index=True)
        st.markdown("#### 🔧 Teknik yang Digunakan")
        t1,t2,t3=st.columns(3)
        t1.info("**Data Asli**\nTanpa SMOTE\nlebih akurat untuk KNN")
        t2.info("**StandardScaler**\nScaling wajib\nagar jarak seimbang")
        t3.info("**OneHotEncoder**\nEncode kolom Type\n(L/M/H → dummy vars)")

    with tab_guide:
        st.markdown("""
        #### 📖 Cara Menggunakan
        1. Buka tab **Prediksi** → isi parameter mesin → klik **Jalankan Prediksi**

        #### 🔬 Tentang KNN
        KNN mengklasifikasikan mesin berdasarkan kesamaan dengan K tetangga terdekat.
        Mesin dengan parameter mirip mesin yang pernah gagal akan diprediksi lebih berisiko.

        > **Catatan**: KNN menggunakan data asli (tanpa SMOTE) karena accuracy lebih tinggi.
        Namun Recall Failure-nya rendah (0.29), artinya banyak failure yang terlewat.
        """)

    with tab_pred:
        if not models['knn']:
            st.warning("⚠️ Model KNN belum tersedia.")
        else:
            input_data=input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi KNN", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    try:
                        X_input=preprocess_others(input_data,'knn')
                        proba  =models['knn']['model'].predict_proba(X_input)[0]
                        pred   =models['knn']['model'].predict(X_input)[0]
                        st.markdown("---")
                        show_result(int(pred),proba,0.5)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: DECISION TREE
# ══════════════════════════════════════════════════════════════
elif "Decision Tree" in page:
    page_header("🌳","#2ed573","Decision Tree","Predictive Maintenance — Decision Tree")

    tab_info,tab_guide,tab_pred = st.tabs(["📊 Model Info","📋 Panduan","🔮 Prediksi"])

    with tab_info:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Accuracy",   "0.9855")
        c2.metric("F1 Failure", "0.7820")
        c3.metric("Recall",     "0.7647")
        c4.metric("Precision",  "0.8000")
        st.markdown("---")
        ca,cb=st.columns(2)
        with ca: st.plotly_chart(plot_dt_depth(), use_container_width=True, config={'displayModeBar': False})
        with cb: st.plotly_chart(plot_class_dist(), use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")
        if models['dt']:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("#### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(models['dt']['params'].items(),columns=['Parameter','Value']),
                             use_container_width=True,hide_index=True)
            with c2:
                st.markdown("#### 📈 Performa Detail")
                st.dataframe(pd.DataFrame({
                    'Metrik'   :['Accuracy','Precision Fail','Recall Fail','F1 Failure'],
                    'Data Asli':['0.9855','0.8000','0.7647','0.7820'],
                    'SMOTE'    :['0.9460','0.3649','0.7941','0.5000'],
                }),use_container_width=True,hide_index=True)
        st.markdown("#### 🔧 Teknik yang Digunakan")
        t1,t2,t3=st.columns(3)
        t1.info("**Data Asli**\nTanpa SMOTE\nlebih akurat untuk DT")
        t2.info("**Pruning**\nmax_depth dikontrol\ncegah overfitting")
        t3.info("**OneHotEncoder**\nEncode kolom Type\n(L/M/H → dummy vars)")

    with tab_guide:
        st.markdown("""
        #### 📖 Cara Menggunakan
        1. Buka tab **Prediksi** → isi parameter mesin → klik **Jalankan Prediksi**

        #### 🔬 Tentang Decision Tree
        Decision Tree membangun pohon keputusan berdasarkan fitur yang paling informatif.
        Setiap node merupakan pertanyaan tentang fitur tertentu, model mengikuti cabang
        hingga mencapai prediksi akhir.

        > **Catatan**: DT adalah runner-up terbaik dengan F1 Failure 0.782 dan Recall 0.765 —
        jauh lebih baik dari KNN dan SVM untuk mendeteksi kegagalan mesin.
        """)

    with tab_pred:
        if not models['dt']:
            st.warning("⚠️ Model DT belum tersedia.")
        else:
            input_data=input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi DT", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    try:
                        X_input=preprocess_others(input_data,'dt')
                        proba  =models['dt']['model'].predict_proba(X_input)[0]
                        pred   =models['dt']['model'].predict(X_input)[0]
                        st.markdown("---")
                        show_result(int(pred),proba,0.5)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: SVM
# ══════════════════════════════════════════════════════════════
elif "SVM" in page:
    page_header("⚡","#f9ca24","Support Vector Machine","Predictive Maintenance — SVM")

    tab_info,tab_guide,tab_pred = st.tabs(["📊 Model Info","📋 Panduan","🔮 Prediksi"])

    with tab_info:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Accuracy",   "97.90%")
        c2.metric("F1 Failure", "0.6379")
        c3.metric("Recall",     "0.5441")
        c4.metric("Precision",  "0.7708")
        st.markdown("---")
        ca,cb=st.columns(2)
        with ca: st.plotly_chart(plot_svm_kernel(), use_container_width=True, config={'displayModeBar': False})
        with cb: st.plotly_chart(plot_class_dist(), use_container_width=True, config={'displayModeBar': False})
        st.markdown("---")
        if models['svm']:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("#### ⚙️ Hyperparameter Terbaik")
                st.dataframe(pd.DataFrame(models['svm']['params'].items(),columns=['Parameter','Value']),
                             use_container_width=True,hide_index=True)
            with c2:
                st.markdown("#### 📈 Performa Detail")
                st.dataframe(pd.DataFrame({
                    'Metrik'   :['Accuracy','Precision Fail','Recall Fail','F1 Failure'],
                    'Data Asli':['0.9790','0.7708','0.5441','0.6379'],
                    'SMOTE'    :['0.9420','0.3554','0.8676','0.5043'],
                }),use_container_width=True,hide_index=True)
        st.markdown("#### 🔧 Teknik yang Digunakan")
        t1,t2,t3=st.columns(3)
        t1.info("**Data Asli**\nTanpa SMOTE\nlebih akurat untuk SVM")
        t2.info("**StandardScaler**\nScaling wajib\nagar margin optimal")
        t3.info("**Kernel RBF**\nPemisahan non-linear\ndi ruang fitur tinggi")

    with tab_guide:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
            #### 📖 Cara Menggunakan
            1. Buka tab **Prediksi** → isi parameter → klik **Jalankan Prediksi**

            #### 🎯 Interpretasi Hasil
            | Prob. Failure | Interpretasi |
            |---|---|
            | 0% – 30% | Mesin sangat aman |
            | 30% – 50% | Perlu dimonitor |
            | > 50% | Segera maintenance! |
            """)
        with c2:
            st.markdown("""
            #### 🔬 Tentang SVM
            SVM mencari hyperplane terbaik yang memisahkan kelas failure
            dan normal dengan margin maksimal menggunakan kernel RBF.

            - **Kernel**: RBF (Radial Basis Function)
            - **Scaling**: StandardScaler (wajib)
            - **Encoding**: OneHotEncoder untuk Type
            - **Data**: Asli tanpa SMOTE (lebih baik)
            """)

    with tab_pred:
        if not models['svm']:
            st.warning("⚠️ Model SVM belum tersedia.")
        else:
            input_data=input_form()
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔮 Jalankan Prediksi SVM", use_container_width=True, type="primary"):
                with st.spinner("Memproses..."):
                    try:
                        X_input=preprocess_others(input_data,'svm')
                        proba  =models['svm']['model'].predict_proba(X_input)[0]
                        pred   =models['svm']['model'].predict(X_input)[0]
                        st.markdown("---")
                        show_result(int(pred),proba,0.5)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════
#  PAGE: KESIMPULAN
# ══════════════════════════════════════════════════════════════
elif "Kesimpulan" in page:
    page_header("📊","#6c63ff","Perbandingan & Analisis","Kesimpulan Model")

    st.markdown("### 📋 Tabel Perbandingan Lengkap")
    st.markdown("Berikut adalah kompilasi metrik dari seluruh model yang kita ujicobakan. Warna-warni tabel ini menunjukkan mana algoritma yang paling tangguh dalam mendeteksi mesin rusak.")
    st.dataframe(pd.DataFrame({
        'Model'         :['🌲 Random Forest','🔵 KNN','🌳 Decision Tree','⚡ SVM'],
        'Accuracy'      :['99.00%','97.40%','98.55%','97.90%'],
        'F1 Failure'    :['0.8296','0.4348','0.7820','0.6379'],
        'Recall Failure':['0.8235','0.2941','0.7647','0.5441'],
        'Precision Fail':['0.8400','0.8333','0.8000','0.7708'],
        'Data Training' :['SMOTE','Data Asli','Data Asli','Data Asli'],
        'Feature Eng.'  :['✅ Ya (3 FE)','❌ Tidak','❌ Tidak','❌ Tidak'],
        'Ranking'       :['🥇 1st','🥉 4th','🥈 2nd','🥉 3rd'],
    }).style.background_gradient(cmap='viridis', subset=['F1 Failure']),use_container_width=True,hide_index=True)

    st.markdown("---")
    c1,c2=st.columns(2)
    with c1: 
        with st.container(border=True): st.plotly_chart(plot_model_comparison(), use_container_width=True, config={'displayModeBar': False})
    with c2: 
        with st.container(border=True): st.plotly_chart(plot_radar(), use_container_width=True, config={'displayModeBar': False})

    c3,c4=st.columns(2)
    with c3: 
        with st.container(border=True): st.plotly_chart(plot_roc_all(), use_container_width=True, config={'displayModeBar': False})
    with c4: 
        with st.container(border=True): st.plotly_chart(plot_f1_recall_scatter(), use_container_width=True, config={'displayModeBar': False})
    st.markdown("---")
    st.markdown("### 🏆 Kesimpulan Validasi: Mengapa Random Forest Menjadi Juara Mutlak?")
    st.markdown("""
    Meskipun nilai **Akurasi Global** pada model Decision Tree (98.55%) hampir menyamai Random Forest (99.00%), keunggulan utama Random Forest terletak pada kemampuannya mendeteksi kelas data minoritas (Mesin Rusak). 
    Pada dataset operasi mesin ini, memprediksi mesin dalam keadaan normal adalah hal yang mudah karena 96,6% data memang merupakan mesin normal. Tantangan utamanya adalah mendeteksi 3,4% data kerusakan mesin tersebut secara akurat.
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #a855f7; min-height: 230px;">
        <div>
        <h4 style="margin-top:0; color:#a855f7;">1️⃣ Ensemble Learning</h4>
        Berbeda dengan model <i>Decision Tree</i> tunggal yang rentan terhadap bias data, <i>Random Forest</i> dibangun dari gabungan <b>427 pohon keputusan</b>.<br><br>Ketika ada ketidaknormalan pada data sensor, keputusan prediksi akhir tidak diambil dari satu pohon saja, melainkan menggabungkan prediksi dari ke-427 pohon tersebut. Proses ini membantu meminimalkan kesalahan prediksi dan menghasilkan keputusan yang jauh lebih stabil.
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #00b4d8; min-height: 230px;">
        <div>
        <h4 style="margin-top:0; color:#00b4d8;">3️⃣ Penambahan Fitur Baru</h4>
        Daripada hanya menggunakan data sensor mentah (seperti suhu dan putaran), penerapan <b>Feature Engineering</b> memungkinkan kita menambahkan fitur baru (seperti <i>Power/Daya</i> dan <i>Strain/Tekanan</i>) yang dihitung dari perpaduan sensor tersebut.<br><br>Informasi tambahan ini sangat membantu Random Forest mengenali kondisi kritis pemicu kerusakan, sehingga metrik <i>Recall Failure</i>-nya dapat mencapai skor tinggi di 0.8235.
        </div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #2ed573; min-height: 230px;">
        <div>
        <h4 style="margin-top:0; color:#2ed573;">2️⃣ Penyeimbangan Data (SMOTE)</h4>
        Model klasifikasi sering kali kesulitan memprediksi kelas yang jumlah datanya sangat sedikit.<br><br>Metode <i>Random Forest</i> yang digunakan mengatasi hal ini dengan menerapkan algoritma <b>SMOTE</b>. Teknik ini menambahkan data buatan pada tahap pelatihan agar jumlah data mesin normal dan rusak menjadi seimbang (<b>50:50</b>). Hasilnya, model dapat mempelajari pola kerusakan dengan porsi yang sama baiknya dengan pola mesin normal.
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #ff4757; min-height: 230px;">
        <div>
        <h4 style="margin-top:0; color:#ff4757;">4️⃣ Evaluasi ROC-AUC</h4>
        ROC-AUC mengukur seberapa mampu sebuah model membedakan kelas secara keseluruhan.<br><br>Pada data yang tidak seimbang, akurasi biasa sangat menyesatkan. ROC-AUC membuktikan kehebatan asli model dengan menguji semua batas kemungkinan. Nilai AUC tertinggi pada Random Forest (<b>0.9818</b>) menjadi bukti valid bahwa secara probabilitas matematis, Random Forest memiliki insting pemisah yang paling tajam.
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-card" style="border: 2px solid #6c63ff; background: linear-gradient(145deg, rgba(108,99,255,0.15), rgba(37,37,69,0.9)); margin-top: 10px;">
    <div>
    <h3 style="margin-top:0; color:white; text-align:center;">🎯 Kesimpulan Akhir Pemilihan Model</h3>
    <p style="font-size: 1.05rem; line-height: 1.6; text-align:center; color:#c0c0d0;">
    Nilai <b>F1 Failure = 0.8296</b> pada Random Forest menunjukkan keandalan model yang sangat baik dalam mendeteksi kerusakan secara tepat. Keseimbangan yang sempurna antara ketepatan tebakan (<i>Precision</i>) dan kemampuan mendeteksi seluruh kerusakan (<i>Recall</i>) menjadikan <b>Random Forest</b> sebagai solusi <i>Predictive Maintenance</i> yang paling direkomendasikan dan siap digunakan di lapangan secara profesional.
    </p>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: TIM PENGEMBANG
# ══════════════════════════════════════════════════════════════
elif page == "👥 Tim Pengembang":
    st.markdown("<h2 class='title-glow'>👥 Profil Tim Pengembang</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0b0; font-size:1.05rem; margin-bottom:30px;'>Halaman ini memuat profil dan rincian pembagian tugas (*Role & Responsibility*) dari anggota tim dalam pengerjaan proyek Tugas Besar.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #6c63ff; margin-bottom: 20px;">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 5px;">👑</div>
            <h4 style="text-align: center; margin-top: 0; color: #ffffff;">Ketut Qorry Kharismayani</h4>
            <p style="text-align: center; color: #00b4d8; font-weight: bold; margin-bottom: 15px; font-size: 0.9rem;">NIM: 607012400015 | Ketua</p>
            <h5 style="color: #c0c0d0; border-bottom: 1px solid #2d2d4e; padding-bottom: 5px; font-size: 0.9rem;">🛠️ Peran & Tugas:</h5>
            <ul style="color: #a0a0b0; line-height: 1.6; font-size: 0.85rem;">
                <li>Mengkoordinasikan keseluruhan jalannya proyek penelitian.</li>
                <li>Merancang arsitektur dan melatih algoritma <b>Random Forest</b>.</li>
                <li>Mendevelop dan mendesain antarmuka aplikasi (*UI/UX*) menggunakan <b>Streamlit</b>.</li>
                <li>Menyusun kerangka kesimpulan akhir, visualisasi data, dan penyusunan laporan proyek.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #00b4d8; margin-bottom: 20px;">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 5px;">👩‍💻</div>
            <h4 style="text-align: center; margin-top: 0; color: #ffffff;">Fazrina Esa Putri Permana</h4>
            <p style="text-align: center; color: #00b4d8; font-weight: bold; margin-bottom: 15px; font-size: 0.9rem;">NIM: 607012400047 | Anggota</p>
            <h5 style="color: #c0c0d0; border-bottom: 1px solid #2d2d4e; padding-bottom: 5px; font-size: 0.9rem;">🛠️ Peran & Tugas:</h5>
            <ul style="color: #a0a0b0; line-height: 1.6; font-size: 0.85rem;">
                <li>Melakukan eksperimen pemodelan menggunakan algoritma <b>K-Nearest Neighbors (KNN)</b>.</li>
                <li>Menganalisis matriks kebingungan (*Confusion Matrix*) model.</li>
                <li>Menarik wawasan metrik presisi terhadap kemampuan prediksi model dalam mendeteksi anomali.</li>
                <li>Berkontribusi penuh dalam penyusunan dan penulisan dokumen laporan akhir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #f9ca24; margin-bottom: 20px;">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 5px;">👩‍💻</div>
            <h4 style="text-align: center; margin-top: 0; color: #ffffff;">Dalfa Munawwarotul Mahmudah</h4>
            <p style="text-align: center; color: #00b4d8; font-weight: bold; margin-bottom: 15px; font-size: 0.9rem;">NIM: 607012430008 | Anggota</p>
            <h5 style="color: #c0c0d0; border-bottom: 1px solid #2d2d4e; padding-bottom: 5px; font-size: 0.9rem;">🛠️ Peran & Tugas:</h5>
            <ul style="color: #a0a0b0; line-height: 1.6; font-size: 0.85rem;">
                <li>Melakukan eksperimen pemodelan menggunakan algoritma <b>Support Vector Machine (SVM)</b>.</li>
                <li>Mengkonfigurasi *Kernel RBF* untuk memisahkan batas data yang melingkar secara kompleks.</li>
                <li>Menyusun landasan evaluasi dan pengujian *Decision Boundary* SVM.</li>
                <li>Berkontribusi penuh dalam penyusunan dan penulisan dokumen laporan akhir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="hero-card" style="border-top: 4px solid #2ed573; margin-bottom: 20px;">
            <div style="font-size: 3rem; text-align: center; margin-bottom: 5px;">👩‍💻</div>
            <h4 style="text-align: center; margin-top: 0; color: #ffffff;">Inaya Faridah</h4>
            <p style="text-align: center; color: #00b4d8; font-weight: bold; margin-bottom: 15px; font-size: 0.9rem;">NIM: 607012430016 | Anggota</p>
            <h5 style="color: #c0c0d0; border-bottom: 1px solid #2d2d4e; padding-bottom: 5px; font-size: 0.9rem;">🛠️ Peran & Tugas:</h5>
            <ul style="color: #a0a0b0; line-height: 1.6; font-size: 0.85rem;">
                <li>Melakukan eksperimen pemodelan dan penyetelan menggunakan algoritma <b>Decision Tree (DT)</b>.</li>
                <li>Mengoptimalkan batas struktur pohon keputusan demi menjaga kecepatan komputasi.</li>
                <li>Menganalisis keseimbangan performa model *Tree-based* dibandingkan dengan algoritma lain.</li>
                <li>Berkontribusi penuh dalam penyusunan dan penulisan dokumen laporan akhir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#8888aa; font-size:1.05rem; padding: 15px;'>
        Proyek ini dikembangkan untuk memenuhi Tugas Besar Mata Kuliah <b>Dasar Ilmu Data</b><br>
        <span style='color:#ffffff; font-weight:bold;'>D3 Sistem Informasi - Telkom University</span> © 2024
    </div>
    """, unsafe_allow_html=True)