"""
app.py - Neural Fraud Sentinel — Modern Enterprise Anomaly Engine
=======================================================================
Run with:  streamlit run app.py
"""

import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

warnings.filterwarnings("ignore")

pd.set_option("styler.render.max_elements", 20_000_000)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEURAL FRAUD SENTINEL // Enterprise Anomaly Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark Cyber"
if "font_size" not in st.session_state:
    st.session_state["font_size"] = "Normal"
if "cookies_enabled" not in st.session_state:
    st.session_state["cookies_enabled"] = True
if "history" not in st.session_state:
    st.session_state["history"] = []

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & MODEL PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "isolation_forest.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join("model", "feature_names.pkl")

DEFAULT_FEATURES = [
    "amt", "lat", "long", "city_pop", "merch_lat",
    "merch_long", "hour", "day_of_week", "age", "amt_log",
]

# ─────────────────────────────────────────────────────────────────────────────
# THEME ENGINE & CONTRAST PRESETS
# ─────────────────────────────────────────────────────────────────────────────
THEME_CONFIGS = {
    "Dark Cyber": {
        "bg_color": "#04060c",
        "card_bg": "#0a0e1a",
        "accent_primary": "#00F0FF",
        "accent_secondary": "#FF007A",
        "accent_green": "#00FF9D",
        "accent_red": "#FF2A55",
        "accent_amber": "#FFB800",
        "text_color": "#e2e8f0",
        "subtext_color": "#94a3b8",
        "sidebar_bg": "#060812",
        "border_color": "#1e293b",
        "input_bg": "#0f172a",
        "plotly_template": "plotly_dark"
    },
    "Red Crimson": {
        "bg_color": "#0d0205",
        "card_bg": "#1a050b",
        "accent_primary": "#FF2A55",
        "accent_secondary": "#FFB800",
        "accent_green": "#00FF9D",
        "accent_red": "#FF0055",
        "accent_amber": "#FFAA00",
        "text_color": "#fecdd3",
        "subtext_color": "#fda4af",
        "sidebar_bg": "#140308",
        "border_color": "#4c0519",
        "input_bg": "#280710",
        "plotly_template": "plotly_dark"
    },
    "Light Neon": {
        "bg_color": "#f8fafc",
        "card_bg": "#ffffff",
        "accent_primary": "#0284c7",
        "accent_secondary": "#e11d48",
        "accent_green": "#16a34a",
        "accent_red": "#dc2626",
        "accent_amber": "#d97706",
        "text_color": "#0f172a",
        "subtext_color": "#334155",
        "sidebar_bg": "#f1f5f9",
        "border_color": "#cbd5e1",
        "input_bg": "#ffffff",
        "plotly_template": "plotly_white"
    }
}

current_theme = THEME_CONFIGS.get(st.session_state["theme"], THEME_CONFIGS["Dark Cyber"])

FONT_SCALES = {
    "Normal": "1.0rem",
    "Grande": "1.15rem",
    "Extragrande": "1.3rem"
}
selected_font_scale = FONT_SCALES.get(st.session_state["font_size"], "1.0rem")

# ─────────────────────────────────────────────────────────────────────────────
# INTERNATIONALIZATION (I18N - ZERO EMOJIS)
# ─────────────────────────────────────────────────────────────────────────────
LANG_TEXTS = {
    "PT": {
        "title": "NEURAL FRAUD SENTINEL",
        "subtitle": "Detector de Anomalias em Cartões de Crédito por Machine Learning Não Supervisionado",
        "status_ready": "MOTOR ISOLATION FOREST: OPERACIONAL",
        "status_offline": "Modelo Offline. Execute python train.py",
        "arch_overview": "Arquitetura do Sistema",
        "arch_desc": "Alimentado por um algoritmo Isolation Forest treinado em 1,3 milhão de transações reais do Kaggle.",
        "session_telemetry": "Telemetria de Sessão",
        "analyzed": "Analisadas",
        "flagged": "Anomalias",
        "clear_telemetry": "Limpar Registro",
        "tab_simulate": "SIMULAÇÃO AO VIVO",
        "tab_batch": "PROCESSADOR EM LOTE",
        "tab_insights": "INSIGHTS NEURAIS",
        "tab_about": "ARQUITETURA",
        "sim_title": "Simulador de Telemetria em Tempo Real",
        "sim_desc": "Injete parâmetros de transação no motor neural para calcular a probabilidade de anomalia em tempo real.",
        "payload_details": "Parâmetros da Transação",
        "node_coords": "Coordenadas do Titular",
        "merchant_target": "Dados do Estabelecimento",
        "amt": "Valor ($)",
        "hour": "Hora de Execução (0-23)",
        "day": "Dia da Semana",
        "card_lat": "Latitude do Titular",
        "card_long": "Longitude do Titular",
        "age": "Idade do Titular",
        "city_pop": "População da Cidade",
        "merch_lat": "Latitude do Estabelecimento",
        "merch_long": "Longitude do Estabelecimento",
        "btn_run": "EXECUTAR DIAGNÓSTICO NEURAL",
        "verdict_fraud_title": "ANOMALIA DETECTADA // RISCO ELEVADO",
        "verdict_fraud_desc": "O comprimento de isolamento da árvore colapsou. Transação sinalizada como anômala/suspeita.",
        "verdict_safe_title": "TRANSAÇÃO VERIFICADA // COMPORTAMENTO NOMINAL",
        "verdict_safe_desc": "A telemetria da transação alinha-se aos agrupamentos legítimos baseline.",
        "raw_score": "Pontuação Bruta",
        "diagnostics": "Diagnóstico do Sistema Neural",
        "session_log": "Log de Auditoria da Sessão",
        "batch_title": "Processamento Massivo em Lote",
        "batch_desc": "Envie arquivos CSV de qualquer tamanho (incluindo o dataset completo com 1,3 milhão de registros).",
        "upload_label": "Carregar Dataset CSV",
        "btn_download_sample": "Baixar Payload CSV de Exemplo",
        "btn_test_sample": "Testar 30 Registros Sintéticos",
        "total_executed": "Total Avaliado",
        "flagged_anomaly": "Sinalizadas como Anomalia",
        "verified_safe": "Verificadas como Seguras",
        "avg_risk": "Risco Médio Avaliado",
        "chart_pie_title": "Distribuição: Anomalia vs Segura",
        "chart_hist_title": "Espectro da Distribuição de Risco",
        "grid_title": "Tabela de Resultados Avaliados",
        "btn_download_results": "BAIXAR CSV COMPLETO COM PREDIÇÕES",
        "large_batch_notice": "Modo de Alto Volume Ativo ({n:,} registros). Exibindo prévia das primeiras 5.000 linhas.",
        "model_params": "Hiperparâmetros do Modelo",
        "theory_title": "Fundamentos da Isolation Forest",
        "theory_desc": "Ao contrário de classificadores supervisionados, a Isolation Forest isola anomalias particionando o espaço de atributos aleatoriamente.",
        "btn_monte_carlo": "Gerar e Avaliar 50 Transmissões Aleatórias",
        "days": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
        "reason_val": "Pico de Valor: ${val:.2f} excede a linha de base habitual.",
        "reason_time": "Horário Incomum: Registrada às {h}:00 AM.",
        "reason_pop": "Densidade Baixa: População da cidade ({pop:,}) diverge da média.",
        "reason_geo": "Anomalia Geográfica: Distância elevada entre cartão e estabelecimento.",
        "reason_weekend": "Janela de Fim de Semana: Registrada no final de semana.",
        "reason_ok": "Telemetria Nominal: Todos os parâmetros estão em conformidade."
    },
    "EN": {
        "title": "NEURAL FRAUD SENTINEL",
        "subtitle": "Real-Time Unsupervised Machine Learning Credit Card Anomaly Detector",
        "status_ready": "ISOLATION FOREST ENGINE: OPERATIONAL",
        "status_offline": "Model Offline. Run python train.py",
        "arch_overview": "Architecture Overview",
        "arch_desc": "Powered by an unsupervised Isolation Forest algorithm trained on 1.3M real Kaggle transactions.",
        "session_telemetry": "Session Telemetry",
        "analyzed": "Analyzed",
        "flagged": "Anomalies",
        "clear_telemetry": "Clear Audit Log",
        "tab_simulate": "LIVE SIMULATION",
        "tab_batch": "BATCH PROCESSOR",
        "tab_insights": "NEURAL INSIGHTS",
        "tab_about": "ARCHITECTURE",
        "sim_title": "Real-Time Telemetry Simulator",
        "sim_desc": "Inject transaction telemetry into the neural engine to evaluate anomaly probability in real time.",
        "payload_details": "Transaction Telemetry",
        "node_coords": "Cardholder Node Coordinates",
        "merchant_target": "Merchant Target Coordinates",
        "amt": "Amount ($)",
        "hour": "Execution Hour (0-23)",
        "day": "Day of Week",
        "card_lat": "Cardholder Latitude",
        "card_long": "Cardholder Longitude",
        "age": "Cardholder Age",
        "city_pop": "City Population",
        "merch_lat": "Merchant Latitude",
        "merch_long": "Merchant Longitude",
        "btn_run": "RUN NEURAL DIAGNOSTIC",
        "verdict_fraud_title": "ANOMALY DETECTED // HIGH RISK PAYLOAD",
        "verdict_fraud_desc": "Isolation tree path length collapsed. Payload flagged as anomalous/fraudulent.",
        "verdict_safe_title": "TRANSACTION VERIFIED // NOMINAL BEHAVIOR",
        "verdict_safe_desc": "Telemetry aligns with standard legitimate baseline clusters.",
        "raw_score": "Raw Score",
        "diagnostics": "Neural System Diagnostics",
        "session_log": "Session Audit Log",
        "batch_title": "Massive Batch Processing",
        "batch_desc": "Upload CSV files of any scale (including datasets with 1.3M+ records).",
        "upload_label": "Upload CSV Dataset",
        "btn_download_sample": "Download Sample Payload CSV",
        "btn_test_sample": "Test 30 Synthetic Records",
        "total_executed": "Total Executed",
        "flagged_anomaly": "Flagged Anomalies",
        "verified_safe": "Verified Safe",
        "avg_risk": "Avg Risk Metric",
        "chart_pie_title": "Anomaly vs Safe Distribution",
        "chart_hist_title": "Risk Spectrum Distribution",
        "grid_title": "Evaluated Results Grid",
        "btn_download_results": "DOWNLOAD COMPLETE PREDICTIONS CSV",
        "large_batch_notice": "High Volume Mode Active ({n:,} records). Displaying top 5,000 rows.",
        "model_params": "Model Hyperparameters",
        "theory_title": "Isolation Forest Fundamentals",
        "theory_desc": "Unlike supervised classifiers, Isolation Forest isolates anomalies by randomly partitioning feature space.",
        "btn_monte_carlo": "Generate & Score 50 Random Transmissions",
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "reason_val": "Value Spike: ${val:.2f} exceeds standard consumer baseline.",
        "reason_time": "Unusual Time Window: Registered at {h}:00 AM.",
        "reason_pop": "Low Node Density: City population ({pop:,}) diverges from average.",
        "reason_geo": "Geographic Anomaly: High distance delta between cardholder and merchant.",
        "reason_weekend": "Weekend Window: Registered during weekend.",
        "reason_ok": "Nominal Telemetry: All parameters align with standard profile."
    }
}
for lang_code in ["ES", "FR", "DE", "ZH"]:
    if lang_code not in LANG_TEXTS:
        LANG_TEXTS[lang_code] = LANG_TEXTS["EN"]

# ─────────────────────────────────────────────────────────────────────────────
# ZERO-EMOJI DYNAMIC STYLESHEET (PERFECT LIGHT & DARK MODE CONTRAST)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* HIDE STREAMLIT DEFAULT HEADER TOOLBAR */
header[data-testid="stHeader"] {{ visibility: hidden !important; height: 0px !important; }}
div[data-testid="stToolbar"] {{ visibility: hidden !important; display: none !important; }}
footer {{ visibility: hidden !important; }}

/* Root Styling */
html, body, [data-testid="stAppViewContainer"], .stApp {{
    background-color: {current_theme['bg_color']} !important;
    font-family: 'Rajdhani', sans-serif;
    font-size: {selected_font_scale} !important;
    color: {current_theme['text_color']} !important;
}}

/* Light mode high-contrast overrides */
p, span, label, h1, h2, h3, h4, h5, h6, li, td, th, div, .stMarkdown, .stSelectbox, .stNumberInput, .stSlider {{
    color: {current_theme['text_color']} !important;
}}

/* Form Inputs, Selectboxes, Number Inputs */
input, select, textarea, [data-baseweb="input"], [data-baseweb="select"] {{
    color: {current_theme['text_color']} !important;
    background-color: {current_theme['input_bg']} !important;
    border: 1px solid {current_theme['border_color']} !important;
    border-radius: 6px !important;
}}

.main .block-container {{
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 95%;
}}

/* Header Banner */
.cyber-header {{
    background: {current_theme['card_bg']};
    border: 1px solid {current_theme['border_color']};
    border-left: 5px solid {current_theme['accent_primary']};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}}

.cyber-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: calc({selected_font_scale} * 2.0);
    font-weight: 900;
    letter-spacing: 2px;
    color: {current_theme['accent_primary']};
    margin: 0;
    text-transform: uppercase;
}}

.cyber-subtitle {{
    font-family: 'Rajdhani', sans-serif;
    font-size: calc({selected_font_scale} * 1.1);
    color: {current_theme['subtext_color']};
    margin-top: 0.3rem;
}}

/* Badges */
.cyber-tag {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: calc({selected_font_scale} * 0.75);
    font-weight: 600;
    border-radius: 4px;
    text-transform: uppercase;
    margin-right: 0.4rem;
}}

.tag-cyan  {{ background: rgba(2, 132, 199, 0.1); color: {current_theme['accent_primary']}; border: 1px solid {current_theme['border_color']}; }}
.tag-pink  {{ background: rgba(225, 29, 72, 0.1); color: {current_theme['accent_secondary']}; border: 1px solid {current_theme['accent_secondary']}; }}
.tag-green {{ background: rgba(22, 163, 74, 0.1); color: {current_theme['accent_green']}; border: 1px solid {current_theme['accent_green']}; }}

/* Verdict Banners */
.verdict-cyber-safe {{
    background: rgba(22, 163, 74, 0.08);
    border: 2px solid {current_theme['accent_green']};
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
}}
.verdict-cyber-safe h2 {{
    font-family: 'Orbitron', sans-serif;
    color: {current_theme['accent_green']};
    margin: 0;
    font-size: calc({selected_font_scale} * 1.4);
}}

.verdict-cyber-fraud {{
    background: rgba(220, 38, 38, 0.08);
    border: 2px solid {current_theme['accent_red']};
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
}}
.verdict-cyber-fraud h2 {{
    font-family: 'Orbitron', sans-serif;
    color: {current_theme['accent_red']};
    margin: 0;
    font-size: calc({selected_font_scale} * 1.4);
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {current_theme['sidebar_bg']} !important;
    border-right: 1px solid {current_theme['border_color']} !important;
}}

/* Tabs Header */
button[data-baseweb="tab"] {{
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: calc({selected_font_scale} * 0.9) !important;
    letter-spacing: 1.5px !important;
    color: {current_theme['subtext_color']} !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.25rem !important;
}}
button[aria-selected="true"] {{
    color: {current_theme['accent_primary']} !important;
    border-bottom: 2px solid {current_theme['accent_primary']} !important;
    background: rgba(2, 132, 199, 0.06) !important;
    border-radius: 6px 6px 0 0 !important;
}}

/* Expander Header Clean Contour */
.stExpander {{
    border: 1px solid {current_theme['border_color']} !important;
    border-radius: 8px !important;
    background-color: {current_theme['card_bg']} !important;
}}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {current_theme['bg_color']}; }}
::-webkit-scrollbar-thumb {{ background: {current_theme['border_color']}; border-radius: 3px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (Cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Model Weights...")
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, DEFAULT_FEATURES

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = (
        joblib.load(FEATURE_NAMES_PATH)
        if os.path.exists(FEATURE_NAMES_PATH)
        else DEFAULT_FEATURES
    )
    return model, scaler, features

model, scaler, FEATURES = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def predict_transaction(values: dict) -> tuple[str, float, float]:
    row = pd.DataFrame([values])[FEATURES].fillna(0)
    X_scaled = scaler.transform(row)

    raw_pred   = model.predict(X_scaled)[0]
    raw_score  = model.decision_function(X_scaled)[0]
    label      = "FRAUD" if raw_pred == -1 else "SAFE"

    pct_score = float(np.clip(((-raw_score + 0.3) / 0.6) * 100, 0, 100))
    return label, raw_score, pct_score


def score_gauge(risk_pct: float) -> go.Figure:
    color = current_theme['accent_red'] if risk_pct >= 60 else (current_theme['accent_amber'] if risk_pct >= 35 else current_theme['accent_green'])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Orbitron"}},
            title={"text": "ANOMALY RISK MATRIX", "font": {"size": 13, "color": current_theme['accent_primary'], "family": "Rajdhani"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": current_theme['accent_primary'], "tickwidth": 2},
                "bar":  {"color": color, "thickness": 0.85},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": current_theme['border_color'],
                "steps": [
                    {"range": [0,  35], "color": "rgba(22, 163, 74, 0.15)"},
                    {"range": [35, 60], "color": "rgba(217, 119, 6, 0.15)"},
                    {"range": [60, 100], "color": "rgba(220, 38, 38, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": current_theme['text_color'], "width": 3},
                    "thickness": 0.8,
                    "value": 60,
                },
            },
        )
    )
    fig.update_layout(
        height=270,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": current_theme['text_color']},
        margin=dict(t=40, b=10, l=25, r=25),
    )
    return fig


def create_radar_chart(values: dict) -> go.Figure:
    categories = ['Valor (Norm)', 'Hora (Norm)', 'Distância Geo', 'Idade (Norm)', 'Densidade Pop']
    
    dist_geo = min(abs(values.get('lat', 0) - values.get('merch_lat', 0)) * 25, 100)
    norm_amt = min((values.get('amt', 0) / 1000) * 100, 100)
    norm_hour = ((values.get('hour', 12) if values.get('hour', 12) <= 12 else 24 - values.get('hour', 12)) / 12) * 100
    norm_pop = max(100 - (values.get('city_pop', 100000) / 10000), 0)
    norm_age = (values.get('age', 35) / 90) * 100

    val_vector = [norm_amt, norm_hour, dist_geo, norm_age, norm_pop]
    baseline_vector = [20, 30, 15, 40, 20]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=baseline_vector,
        theta=categories,
        fill='toself',
        name='Baseline Segura',
        line_color=current_theme['accent_green'],
        fillcolor='rgba(22, 163, 74, 0.15)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=val_vector,
        theta=categories,
        fill='toself',
        name='Transação Atual',
        line_color=current_theme['accent_primary'],
        fillcolor='rgba(2, 132, 199, 0.25)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=current_theme['subtext_color']),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=current_theme['text_color'], family="Rajdhani"),
        margin=dict(t=30, b=20, l=30, r=30)
    )
    return fig


def explain_prediction(values: dict, label: str, risk_pct: float, t: dict) -> str:
    reasons = []
    if values.get("amt", 0) > 500:
        reasons.append(t["reason_val"].format(val=values['amt']))
    if values.get("hour", 12) in range(0, 5):
        reasons.append(t["reason_time"].format(h=values['hour']))
    if values.get("city_pop", 100_000) < 5_000:
        reasons.append(t["reason_pop"].format(pop=int(values['city_pop'])))
    if abs(values.get("lat", 0) - values.get("merch_lat", 0)) > 2:
        reasons.append(t["reason_geo"])
    if values.get("day_of_week", 1) in [5, 6]:
        reasons.append(t["reason_weekend"])

    if not reasons:
        reasons.append(t["reason_ok"])

    header = f"### {t['diagnostics']}\n\n"
    return header + "\n\n".join(f"- {r}" for r in reasons)


def batch_predict(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"]        = df["trans_date_trans_time"].dt.hour.fillna(12)
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek.fillna(0)

    if "dob" in df.columns and "trans_date_trans_time" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["age"] = (df["trans_date_trans_time"].dt.year - df["dob"].dt.year).fillna(35)
    elif "age" not in df.columns:
        df["age"] = 35

    if "amt" in df.columns:
        df["amt_log"] = np.log1p(df["amt"].fillna(0))

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURES].fillna(0)
    X_scaled = scaler.transform(X)

    raw_preds  = model.predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)

    df["prediction"]  = np.where(raw_preds == -1, "FRAUD", "SAFE")
    df["risk_score"]  = np.clip(((-raw_scores + 0.3) / 0.6) * 100, 0, 100).round(1)
    df["anomaly_raw"] = raw_scores.round(4)
    return df


def generate_sample_csv() -> bytes:
    rng = np.random.default_rng(42)
    n = 30
    data = {
        "amt":         rng.exponential(80, n).round(2),
        "lat":         rng.uniform(25, 48, n).round(4),
        "long":        rng.uniform(-122, -70, n).round(4),
        "city_pop":    rng.integers(500, 2_000_000, n),
        "merch_lat":   rng.uniform(25, 48, n).round(4),
        "merch_long":  rng.uniform(-122, -70, n).round(4),
        "hour":        rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "age":         rng.integers(18, 75, n),
        "amt_log":     np.log1p(rng.exponential(80, n)).round(4),
    }
    data["amt"][-3:]      = [4500.0, 9999.99, 3200.0]
    data["hour"][-3:]     = [2, 3, 4]
    data["city_pop"][-3:] = [120, 80, 200]
    data["amt_log"][-3:]  = np.log1p(data["amt"][-3:]).round(4)
    return pd.DataFrame(data).to_csv(index=False).encode()

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS FOR INSTANT THEME & FONT CONTROL
# ─────────────────────────────────────────────────────────────────────────────
def on_theme_change():
    st.session_state["theme"] = st.session_state["theme_radio_input"]

def on_font_change():
    st.session_state["font_size"] = st.session_state["font_select_input"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROL PANEL (ZERO EMOJIS)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='font-family:Orbitron; color:" + current_theme['accent_primary'] + "; margin-bottom:0;'>NEURAL SENTINEL</h2>", unsafe_allow_html=True)
    st.markdown("<span class='cyber-tag tag-cyan'>ENTERPRISE // v4.0</span>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("[SYS_CONFIG]", expanded=True):
        selected_lang = st.selectbox(
            "Idioma / Language",
            options=["PT", "EN", "ES", "FR", "DE", "ZH"],
            format_func=lambda x: {
                "PT": "Português (PT)",
                "EN": "English (EN)",
                "ES": "Español (ES)",
                "FR": "Français (FR)",
                "DE": "Deutsch (DE)",
                "ZH": "中文 (ZH)"
            }[x],
            key="selected_lang"
        )

        st.radio(
            "Tema de Cores",
            options=["Dark Cyber", "Red Crimson", "Light Neon"],
            index=["Dark Cyber", "Red Crimson", "Light Neon"].index(st.session_state["theme"]),
            key="theme_radio_input",
            on_change=on_theme_change,
            horizontal=True
        )

        st.select_slider(
            "Tamanho da Fonte",
            options=["Normal", "Grande", "Extragrande"],
            value=st.session_state["font_size"],
            key="font_select_input",
            on_change=on_font_change
        )

        st.session_state["cookies_enabled"] = st.toggle(
            "Manter Sessão / Cookies",
            value=st.session_state["cookies_enabled"]
        )

        if st.session_state["cookies_enabled"]:
            st.caption("Status: Cookies/Sessão Ativos")
        else:
            st.caption("Status: Sessão Anônima")

    t = LANG_TEXTS[selected_lang]

    st.markdown("---")
    model_ok = model is not None and scaler is not None
    if model_ok:
        st.success(f"[STATUS] {t['status_ready']}")
    else:
        st.error(f"[STATUS] {t['status_offline']}")

    st.markdown("---")
    st.markdown(f"### {t['arch_overview']}")
    st.markdown(t['arch_desc'])
    st.markdown("---")

    hist_df = pd.DataFrame(st.session_state["history"])
    if not hist_df.empty:
        total  = len(hist_df)
        frauds = (hist_df["verdict"] == "FRAUD").sum()
        st.markdown(f"### {t['session_telemetry']}")
        c1, c2 = st.columns(2)
        c1.metric(t['analyzed'], total)
        c2.metric(t['flagged'], frauds)
        if st.button(t['clear_telemetry'], use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER BANNER (ZERO EMOJIS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="cyber-header">
    <div class="cyber-title">{t['title']}</div>
    <div class="cyber-subtitle">{t['subtitle']}</div>
    <div style="margin-top: 0.8rem;">
        <span class="cyber-tag tag-cyan">[ISOLATION FOREST]</span>
        <span class="cyber-tag tag-pink">[STREAMLIT + PLOTLY]</span>
        <span class="cyber-tag tag-green">[1.3M DATASET TRAINED]</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if not model_ok:
    st.warning(t['status_offline'])
    st.stop()

# Modern, clean Tab headers (Zero emojis)
tab1, tab2, tab3, tab4 = st.tabs(
    [
        f"► {t['tab_simulate']}", 
        f"► {t['tab_batch']}", 
        f"► {t['tab_insights']}", 
        f"► {t['tab_about']}"
    ]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATE TRANSACTION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"### {t['sim_title']}")
    st.markdown(t['sim_desc'])

    with st.form("transaction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>[PAYLOAD_DATA] {t['payload_details']}</h4>", unsafe_allow_html=True)
            amt = st.number_input(t['amt'], min_value=0.01, max_value=25_000.0, value=85.50, step=0.01, format="%.2f")
            hour = st.slider(t['hour'], 0, 23, 14)
            day_of_week = st.selectbox(
                t['day'],
                options=list(range(7)),
                format_func=lambda x: t['days'][x],
                index=1,
            )

        with c2:
            st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>[NODE_LOCATION] {t['node_coords']}</h4>", unsafe_allow_html=True)
            lat  = st.number_input(t['card_lat'],  value=40.71, format="%.4f")
            long = st.number_input(t['card_long'], value=-74.00, format="%.4f")
            age  = st.slider(t['age'], 18, 90, 35)
            city_pop = st.number_input(t['city_pop'], min_value=100, max_value=5_000_000, value=250_000, step=1000)

        with c3:
            st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>[TARGET_MERCHANT] {t['merchant_target']}</h4>", unsafe_allow_html=True)
            merch_lat  = st.number_input(t['merch_lat'],  value=40.73, format="%.4f")
            merch_long = st.number_input(t['merch_long'], value=-73.93, format="%.4f")

        submitted = st.form_submit_button(t['btn_run'], use_container_width=True, type="primary")

    if submitted:
        amt_log = float(np.log1p(amt))
        values = dict(
            amt=amt, lat=lat, long=long, city_pop=city_pop,
            merch_lat=merch_lat, merch_long=merch_long,
            hour=hour, day_of_week=day_of_week, age=age, amt_log=amt_log,
        )

        with st.spinner("Processing isolation tree path length…"):
            time.sleep(0.15)
            label, raw_score, risk_pct = predict_transaction(values)

        if label == "FRAUD":
            st.markdown(
                f'<div class="verdict-cyber-fraud"><h2>{t["verdict_fraud_title"]}</h2>'
                f'<p style="color:{current_theme["text_color"]};margin:0.4rem 0 0">{t["verdict_fraud_desc"]}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-cyber-safe"><h2>{t["verdict_safe_title"]}</h2>'
                f'<p style="color:{current_theme["text_color"]};margin:0.4rem 0 0">{t["verdict_safe_desc"]}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        
        sim_chart_tab1, sim_chart_tab2 = st.tabs(["[DIAGNOSTICS] Medidor & Diagnósticos", "[RADAR] Perfil Radar & Comparativo"])

        with sim_chart_tab1:
            col_gauge, col_explain = st.columns([1, 1])
            with col_gauge:
                st.plotly_chart(score_gauge(risk_pct), use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center;color:{current_theme['subtext_color']};font-family:JetBrains Mono;font-size:0.85rem;'>"
                    f"{t['raw_score']}: <b style='color:{current_theme['accent_primary']}'>{raw_score:.4f}</b></p>",
                    unsafe_allow_html=True,
                )

            with col_explain:
                explanation = explain_prediction(values, label, risk_pct, t)
                st.markdown(explanation)

        with sim_chart_tab2:
            st.markdown("#### [RADAR] Gráfico de Radar: Atributos da Transação vs Linha de Base Padrão")
            col_radar, col_bar = st.columns([1.2, 1])
            with col_radar:
                st.plotly_chart(create_radar_chart(values), use_container_width=True)
            with col_bar:
                features_names = ['Valor', 'Hora', 'Idade', 'Distância Geo', 'Densidade Pop']
                dist_geo = abs(values.get('lat', 0) - values.get('merch_lat', 0)) * 100
                variances = [values.get('amt', 0), values.get('hour', 0)*10, values.get('age', 0), dist_geo, values.get('city_pop', 0)/5000]
                
                fig_var = px.bar(
                    x=features_names, y=variances,
                    title="Variação de Atributos do Payload",
                    labels={'x': 'Atributo', 'y': 'Magnitude'},
                    color_discrete_sequence=[current_theme['accent_primary']],
                    template=current_theme['plotly_template']
                )
                fig_var.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=current_theme['text_color'], family="Rajdhani"),
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_var, use_container_width=True)

        st.session_state["history"].append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "amount":    f"${amt:.2f}",
                "hour":      hour,
                "age":       age,
                "verdict":   label,
                "risk_pct":  round(risk_pct, 1),
            }
        )

    if st.session_state["history"]:
        st.markdown("---")
        st.markdown(f"### [LOG] {t['session_log']}")
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### {t['batch_title']}")
    st.markdown(t['batch_desc'])

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(t['upload_label'], type=["csv"])

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            t['btn_download_sample'],
            data=generate_sample_csv(),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )
        use_sample = st.button(t['btn_test_sample'], use_container_width=True)

    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"[PAYLOAD_OK] Loaded Payload: {len(df_batch):,} records successfully.")
        except Exception as e:
            st.error(f"[ERROR] Failed to read CSV: {e}")

    elif use_sample:
        df_batch = pd.read_csv(io.BytesIO(generate_sample_csv()))
        st.info("[SAMPLE_ACTIVE] Using 30 synthetic transactions sample payload.")

    if df_batch is not None:
        with st.spinner("Processing isolation forest inference..."):
            result_df = batch_predict(df_batch)

        total_tx = len(result_df)
        fraud_tx = (result_df["prediction"] == "FRAUD").sum()
        safe_tx  = total_tx - fraud_tx
        avg_risk = result_df["risk_score"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t['total_executed'], f"{total_tx:,}")
        m2.metric(t['flagged_anomaly'], f"{fraud_tx:,}", delta=f"{fraud_tx/total_tx:.2%}")
        m3.metric(t['verified_safe'], f"{safe_tx:,}")
        m4.metric(t['avg_risk'], f"{avg_risk:.1f}%")

        st.markdown("---")
        st.markdown("### [ANALYTICS] Painel de Análise Gráfica Avançada")

        b_tab1, b_tab2, b_tab3, b_tab4 = st.tabs([
            "[OVERVIEW] Distribuição Geral", 
            "[TEMPORAL] Padrões Temporais", 
            "[SCATTER] Relação Valor x Risco", 
            "[DEMO] Geografia & Demografia"
        ])

        with b_tab1:
            col_pie, col_hist = st.columns(2)
            with col_pie:
                pie_data = result_df["prediction"].value_counts().reset_index()
                pie_data.columns = ["Prediction", "Count"]
                fig_pie = px.pie(
                    pie_data, names="Prediction", values="Count",
                    color="Prediction",
                    color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
                    title=t['chart_pie_title'],
                    hole=0.5,
                    template=current_theme['plotly_template']
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=current_theme['text_color'],
                    font_family="Rajdhani",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_hist:
                fig_hist = px.histogram(
                    result_df, x="risk_score", color="prediction",
                    nbins=25, barmode="overlay",
                    color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
                    title=t['chart_hist_title'],
                    labels={"risk_score": "Risk Percentage (%)", "count": "Frequency"},
                    template=current_theme['plotly_template']
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=current_theme['text_color'],
                    font_family="Rajdhani",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        with b_tab2:
            col_h, col_d = st.columns(2)
            with col_h:
                if "hour" in result_df.columns:
                    hour_fraud = result_df.groupby(["hour", "prediction"]).size().reset_index(name="count")
                    fig_hour = px.bar(
                        hour_fraud, x="hour", y="count", color="prediction",
                        color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
                        title="Transações por Hora do Dia (0-23h)",
                        barmode="stack",
                        template=current_theme['plotly_template']
                    )
                    fig_hour.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color=current_theme['text_color'],
                        font_family="Rajdhani",
                    )
                    st.plotly_chart(fig_hour, use_container_width=True)
            with col_d:
                if "day_of_week" in result_df.columns:
                    day_fraud = result_df.groupby(["day_of_week", "prediction"]).size().reset_index(name="count")
                    fig_day = px.bar(
                        day_fraud, x="day_of_week", y="count", color="prediction",
                        color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
                        title="Transações por Dia da Semana (0=Seg, 6=Dom)",
                        barmode="group",
                        template=current_theme['plotly_template']
                    )
                    fig_day.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color=current_theme['text_color'],
                        font_family="Rajdhani",
                    )
                    st.plotly_chart(fig_day, use_container_width=True)

        with b_tab3:
            if "amt" in result_df.columns:
                fig_scatter_batch = px.scatter(
                    result_df, x="amt", y="risk_score", color="prediction",
                    size="risk_score",
                    color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
                    title="Dispersão: Valor da Transação ($) vs Pontuação de Risco (%)",
                    labels={"amt": "Valor ($)", "risk_score": "Risco (%)"},
                    template=current_theme['plotly_template']
                )
                fig_scatter_batch.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=current_theme['text_color'],
                    font_family="Rajdhani",
                )
                st.plotly_chart(fig_scatter_batch, use_container_width=True)

        with b_tab4:
            col_pop, col_dist = st.columns(2)
            with col_pop:
                if "city_pop" in result_df.columns:
                    fig_pop = px.box(
                        result_df, x="prediction", y="city_pop", color="prediction",
                        color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
                        title="Distribuição de População da Cidade por Status",
                        labels={"city_pop": "População"},
                        template=current_theme['plotly_template']
                    )
                    fig_pop.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color=current_theme['text_color'],
                        font_family="Rajdhani",
                    )
                    st.plotly_chart(fig_pop, use_container_width=True)
            with col_dist:
                if "age" in result_df.columns:
                    fig_age = px.histogram(
                        result_df, x="age", color="prediction", barmode="overlay",
                        color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
                        title="Distribuição por Faixa Etária do Titular",
                        labels={"age": "Idade"},
                        template=current_theme['plotly_template']
                    )
                    fig_age.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color=current_theme['text_color'],
                        font_family="Rajdhani",
                    )
                    st.plotly_chart(fig_age, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### [DATA_GRID] {t['grid_title']}")

        def highlight_fraud(val):
            if val == "FRAUD":
                return f"background-color: rgba(220, 38, 38, 0.2); color: {current_theme['accent_red']}; font-weight: bold;"
            return f"background-color: rgba(22, 163, 74, 0.15); color: {current_theme['accent_green']}; font-weight: bold;"

        show_cols = ["prediction", "risk_score", "anomaly_raw"] + [
            c for c in FEATURES if c in result_df.columns
        ]

        DISPLAY_LIMIT = 5000
        if len(result_df) > DISPLAY_LIMIT:
            st.info(t['large_batch_notice'].format(n=len(result_df)))
            display_df = result_df.head(DISPLAY_LIMIT)
        else:
            display_df = result_df

        styled_table = display_df[show_cols].style.map(
            highlight_fraud, subset=["prediction"]
        )
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

        st.download_button(
            t['btn_download_results'],
            data=result_df.to_csv(index=False).encode(),
            file_name="neural_fraud_analysis_results.csv",
            mime="text/csv",
            type="primary"
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### [MODEL_SPEC] Neural Engine Parameters & Architecture")

    col_params, col_info = st.columns([1, 2])
    with col_params:
        st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>[PARAMS] {t['model_params']}</h4>", unsafe_allow_html=True)
        params = model.get_params()
        param_df = pd.DataFrame(list(params.items()), columns=["Hyperparameter", "Value"])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

    with col_info:
        st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>[THEORY] {t['theory_title']}</h4>", unsafe_allow_html=True)
        st.markdown(t['theory_desc'])

    st.markdown("---")
    st.markdown("### [SIMULATOR] Monte Carlo Visualizer")
    if st.button(t['btn_monte_carlo'], use_container_width=False):
        rng = np.random.default_rng()
        n   = 50
        rand_data = {
            "amt":         rng.exponential(100, n),
            "lat":         rng.uniform(25, 48, n),
            "long":        rng.uniform(-122, -70, n),
            "city_pop":    rng.integers(100, 2_000_000, n).astype(float),
            "merch_lat":   rng.uniform(25, 48, n),
            "merch_long":  rng.uniform(-122, -70, n),
            "hour":        rng.integers(0, 24, n).astype(float),
            "day_of_week": rng.integers(0, 7, n).astype(float),
            "age":         rng.integers(18, 80, n).astype(float),
        }
        rand_df = pd.DataFrame(rand_data)
        rand_df["amt_log"] = np.log1p(rand_df["amt"])
        result = batch_predict(rand_df)

        fig_scatter = px.scatter(
            result,
            x="amt", y="risk_score",
            color="prediction",
            color_discrete_map={"FRAUD": current_theme['accent_red'], "SAFE": current_theme['accent_green']},
            size="risk_score",
            hover_data=["hour", "age", "city_pop"],
            title="Amount vs Risk Score Distribution",
            labels={"amt": "Transaction Amount ($)", "risk_score": "Risk Percentage (%)"},
            template=current_theme['plotly_template']
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=current_theme['text_color'],
            font_family="Rajdhani",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT / ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"### [SPEC] {t['arch_overview']}")
    st.markdown(
        """
## Neural Fraud Sentinel Enterprise

End-to-end Machine Learning pipeline for real-time credit card fraud detection.

---

### Core Technology Stack

- **ML Core**: `scikit-learn` Isolation Forest (Unsupervised Anomaly Detection)
- **Data Engineering**: `pandas`, `numpy`, `StandardScaler`
- **Dashboard**: `Streamlit`, `Plotly Express` (Dynamic Theme Engine)
- **Dataset**: Kaggle Credit Card Fraud Detection dataset (**1,296,675 records**)
        """
    )
