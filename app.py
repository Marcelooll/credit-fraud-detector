"""
app.py - FraudSentinel — Enterprise Anomaly Engine
===================================================
Run with:  streamlit run app.py
"""

import os
import io
import json
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
APP_NAME = "FraudSentinel"
APP_VERSION = "1.2"
APP_DEPLOY_URL = "https://share.streamlit.io/Marcelooll/credit-fraud-detector/main/app.py"

st.set_page_config(
    page_title=f"{APP_NAME} // Enterprise Anomaly Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & MODEL PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "isolation_forest.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join("model", "feature_names.pkl")
AUDIT_HISTORY_FILE = os.path.join("data", "simulation_history.json")

def load_simulation_history() -> list:
    try:
        if os.path.exists(AUDIT_HISTORY_FILE):
            with open(AUDIT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []

def save_simulation_history(history_list: list):
    try:
        os.makedirs(os.path.dirname(AUDIT_HISTORY_FILE), exist_ok=True)
        with open(AUDIT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def clear_simulation_history():
    try:
        if os.path.exists(AUDIT_HISTORY_FILE):
            with open(AUDIT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
    except Exception:
        pass

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
        "subtext_color": "#475569",
        "sidebar_bg": "#f1f5f9",
        "border_color": "#cbd5e1",
        "input_bg": "#ffffff",
        "plotly_template": "plotly_white"
    }
}

FONT_SCALES = {
    "Normal": "1.0rem",
    "Grande": "1.15rem",
    "Extragrande": "1.3rem"
}

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT SESSION STATE & COOKIE SYNCHRONIZATION
# ─────────────────────────────────────────────────────────────────────────────
qp = st.query_params
cookies_in_url = qp.get("cookies", "0") == "1"

if "cookies_enabled" not in st.session_state:
    st.session_state["cookies_enabled"] = cookies_in_url

if "theme" not in st.session_state:
    stored_theme = qp.get("theme", "Dark Cyber")
    st.session_state["theme"] = stored_theme if stored_theme in THEME_CONFIGS else "Dark Cyber"

if "font_size" not in st.session_state:
    stored_font = qp.get("font", "Normal")
    st.session_state["font_size"] = stored_font if stored_font in FONT_SCALES else "Normal"

if "selected_lang" not in st.session_state:
    stored_lang = qp.get("lang", "PT")
    st.session_state["selected_lang"] = stored_lang if stored_lang in ["PT", "EN"] else "PT"

if "history" not in st.session_state:
    st.session_state["history"] = load_simulation_history()

current_theme = THEME_CONFIGS.get(st.session_state["theme"], THEME_CONFIGS["Dark Cyber"])
selected_font_scale = FONT_SCALES.get(st.session_state["font_size"], "1.0rem")

# ─────────────────────────────────────────────────────────────────────────────
# INTERNATIONALIZATION (I18N — PT & EN EXCLUSIVELY)
# ─────────────────────────────────────────────────────────────────────────────
LANG_TEXTS = {
    "PT": {
        "title": "FraudSentinel",
        "subtitle": "Motor Neural de Detecção de Anomalias em Cartão de Crédito via Machine Learning Não Supervisionado",
        "status_ready": "MOTOR ISOLATION FOREST: OPERACIONAL",
        "status_offline": "Modelo Offline. Execute: python train.py",
        "arch_overview": "Arquitetura do Sistema",
        "arch_desc": "Alimentado por um algoritmo Isolation Forest treinado em 1,3 milhão de transações reais do Kaggle.",
        "session_telemetry": "Telemetria de Sessão",
        "analyzed": "Analisadas",
        "flagged": "Anomalias",
        "clear_telemetry": "Limpar Auditoria",
        "clear_all_cookies": "Redefinir Preferências",
        "tab_simulate": "SIMULAÇÃO AO VIVO",
        "tab_batch": "PROCESSADOR EM LOTE",
        "tab_insights": "INSIGHTS & PARÂMETROS",
        "tab_about": "ARQUITETURA DO PROJETO",
        "sim_title": "Simulador de Telemetria Transacional em Tempo Real",
        "sim_desc": "Injete parâmetros de transação no motor de inteligência artificial para diagnosticar o risco de anomalia instantaneamente.",
        "payload_details": "Parâmetros da Transação",
        "node_coords": "Coordenadas do Titular",
        "merchant_target": "Estabelecimento Comercial",
        "amt": "Valor da Transação ($)",
        "hour": "Hora do Evento (0-23h)",
        "day": "Dia da Semana",
        "card_lat": "Latitude do Titular",
        "card_long": "Longitude do Titular",
        "age": "Idade do Titular",
        "city_pop": "População da Cidade",
        "merch_lat": "Latitude do Estabelecimento",
        "merch_long": "Longitude do Estabelecimento",
        "btn_run": "EXECUTAR DIAGNÓSTICO",
        "verdict_fraud_title": "ANOMALIA DETECTADA // RISCO ELEVADO DE FRAUDE",
        "verdict_fraud_desc": "O comprimento médio dos caminhos nas árvores de isolamento colapsou rapidamente. Transação isolada como anômala.",
        "verdict_safe_title": "TRANSAÇÃO VERIFICADA // COMPORTAMENTO NOMINAL",
        "verdict_safe_desc": "A telemetria da transação se alinha perfeitamente com os agrupamentos legítimos da linha de base.",
        "raw_score": "Score Bruto do Modelo",
        "diagnostics": "Diagnóstico dos Fatores de Risco",
        "session_log": "Log de Auditoria da Sessão",
        "log_col_time": "Data/Hora Padrão (Requisição)",
        "log_col_sim": "Janela Simulada",
        "log_col_amt": "Valor",
        "log_col_age": "Idade",
        "log_col_verdict": "Veredito",
        "log_col_risk": "Risco (%)",
        "batch_title": "Processamento Massivo em Lote",
        "batch_desc": "Envie arquivos CSV com grandes volumes de transações para inferência vetorizada e análise estatística avançada.",
        "upload_label": "Carregar Dataset CSV",
        "btn_download_sample": "Baixar CSV de Exemplo",
        "btn_test_sample": "Testar Amostra Sintética (30 registros)",
        "total_executed": "Total Avaliado",
        "flagged_anomaly": "Sinalizadas como Anomalia",
        "verified_safe": "Verificadas como Seguras",
        "avg_risk": "Risco Médio",
        "chart_pie_title": "Distribuição Geral: Anomalias vs Seguras",
        "chart_hist_title": "Espectro da Distribuição de Risco",
        "grid_title": "Tabela de Resultados Avaliados",
        "btn_download_results": "BAIXAR CSV COMPLETO COM PREDIÇÕES",
        "large_batch_notice": "Modo de Alto Volume Ativo ({n:,} registros). Exibindo prévia das primeiras 5.000 linhas.",
        "model_params": "Hiperparâmetros do Modelo",
        "theory_title": "Fundamentos da Isolation Forest",
        "theory_desc": "Diferente de modelos supervisionados tradicionais que exigem rótulos históricos de fraude (frequentemente escassos e desbalanceados), a Isolation Forest detecta anomalias isolando observações atípicas por meio de particionamentos aleatórios recursivos no espaço vetorial.",
        "btn_monte_carlo": "Executar Teste Estocástico de Monte Carlo (50 amostras)",
        "days": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
        "lang_label": "Idioma / Language",
        "sys_config": "Configurações do Sistema",
        "theme_label": "Tema Visual",
        "font_label": "Escala de Fonte",
        "cookie_corner_title": "Preferências de Cookies",
        "cookie_corner_desc": "Utilizamos cookies e armazenamento local para manter suas preferências de tema, idioma e histórico de auditoria salvas.",
        "cookie_accept_btn": "Aceitar",
        "cookie_decline_btn": "Recusar",
        "sidebar_title": "FraudSentinel",
        "sidebar_tag": "ENTERPRISE // v1.2",
        "analytics_title": "Painel de Análise Gráfica Avançada",
        "overview_tab": "Distribuição Geral",
        "temporal_tab": "Padrões Temporais",
        "scatter_tab": "Relação Valor x Risco",
        "geography_tab": "Geografia & Demografia",
        "hour_title": "Volume de Transações por Hora do Dia (0-23h)",
        "day_title": "Transações por Dia da Semana (0=Seg, 6=Dom)",
        "scatter_title": "Dispersão: Valor da Transação ($) vs Pontuação de Risco (%)",
        "pop_title": "Distribuição da População da Cidade por Status",
        "age_title": "Distribuição por Faixa Etária do Titular",
        "model_spec_title": "Parâmetros e Arquitetura do Modelo",
        "monte_carlo_title": "Simulador Estocástico de Monte Carlo",
        "monte_carlo_scatter_title": "Distribuição Monte Carlo: Valor x Pontuação de Risco",
        "chart_amount_label": "Valor da Transação ($)",
        "chart_risk_label": "Percentual de Risco (%)",
        "chart_prediction_label": "Predição",
        "chart_count_label": "Contagem",
        "chart_frequency_label": "Frequência",
        "chart_population_label": "População",
        "chart_age_label": "Idade",
        "about_title": "FraudSentinel — Enterprise Anomaly Engine",
        "about_intro": "Pipeline de Machine Learning e Engenharia de Dados para detecção de fraudes financeiras em tempo real sem dependência de rótulos prévios.",
        "about_stack_title": "Stack Tecnológica & Engenharia",
        "about_ml_core": "Núcleo de ML",
        "about_data_eng": "Engenharia de Dados",
        "about_dashboard": "Interface & Visualização",
        "about_dataset": "Base de Treinamento",
        "sim_tabs_diagnostics": "Diagnósticos & Indicadores",
        "sim_tabs_radar": "Perfil Radar Multidimensional",
        "radar_title": "Radar Multidimensional: Transação Atual vs Linha de Base Nominal",
        "radar_category_value": "Valor (Norm)",
        "radar_category_hour": "Hora (Norm)",
        "radar_category_geo": "Distância Geo",
        "radar_category_age": "Idade (Norm)",
        "radar_category_pop": "Densidade Pop",
        "radar_baseline": "Linha de Base Segura",
        "radar_current": "Transação Avaliada",
        "payload_bar_title": "Magnitude dos Atributos da Transação",
        "payload_bar_x": "Atributo",
        "payload_bar_y": "Magnitude",
        "spinner_simulate": "Calculando profundidade média de isolamento...",
        "spinner_batch": "Processando inferência vetorizada em lote...",
        "status_payload_loaded": "Carga de dados carregada com sucesso: {n:,} registros.",
        "status_csv_error": "Erro ao ler o arquivo CSV: {error}",
        "status_sample_active": "Amostra sintética de 30 registros carregada.",
        "reason_val": "Pico de Valor: ${val:.2f} excede substancialmente a média de gastos usuais.",
        "reason_time": "Horário Crítico: Transação executada na madrugada ({h}:00h).",
        "reason_pop": "Baixa Densidade Populacional: Município com {pop:,} habitantes diverge da linha de base.",
        "reason_geo": "Anomalia Geodésica: Distância física acentuada entre o domicílio do titular e o terminal de cobrança.",
        "reason_weekend": "Janela de Fim de Semana: Operação realizada no sábado/domingo.",
        "reason_ok": "Telemetria Nominal: Todos os atributos estão perfeitamente em conformidade com o perfil legítimo."
    },
    "EN": {
        "title": "FraudSentinel",
        "subtitle": "Unsupervised Machine Learning Real-Time Credit Card Anomaly Engine",
        "status_ready": "ISOLATION FOREST ENGINE: OPERATIONAL",
        "status_offline": "Model Offline. Run: python train.py",
        "arch_overview": "Architecture Overview",
        "arch_desc": "Powered by an unsupervised Isolation Forest algorithm trained on 1.3M real Kaggle transactions.",
        "session_telemetry": "Session Telemetry",
        "analyzed": "Analyzed",
        "flagged": "Anomalies",
        "clear_telemetry": "Clear Audit Log",
        "clear_all_cookies": "Reset Preferences",
        "tab_simulate": "LIVE SIMULATION",
        "tab_batch": "BATCH PROCESSOR",
        "tab_insights": "INSIGHTS & PARAMETERS",
        "tab_about": "PROJECT ARCHITECTURE",
        "sim_title": "Real-Time Telemetry Diagnostic Simulator",
        "sim_desc": "Inject transaction parameters into the AI anomaly engine to calculate fraud probability instantly.",
        "payload_details": "Transaction Parameters",
        "node_coords": "Cardholder Coordinates",
        "merchant_target": "Merchant Coordinates",
        "amt": "Transaction Amount ($)",
        "hour": "Execution Hour (0-23h)",
        "day": "Day of Week",
        "card_lat": "Cardholder Latitude",
        "card_long": "Cardholder Longitude",
        "age": "Cardholder Age",
        "city_pop": "City Population",
        "merch_lat": "Merchant Latitude",
        "merch_long": "Merchant Longitude",
        "btn_run": "RUN DIAGNOSTIC",
        "verdict_fraud_title": "ANOMALY DETECTED // ELEVATED RISK",
        "verdict_fraud_desc": "Isolation tree average path length collapsed rapidly. Payload isolated as an anomalous transaction.",
        "verdict_safe_title": "TRANSACTION VERIFIED // NOMINAL BEHAVIOR",
        "verdict_safe_desc": "Transaction telemetry aligns with standard baseline legitimate consumer clusters.",
        "raw_score": "Raw Decision Score",
        "diagnostics": "Risk Factor Diagnostics",
        "session_log": "Session Audit Log",
        "log_col_time": "Standard Timestamp (Request)",
        "log_col_sim": "Simulated Window",
        "log_col_amt": "Amount",
        "log_col_age": "Age",
        "log_col_verdict": "Verdict",
        "log_col_risk": "Risk (%)",
        "batch_title": "Massive Batch Processing Engine",
        "batch_desc": "Upload CSV files for high-throughput vectorized inference and comprehensive statistical profiling.",
        "upload_label": "Upload CSV Dataset",
        "btn_download_sample": "Download Sample CSV",
        "btn_test_sample": "Test Synthetic Sample (30 records)",
        "total_executed": "Total Analyzed",
        "flagged_anomaly": "Flagged Anomalies",
        "verified_safe": "Verified Safe",
        "avg_risk": "Average Risk",
        "chart_pie_title": "Overall Distribution: Anomaly vs Safe",
        "chart_hist_title": "Risk Spectrum Distribution",
        "grid_title": "Evaluated Results Grid",
        "btn_download_results": "DOWNLOAD COMPLETE CSV PREDICTIONS",
        "large_batch_notice": "High Volume Mode Active ({n:,} records). Displaying preview of top 5,000 rows.",
        "model_params": "Model Hyperparameters",
        "theory_title": "Isolation Forest Fundamentals",
        "theory_desc": "Unlike supervised classifiers that require historical fraud labels (frequently scarce and imbalanced), Isolation Forest isolates anomalies by recursively partitioning feature space via random binary trees.",
        "btn_monte_carlo": "Run Monte Carlo Stochastic Test (50 samples)",
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "lang_label": "Language / Idioma",
        "sys_config": "System Configuration",
        "theme_label": "Visual Theme",
        "font_label": "Font Scale",
        "cookie_corner_title": "Cookie Preferences",
        "cookie_corner_desc": "We use cookies and local storage to preserve your theme, language, and session audit history.",
        "cookie_accept_btn": "Accept",
        "cookie_decline_btn": "Decline",
        "sidebar_title": "FraudSentinel",
        "sidebar_tag": "ENTERPRISE // v1.2",
        "analytics_title": "Advanced Graph Analysis Panel",
        "overview_tab": "Overall Distribution",
        "temporal_tab": "Temporal Patterns",
        "scatter_tab": "Amount vs Risk Relation",
        "geography_tab": "Geography & Demographics",
        "hour_title": "Transaction Volume by Hour of Day (0-23h)",
        "day_title": "Transactions by Day of Week (0=Mon, 6=Sun)",
        "scatter_title": "Scatter: Transaction Amount ($) vs Risk Score (%)",
        "pop_title": "City Population Distribution by Status",
        "age_title": "Distribution by Cardholder Age Group",
        "model_spec_title": "Model Parameters & Engine Architecture",
        "monte_carlo_title": "Monte Carlo Stochastic Simulator",
        "monte_carlo_scatter_title": "Monte Carlo Distribution: Amount vs Risk Score",
        "chart_amount_label": "Transaction Amount ($)",
        "chart_risk_label": "Risk Percentage (%)",
        "chart_prediction_label": "Prediction",
        "chart_count_label": "Count",
        "chart_frequency_label": "Frequency",
        "chart_population_label": "Population",
        "chart_age_label": "Age",
        "about_title": "FraudSentinel — Enterprise Anomaly Engine",
        "about_intro": "End-to-end machine learning and data engineering pipeline for zero-label real-time financial fraud detection.",
        "about_stack_title": "Core Technology Stack & Engineering",
        "about_ml_core": "ML Core",
        "about_data_eng": "Data Engineering",
        "about_dashboard": "Interface & Visualization",
        "about_dataset": "Training Corpus",
        "sim_tabs_diagnostics": "Diagnostics & Metrics",
        "sim_tabs_radar": "Multidimensional Radar Profile",
        "radar_title": "Multidimensional Radar: Current Transaction vs Nominal Baseline",
        "radar_category_value": "Amount (Norm)",
        "radar_category_hour": "Hour (Norm)",
        "radar_category_geo": "Geo Distance",
        "radar_category_age": "Age (Norm)",
        "radar_category_pop": "Pop Density",
        "radar_baseline": "Nominal Baseline",
        "radar_current": "Evaluated Transaction",
        "payload_bar_title": "Transaction Attribute Magnitude",
        "payload_bar_x": "Attribute",
        "payload_bar_y": "Magnitude",
        "spinner_simulate": "Computing isolation path lengths...",
        "spinner_batch": "Processing vectorized batch inference...",
        "status_payload_loaded": "Payload loaded successfully: {n:,} records.",
        "status_csv_error": "Failed to parse CSV file: {error}",
        "status_sample_active": "Synthetic sample payload of 30 records active.",
        "reason_val": "Amount Spike: ${val:.2f} significantly exceeds standard consumer baseline.",
        "reason_time": "Critical Time Window: Transaction initiated late at night ({h}:00 AM).",
        "reason_pop": "Low Population Density: Node locality with {pop:,} residents deviates from baseline.",
        "reason_geo": "Geodesic Anomaly: Unusually large physical delta between cardholder node and point of sale.",
        "reason_weekend": "Weekend Window: Activity registered during non-business weekend hours.",
        "reason_ok": "Nominal Telemetry: All parameters align with standard legitimate transaction profile."
    }
}

selected_lang = st.session_state.get("selected_lang", "PT")
t = LANG_TEXTS.get(selected_lang, LANG_TEXTS["PT"])

def get_text(key: str, fallback: str = "") -> str:
    return t.get(key, fallback or key)

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC STYLESHEET (PERFECT HIGH CONTRAST & RESPONSIVE DESIGN)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800;900&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Streamlit Header / Toolbar */
header[data-testid="stHeader"] {{ visibility: visible !important; }}
div[data-testid="stToolbar"] {{ visibility: visible !important; display: flex !important; }}
footer {{ visibility: visible !important; }}

/* Root Styling */
html, body, [data-testid="stAppViewContainer"], .stApp {{
    background-color: {current_theme['bg_color']} !important;
    font-family: 'Rajdhani', sans-serif;
    font-size: {selected_font_scale} !important;
    color: {current_theme['text_color']} !important;
}}

/* Global Typography Overrides */
p, span, label, h1, h2, h3, h4, h5, h6, li, td, th, div, .stMarkdown {{
    color: {current_theme['text_color']} !important;
}}

/* Form Widget Labels */
div[data-testid="stWidgetLabel"] label, 
div[data-testid="stWidgetLabel"] p, 
div[data-testid="stWidgetLabel"] span {{
    color: {current_theme['text_color']} !important;
    font-weight: 600 !important;
    font-size: calc({selected_font_scale} * 0.95) !important;
}}

/* Form Inputs, Selectboxes & Number Inputs */
input, select, textarea, [data-baseweb="input"], [data-baseweb="select"] {{
    color: {current_theme['text_color']} !important;
    background-color: {current_theme['input_bg']} !important;
    border: 1px solid {current_theme['border_color']} !important;
    border-radius: 6px !important;
}}

/* Selectbox Dropdown Menu Options & Popover */
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
    background-color: {current_theme['card_bg']} !important;
    color: {current_theme['text_color']} !important;
    border: 1px solid {current_theme['border_color']} !important;
}}
li[role="option"] {{
    background-color: {current_theme['card_bg']} !important;
    color: {current_theme['text_color']} !important;
}}
li[role="option"]:hover {{
    background-color: {current_theme['sidebar_bg']} !important;
}}

/* Slider ticks and numbers */
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] div,
div[data-testid="stSlider"] span {{
    color: {current_theme['text_color']} !important;
}}

/* Step buttons on Number Inputs */
button[data-testid="stNumberInputStepDown"], 
button[data-testid="stNumberInputStepUp"] {{
    background-color: {current_theme['card_bg']} !important;
    color: {current_theme['text_color']} !important;
    border-color: {current_theme['border_color']} !important;
}}

/* Primary & Secondary Action Buttons */
button[kind="primary"], .stButton > button[kind="primary"] {{
    background-color: {current_theme['accent_primary']} !important;
    color: #ffffff !important;
    border: 1px solid {current_theme['accent_primary']} !important;
    font-weight: 700 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.5px !important;
}}

button[kind="secondary"], .stButton > button, .stDownloadButton > button {{
    background-color: {current_theme['card_bg']} !important;
    color: {current_theme['text_color']} !important;
    border: 1px solid {current_theme['border_color']} !important;
    font-weight: 600 !important;
    font-family: 'Rajdhani', sans-serif !important;
}}

/* File Uploader */
div[data-testid="stFileUploader"] section {{
    background-color: {current_theme['card_bg']} !important;
    border: 1px dashed {current_theme['border_color']} !important;
}}
div[data-testid="stFileUploader"] p, 
div[data-testid="stFileUploader"] span, 
div[data-testid="stFileUploader"] small {{
    color: {current_theme['text_color']} !important;
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
    border-left: 6px solid {current_theme['accent_primary']};
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}}

.cyber-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: calc({selected_font_scale} * 2.1);
    font-weight: 900;
    letter-spacing: 2px;
    color: {current_theme['accent_primary']};
    margin: 0;
    text-transform: uppercase;
}}

.cyber-subtitle {{
    font-family: 'Rajdhani', sans-serif;
    font-size: calc({selected_font_scale} * 1.15);
    color: {current_theme['subtext_color']};
    margin-top: 0.35rem;
    font-weight: 600;
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

.tag-cyan  {{ background: rgba(2, 132, 199, 0.12); color: {current_theme['accent_primary']}; border: 1px solid {current_theme['accent_primary']}; }}
.tag-pink  {{ background: rgba(225, 29, 72, 0.12); color: {current_theme['accent_secondary']}; border: 1px solid {current_theme['accent_secondary']}; }}
.tag-green {{ background: rgba(22, 163, 74, 0.12); color: {current_theme['accent_green']}; border: 1px solid {current_theme['accent_green']}; }}

/* Verdict Banners */
.verdict-cyber-safe {{
    background: rgba(22, 163, 74, 0.1);
    border: 2px solid {current_theme['accent_green']};
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
}}
.verdict-cyber-safe h2 {{
    font-family: 'Orbitron', sans-serif;
    color: {current_theme['accent_green']} !important;
    margin: 0;
    font-size: calc({selected_font_scale} * 1.4);
}}

.verdict-cyber-fraud {{
    background: rgba(220, 38, 38, 0.1);
    border: 2px solid {current_theme['accent_red']};
    border-radius: 12px;
    padding: 1.25rem 1.75rem;
}}
.verdict-cyber-fraud h2 {{
    font-family: 'Orbitron', sans-serif;
    color: {current_theme['accent_red']} !important;
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
    font-size: calc({selected_font_scale} * 0.88) !important;
    letter-spacing: 1px !important;
    color: {current_theme['subtext_color']} !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.25rem !important;
}}
button[aria-selected="true"] {{
    color: {current_theme['accent_primary']} !important;
    border-bottom: 3px solid {current_theme['accent_primary']} !important;
    background: rgba(2, 132, 199, 0.08) !important;
    border-radius: 6px 6px 0 0 !important;
}}

/* Card containers & Expanders */
.stExpander {{
    border: 1px solid {current_theme['border_color']} !important;
    border-radius: 8px !important;
    background-color: {current_theme['card_bg']} !important;
}}
.stExpander details summary, .stExpander details summary p {{
    color: {current_theme['text_color']} !important;
    font-weight: 700 !important;
}}

/* Metric Cards */
[data-testid="stMetricValue"] {{
    font-family: 'Orbitron', sans-serif !important;
    font-size: calc({selected_font_scale} * 1.6) !important;
    color: {current_theme['accent_primary']} !important;
}}

/* Captions & Alerts */
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {current_theme['subtext_color']} !important;
    font-weight: 600 !important;
}}

div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {{
    color: {current_theme['text_color']} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {current_theme['bg_color']}; }}
::-webkit-scrollbar-thumb {{ background: {current_theme['border_color']}; border-radius: 3px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# FLOATING COOKIE CONSENT CORNER WIDGET (STANDARD BROWSER POPUP)
# ─────────────────────────────────────────────────────────────────────────────
cookie_html = f"""
<div id="fraudsentinel-cookie-corner" style="
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 320px;
    background-color: {current_theme['card_bg']};
    border: 1px solid {current_theme['border_color']};
    border-left: 4px solid {current_theme['accent_primary']};
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.45);
    z-index: 9999999;
    font-family: 'Rajdhani', sans-serif;
    display: none;
">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:700; color:{current_theme['accent_primary']}; border:1px solid {current_theme['accent_primary']}; border-radius:4px; padding:1px 5px;">SYS</span>
            <span style="font-family:'Orbitron',sans-serif; font-size:0.85rem; font-weight:700; color:{current_theme['accent_primary']};">
                {t['cookie_corner_title']}
            </span>
        </div>
        <span onclick="window.closeCookieNotice(false)" style="cursor:pointer; color:{current_theme['subtext_color']}; font-size:1.1rem; line-height:1; font-weight:bold;">&times;</span>
    </div>
    <p style="font-size:0.85rem; color:{current_theme['text_color']}; margin:0 0 12px 0; line-height:1.35; font-weight:500;">
        {t['cookie_corner_desc']}
    </p>
    <div style="display:flex; gap:8px; justify-content:flex-end;">
        <button onclick="window.closeCookieNotice(false)" style="
            background:transparent;
            border:1px solid {current_theme['border_color']};
            color:{current_theme['text_color']};
            padding:5px 12px;
            border-radius:6px;
            font-size:0.82rem;
            font-weight:600;
            cursor:pointer;
        ">{t['cookie_decline_btn']}</button>
        <button onclick="window.closeCookieNotice(true)" style="
            background:{current_theme['accent_primary']};
            border:none;
            color:#ffffff;
            padding:5px 14px;
            border-radius:6px;
            font-size:0.82rem;
            font-weight:700;
            cursor:pointer;
        ">{t['cookie_accept_btn']}</button>
    </div>
</div>
"""

cookie_script = """
<script>
(function() {
    function initCookieNotice() {
        try {
            const consent = localStorage.getItem('fraudsentinel_cookie_consent');
            const urlParams = new URLSearchParams(window.location.search);
            const hasUrlParam = urlParams.get('cookies') === '1';
            
            const banner = document.getElementById('fraudsentinel-cookie-corner');
            if (!banner) return;
            
            if (!consent && !hasUrlParam) {
                banner.style.display = 'block';
            }
        } catch(e) {}
    }
    
    window.closeCookieNotice = function(accepted) {
        const banner = document.getElementById('fraudsentinel-cookie-corner');
        if (banner) banner.style.display = 'none';
        
        try {
            if (accepted) {
                localStorage.setItem('fraudsentinel_cookie_consent', 'accepted');
                const url = new URL(window.location);
                url.searchParams.set('cookies', '1');
                window.history.replaceState({}, '', url);
            } else {
                localStorage.setItem('fraudsentinel_cookie_consent', 'declined');
            }
        } catch(e) {}
    };
    
    setTimeout(initCookieNotice, 400);
})();
</script>
"""

st.markdown(cookie_html + cookie_script, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (Cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Model Weights...")
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, DEFAULT_FEATURES

    model = joblib.load(MODEL_PATH)
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

    raw_pred = model.predict(X_scaled)[0]
    raw_score = model.decision_function(X_scaled)[0]
    label = "FRAUD" if raw_pred == -1 else "SAFE"

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
                "bar": {"color": color, "thickness": 0.85},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": current_theme['border_color'],
                "steps": [
                    {"range": [0, 35], "color": "rgba(22, 163, 74, 0.15)"},
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


def create_radar_chart(values: dict, texts: dict) -> go.Figure:
    categories = [
        texts.get('radar_category_value', 'Value (Norm)'),
        texts.get('radar_category_hour', 'Hour (Norm)'),
        texts.get('radar_category_geo', 'Geo Distance'),
        texts.get('radar_category_age', 'Age (Norm)'),
        texts.get('radar_category_pop', 'Population Density'),
    ]
    
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
        name=texts.get('radar_baseline', 'Baseline Safe'),
        line_color=current_theme['accent_green'],
        fillcolor='rgba(22, 163, 74, 0.15)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=val_vector,
        theta=categories,
        fill='toself',
        name=texts.get('radar_current', 'Current Transaction'),
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


def explain_prediction(values: dict, label: str, risk_pct: float, texts: dict) -> str:
    reasons = []
    if values.get("amt", 0) > 500:
        reasons.append(texts["reason_val"].format(val=values['amt']))
    if values.get("hour", 12) in range(0, 5):
        reasons.append(texts["reason_time"].format(h=values['hour']))
    if values.get("city_pop", 100_000) < 5_000:
        reasons.append(texts["reason_pop"].format(pop=int(values['city_pop'])))
    if abs(values.get("lat", 0) - values.get("merch_lat", 0)) > 2:
        reasons.append(texts["reason_geo"])
    if values.get("day_of_week", 1) in [5, 6]:
        reasons.append(texts["reason_weekend"])

    if not reasons:
        reasons.append(texts["reason_ok"])

    header = f"### {texts['diagnostics']}\n\n"
    return header + "\n\n".join(f"- {r}" for r in reasons)


def batch_predict(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"] = df["trans_date_trans_time"].dt.hour.fillna(12)
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

    raw_preds = model.predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)

    df["prediction"] = np.where(raw_preds == -1, "FRAUD", "SAFE")
    df["risk_score"] = np.clip(((-raw_scores + 0.3) / 0.6) * 100, 0, 100).round(1)
    df["anomaly_raw"] = raw_scores.round(4)
    return df


def generate_sample_csv() -> bytes:
    rng = np.random.default_rng(42)
    n = 30
    data = {
        "amt": rng.exponential(80, n).round(2),
        "lat": rng.uniform(25, 48, n).round(4),
        "long": rng.uniform(-122, -70, n).round(4),
        "city_pop": rng.integers(500, 2_000_000, n),
        "merch_lat": rng.uniform(25, 48, n).round(4),
        "merch_long": rng.uniform(-122, -70, n).round(4),
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "age": rng.integers(18, 75, n),
        "amt_log": np.log1p(rng.exponential(80, n)).round(4),
    }
    data["amt"][-3:] = [4500.0, 9999.99, 3200.0]
    data["hour"][-3:] = [2, 3, 4]
    data["city_pop"][-3:] = [120, 80, 200]
    data["amt_log"][-3:] = np.log1p(data["amt"][-3:]).round(4)
    return pd.DataFrame(data).to_csv(index=False).encode()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROL PANEL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='font-family:Orbitron; color:{current_theme['accent_primary']}; margin-bottom:0;'>{t['sidebar_title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<span class='cyber-tag tag-cyan'>{t['sidebar_tag']}</span>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander(t["sys_config"], expanded=True):
        # 1. Language selector (PT and EN strictly)
        lang_idx = ["PT", "EN"].index(st.session_state["selected_lang"]) if st.session_state["selected_lang"] in ["PT", "EN"] else 0
        lang_choice = st.selectbox(
            t["lang_label"],
            options=["PT", "EN"],
            format_func=lambda x: "Português (PT-BR)" if x == "PT" else "English (EN-US)",
            index=lang_idx,
            key="widget_lang"
        )
        if lang_choice != st.session_state["selected_lang"]:
            st.session_state["selected_lang"] = lang_choice
            if st.session_state.get("cookies_enabled", False):
                st.query_params["lang"] = lang_choice
            st.rerun()

        # 2. Color Theme Selector
        theme_options = list(THEME_CONFIGS.keys())
        theme_idx = theme_options.index(st.session_state["theme"]) if st.session_state["theme"] in theme_options else 0
        theme_choice = st.selectbox(
            t["theme_label"],
            options=theme_options,
            index=theme_idx,
            key="widget_theme"
        )
        if theme_choice != st.session_state["theme"]:
            st.session_state["theme"] = theme_choice
            if st.session_state.get("cookies_enabled", False):
                st.query_params["theme"] = theme_choice
            st.rerun()

        # 3. Font Size Slider
        font_options = list(FONT_SCALES.keys())
        font_idx = font_options.index(st.session_state["font_size"]) if st.session_state["font_size"] in font_options else 0
        font_choice = st.select_slider(
            t["font_label"],
            options=font_options,
            value=font_options[font_idx],
            key="widget_font"
        )
        if font_choice != st.session_state["font_size"]:
            st.session_state["font_size"] = font_choice
            if st.session_state.get("cookies_enabled", False):
                st.query_params["font"] = font_choice
            st.rerun()

        if st.button(t["clear_all_cookies"], use_container_width=True):
            st.session_state["theme"] = "Dark Cyber"
            st.session_state["font_size"] = "Normal"
            st.session_state["selected_lang"] = "PT"
            st.session_state["cookies_enabled"] = False
            st.session_state["history"] = []
            clear_simulation_history()
            st.query_params.clear()
            st.rerun()

    st.markdown("---")
    model_ok = model is not None and scaler is not None
    if model_ok:
        st.success(f"{t['status_ready']}")
    else:
        st.error(f"{t['status_offline']}")

    st.markdown("---")
    st.markdown(f"### {t['arch_overview']}")
    st.markdown(t['arch_desc'])
    st.markdown("---")

    hist_df = pd.DataFrame(st.session_state["history"])
    if not hist_df.empty:
        total = len(hist_df)
        frauds = (hist_df["verdict"] == "FRAUD").sum()
        st.markdown(f"### {t['session_telemetry']}")
        c1, c2 = st.columns(2)
        c1.metric(t['analyzed'], total)
        c2.metric(t['flagged'], frauds)
        if st.button(t['clear_telemetry'], use_container_width=True):
            st.session_state["history"] = []
            clear_simulation_history()
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER BANNER
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

# Clean Tab headers
tab1, tab2, tab3, tab4 = st.tabs(
    [
        f"◈ [01] {t['tab_simulate']}", 
        f"◈ [02] {t['tab_batch']}", 
        f"◈ [03] {t['tab_insights']}", 
        f"◈ [04] {t['tab_about']}"
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
            st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>{t['payload_details']}</h4>", unsafe_allow_html=True)
            amt = st.number_input(t['amt'], min_value=0.01, max_value=25_000.0, value=85.50, step=0.01, format="%.2f")
            hour = st.slider(t['hour'], 0, 23, 14)
            day_of_week = st.selectbox(
                t['day'],
                options=list(range(7)),
                format_func=lambda x: t['days'][x],
                index=1,
            )

        with c2:
            st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>{t['node_coords']}</h4>", unsafe_allow_html=True)
            lat = st.number_input(t['card_lat'], value=40.7128, format="%.4f")
            long = st.number_input(t['card_long'], value=-74.0060, format="%.4f")
            age = st.slider(t['age'], 18, 90, 35)
            city_pop = st.number_input(t['city_pop'], min_value=100, max_value=5_000_000, value=250_000, step=1000)

        with c3:
            st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>{t['merchant_target']}</h4>", unsafe_allow_html=True)
            merch_lat = st.number_input(t['merch_lat'], value=40.7306, format="%.4f")
            merch_long = st.number_input(t['merch_long'], value=-73.9352, format="%.4f")

        submitted = st.form_submit_button(t['btn_run'], use_container_width=True, type="primary")

    if submitted:
        amt_log = float(np.log1p(amt))
        values = dict(
            amt=amt, lat=lat, long=long, city_pop=city_pop,
            merch_lat=merch_lat, merch_long=merch_long,
            hour=hour, day_of_week=day_of_week, age=age, amt_log=amt_log,
        )

        with st.spinner(t.get('spinner_simulate', 'Processing isolation tree path length...')):
            time.sleep(0.1)
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
        
        sim_chart_tab1, sim_chart_tab2 = st.tabs([
            f"⬡ {t.get('sim_tabs_diagnostics', 'Diagnostics')}",
            f"▲ {t.get('sim_tabs_radar', 'Radar Profile')}"
        ])

        with sim_chart_tab1:
            col_gauge, col_explain = st.columns([1, 1])
            with col_gauge:
                st.plotly_chart(score_gauge(risk_pct), use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center;color:{current_theme['subtext_color']};font-family:JetBrains Mono;font-size:0.9rem;font-weight:600;'>"
                    f"{t['raw_score']}: <b style='color:{current_theme['accent_primary']}'>{raw_score:.4f}</b></p>",
                    unsafe_allow_html=True,
                )

            with col_explain:
                explanation = explain_prediction(values, label, risk_pct, t)
                st.markdown(explanation)

        with sim_chart_tab2:
            st.markdown(f"#### {t.get('radar_title', 'Radar Chart')}")
            col_radar, col_bar = st.columns([1.2, 1])
            with col_radar:
                st.plotly_chart(create_radar_chart(values, t), use_container_width=True)
            with col_bar:
                features_names = [
                    t.get('radar_category_value', 'Value'),
                    t.get('radar_category_hour', 'Hour'),
                    t.get('radar_category_age', 'Age'),
                    t.get('radar_category_geo', 'Geo Dist'),
                    t.get('radar_category_pop', 'Pop Density')
                ]
                dist_geo = abs(values.get('lat', 0) - values.get('merch_lat', 0)) * 100
                variances = [values.get('amt', 0), values.get('hour', 0) * 10, values.get('age', 0), dist_geo, values.get('city_pop', 0) / 5000]
                
                fig_var = px.bar(
                    x=features_names, y=variances,
                    title=t.get('payload_bar_title', 'Payload Attribute Magnitude'),
                    labels={'x': t.get('payload_bar_x', 'Attribute'), 'y': t.get('payload_bar_y', 'Magnitude')},
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

        now_local = datetime.now().astimezone()
        tz_offset = now_local.strftime("%z")
        tz_str = f"UTC{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) >= 5 else "UTC-03:00"
        req_time = now_local.strftime(f"%d/%m/%Y %H:%M:%S ({tz_str})")
        sim_window = f"{hour:02d}:00:00 — {t['days'][day_of_week]}"

        new_record = {
            "req_time": req_time,
            "sim_window": sim_window,
            "amt": f"${amt:.2f}",
            "age": int(age),
            "verdict": label,
            "risk_pct": f"{risk_pct:.1f}%",
            "raw_score": f"{raw_score:.4f}",
            "city_pop": int(city_pop),
        }
        st.session_state["history"].insert(0, new_record)
        save_simulation_history(st.session_state["history"])

    if st.session_state.get("history"):
        st.markdown("---")
        total_logs = len(st.session_state["history"])
        st.markdown(f"### ◈ {t['session_log']} ({total_logs})")
        
        display_hist = []
        for row in st.session_state["history"]:
            display_hist.append({
                t['log_col_time']: row.get("req_time") or row.get("timestamp") or row.get(t['log_col_time'], ""),
                t['log_col_sim']: row.get("sim_window") or row.get(t['log_col_sim'], ""),
                t['log_col_amt']: row.get("amt") or row.get("amount") or row.get(t['log_col_amt'], ""),
                t['log_col_age']: row.get("age") or row.get(t['log_col_age'], ""),
                t['log_col_verdict']: row.get("verdict") or row.get(t['log_col_verdict'], ""),
                t['log_col_risk']: row.get("risk_pct") or row.get("risk_score") or row.get(t['log_col_risk'], ""),
            })
        hist_df = pd.DataFrame(display_hist)
        
        def highlight_verdict(val):
            if val == "FRAUD":
                return f"background-color: rgba(220, 38, 38, 0.2); color: {current_theme['accent_red']}; font-weight: bold;"
            return f"background-color: rgba(22, 163, 74, 0.15); color: {current_theme['accent_green']}; font-weight: bold;"
        
        styled_hist = hist_df.style.map(highlight_verdict, subset=[t['log_col_verdict']])
        st.dataframe(styled_hist, use_container_width=True, hide_index=True)

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            st.download_button(
                f"⤓ {t['btn_download_results']}",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="simulation_audit_log.csv",
                mime="text/csv",
                key="btn_download_sim_history"
            )
        with col_clr:
            if st.button(f"⟲ {t['clear_telemetry']}", use_container_width=True, key="btn_clear_tab1_history"):
                st.session_state["history"] = []
                clear_simulation_history()
                st.rerun()

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
            f"⤓ {t['btn_download_sample']}",
            data=generate_sample_csv(),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )
        use_sample = st.button(t['btn_test_sample'], use_container_width=True)

    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(t['status_payload_loaded'].format(n=len(df_batch)))
        except Exception as e:
            st.error(t['status_csv_error'].format(error=e))

    elif use_sample:
        df_batch = pd.read_csv(io.BytesIO(generate_sample_csv()))
        st.info(t['status_sample_active'])

    if df_batch is not None:
        with st.spinner(t.get('spinner_batch', 'Processing isolation forest inference...')):
            result_df = batch_predict(df_batch)

        total_tx = len(result_df)
        fraud_tx = (result_df["prediction"] == "FRAUD").sum()
        safe_tx = total_tx - fraud_tx
        avg_risk = result_df["risk_score"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t['total_executed'], f"{total_tx:,}")
        m2.metric(t['flagged_anomaly'], f"{fraud_tx:,}", delta=f"{fraud_tx/total_tx:.2%}")
        m3.metric(t['verified_safe'], f"{safe_tx:,}")
        m4.metric(t['avg_risk'], f"{avg_risk:.1f}%")

        st.markdown("---")
        st.markdown(f"### {t.get('analytics_title', 'Advanced Graph Analysis Panel')}")

        b_tab1, b_tab2, b_tab3, b_tab4 = st.tabs([
            f"⬡ {t.get('overview_tab', 'Overall Distribution')}",
            f"◈ {t.get('temporal_tab', 'Temporal Patterns')}",
            f"▲ {t.get('scatter_tab', 'Value vs Risk Relationship')}",
            f"◆ {t.get('geography_tab', 'Geography & Demographics')}"
        ])

        with b_tab1:
            col_pie, col_hist = st.columns(2)
            with col_pie:
                pie_data = result_df["prediction"].value_counts().reset_index()
                pie_data.columns = [t.get('chart_prediction_label', 'Prediction'), t.get('chart_count_label', 'Count')]
                fig_pie = px.pie(
                    pie_data, names=t.get('chart_prediction_label', 'Prediction'), values=t.get('chart_count_label', 'Count'),
                    color=t.get('chart_prediction_label', 'Prediction'),
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
                    labels={"risk_score": t.get('chart_risk_label', 'Risk Percentage (%)'), "count": t.get('chart_frequency_label', 'Frequency')},
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
                        title=t.get('hour_title', 'Transactions by Hour of Day (0-23h)'),
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
                        title=t.get('day_title', 'Transactions by Day of Week (0=Mon, 6=Sun)'),
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
                    title=t.get('scatter_title', 'Scatter: Transaction Value ($) vs Risk Score (%)'),
                    labels={"amt": t.get('chart_amount_label', 'Transaction Amount ($)'), "risk_score": t.get('chart_risk_label', 'Risk Percentage (%)')},
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
                        title=t.get('pop_title', 'City Population Distribution by Status'),
                        labels={"city_pop": t.get('chart_population_label', 'Population')},
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
                        title=t.get('age_title', 'Distribution by Cardholder Age Range'),
                        labels={"age": t.get('chart_age_label', 'Age')},
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
        st.markdown(f"### {t['grid_title']}")

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
            file_name="fraudsentinel_predictions.csv",
            mime="text/csv",
            type="primary"
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### {t.get('model_spec_title', 'Model Parameters & Engine Architecture')}")

    col_params, col_info = st.columns([1, 2])
    with col_params:
        st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>{t['model_params']}</h4>", unsafe_allow_html=True)
        params = model.get_params()
        param_df = pd.DataFrame(list(params.items()), columns=["Hyperparameter", "Value"])

        def normalize_param_value(value):
            if value is None:
                return ""
            if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple, set)):
                try:
                    value = value.item()
                except Exception:
                    pass
            if isinstance(value, (str, int, float, bool)):
                return value
            return str(value)

        param_df["Value"] = param_df["Value"].apply(normalize_param_value)
        st.dataframe(param_df.astype({"Hyperparameter": "string", "Value": "string"}), use_container_width=True, hide_index=True)

    with col_info:
        st.markdown(f"<h4 style='color:{current_theme['accent_primary']}; font-family:Orbitron;'>{t['theory_title']}</h4>", unsafe_allow_html=True)
        st.markdown(t['theory_desc'])

    st.markdown("---")
    st.markdown(f"### {t.get('monte_carlo_title', 'Monte Carlo Visualizer')}")
    if st.button(t['btn_monte_carlo'], use_container_width=False):
        rng = np.random.default_rng()
        n = 50
        rand_data = {
            "amt": rng.exponential(100, n),
            "lat": rng.uniform(25, 48, n),
            "long": rng.uniform(-122, -70, n),
            "city_pop": rng.integers(100, 2_000_000, n).astype(float),
            "merch_lat": rng.uniform(25, 48, n),
            "merch_long": rng.uniform(-122, -70, n),
            "hour": rng.integers(0, 24, n).astype(float),
            "day_of_week": rng.integers(0, 7, n).astype(float),
            "age": rng.integers(18, 80, n).astype(float),
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
            title=t.get('monte_carlo_scatter_title', 'Amount vs Risk Score Distribution'),
            labels={"amt": t.get('chart_amount_label', 'Transaction Amount ($)'), "risk_score": t.get('chart_risk_label', 'Risk Percentage (%)')},
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
    st.markdown(f"### {t['arch_overview']}")
    st.markdown(
        f"""
## {t['about_title']}

{t['about_intro']}

---

### {t['about_stack_title']}

- **{t['about_ml_core']}**: `scikit-learn` Isolation Forest (Unsupervised Tree Anomaly Detection)
- **{t['about_data_eng']}**: `pandas`, `numpy`, `StandardScaler`, Feature Extraction (Haversine/Geodesic, Log Transforms, Circadian Encodings)
- **{t['about_dashboard']}**: `Streamlit`, `Plotly Express` (Dynamic Theme & i18n Engine)
- **{t['about_dataset']}**: Kaggle Credit Card Fraud Detection dataset (**1,296,675 transactions**)
        """
    )
