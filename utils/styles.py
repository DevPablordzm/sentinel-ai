import streamlit as st

def load_css():
    st.markdown("""
    <style>
    /* ── Fuente principal ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Fondo principal ── */
    .main {
        background-color: #0A0E17;
        color: #E2E8F0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #1E2A3A;
    }

    /* ── Título del sidebar ── */
    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #00D4FF;
        letter-spacing: -0.5px;
    }

    /* ── Botones principales ── */
    .stButton > button {
        background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
        color: #0A0E17;
        font-weight: 600;
        font-size: 14px;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.2s ease;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.15);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }

    /* ── Métricas ── */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1E2A3A;
        border-radius: 12px;
        padding: 16px;
    }

    /* ── Cards de info ── */
    .info-card {
        background: #111827;
        border: 1px solid #1E2A3A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }

    .info-card:hover {
        border-color: #00D4FF40;
    }

    /* ── Badge de riesgo ── */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .risk-alto   { background: #FF4D4D20; color: #FF4D4D; border: 1px solid #FF4D4D40; }
    .risk-medio  { background: #FFB30020; color: #FFB300; border: 1px solid #FFB30040; }
    .risk-bajo   { background: #00D4FF20; color: #00D4FF; border: 1px solid #00D4FF40; }

    /* ── Título de sección ── */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 24px;
    }

    /* ── Alerta de anomalía ── */
    .anomaly-alert {
        background: linear-gradient(135deg, #FF4D4D08, #FF4D4D15);
        border: 1px solid #FF4D4D40;
        border-left: 3px solid #FF4D4D;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    .safe-alert {
        background: linear-gradient(135deg, #00D4FF08, #00D4FF15);
        border: 1px solid #00D4FF40;
        border-left: 3px solid #00D4FF;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }

    /* ── Divider ── */
    hr {
        border-color: #1E2A3A !important;
    }

    /* ── Footer ── */
    .footer {
        position: fixed;
        bottom: 16px;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 11px;
        color: #334155;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E2A3A;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] > div > div > div > div {
        background-color: #00D4FF !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background-color: #111827;
        border-color: #1E2A3A;
        color: #E2E8F0;
    }
    </style>
    """, unsafe_allow_html=True)
    





