import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.styles import load_css
from streamlit_option_menu import option_menu

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# ── Estado inicial ───────────────────────────────────────────────────────────
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "last_risk_level" not in st.session_state:
    st.session_state.last_risk_level = "Bajo"

if "last_risk_pct" not in st.session_state:
    st.session_state.last_risk_pct = 10

# ── Mapeos para encoding  ─────────────────────────
PROTOCOL_MAP = {"icmp": 0, "tcp": 1, "udp": 2}
 
SERVICE_MAP = {
    "ftp_data": 0, "other": 1, "private": 2, "http": 3,
    "remote_job": 4, "name": 5, "netbios_ns": 6, "eco_i": 7,
    "mtp": 8, "telnet": 9, "finger": 10, "domain_u": 11,
    "supdup": 12, "uucp_path": 13, "Z39_50": 14, "smtp": 15,
    "csnet_ns": 16, "uucp": 17, "netbios_dgm": 18, "urp_i": 19,
    "auth": 20, "domain": 21, "ftp": 22, "bgp": 23, "ldap": 24,
    "ecr_i": 25, "gopher": 26, "vmnet": 27, "systat": 28,
    "http_443": 29, "efs": 30, "whois": 31, "imap4": 32,
    "iso_tsap": 33, "echo": 34, "klogin": 35, "link": 36,
    "sunrpc": 37, "login": 38, "kshell": 39, "sql_net": 40,
    "time": 41, "hostnames": 42, "exec": 43, "ntp_u": 44,
    "discard": 45, "nntp": 46, "courier": 47, "ctf": 48,
    "ssh": 49, "daytime": 50, "shell": 51, "netstat": 52,
    "pop_3": 53, "nnsp": 54, "IRC": 55, "pop_2": 56,
    "printer": 57, "tim_i": 58, "pm_dump": 59, "red_i": 60,
    "netbios_ssn": 61, "rje": 62, "X11": 63, "urh_i": 64,
    "http_8001": 65, "aol": 66, "http_2784": 67, "tftp_u": 68,
    "harvest": 69
}
 
FLAG_MAP = {
    "OTH": 0, "REJ": 1, "RSTO": 2, "RSTOS0": 3,
    "RSTR": 4, "S0": 5, "S1": 6, "S2": 7,
    "S3": 8, "SF": 9, "SH": 10
}
 
LOGGED_IN_MAP = {"No (0)": 0, "Sí (1)": 1}
 

# ── Rutas absolutas basadas en la ubicación del script ──────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH      = os.path.join(BASE_DIR, "model", "anomaly_model.pkl")
DATA_PATH       = os.path.join(BASE_DIR, "data",  "processed_data.csv")
RAW_DATA_PATH   = os.path.join(BASE_DIR, "data",  "user_behavior.csv")

# ── Carga del modelo y datos ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
data  = load_data()

st.sidebar.caption(f"📂 Dataset: {len(data):,} registros")
st.sidebar.caption(f"Columnas: {list(data.columns)}")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🔐 Sentinel AI</div>", unsafe_allow_html=True)
    st.caption("Sistema de monitoreo con IA")
    st.markdown("---")

    menu = option_menu(
        None,
        ["Inicio", "Análisis", "Dashboard", "Acerca de"],
        icons=["house", "shield-lock", "bar-chart", "info-circle"],
        default_index=0,
        styles={
            "container":     {"background-color": "transparent"},
            "icon":          {"color": "#00D4FF", "font-size": "14px"},
            "nav-link":      {"font-size": "14px", "color": "#94A3B8"},
            "nav-link-selected": {"background-color": "#00D4FF15", "color": "#00D4FF", "font-weight": "500"},
        }
    )

    st.markdown("---")
    anomaly_count = data["anomaly"].value_counts().get(-1, 0)
    total         = len(data)
    risk_pct      = round((anomaly_count / total) * 100, 1) if total > 0 else 0

    if risk_pct > 15:
        st.error(f"⚠️ Riesgo global: {risk_pct}%")
    elif risk_pct > 8:
        st.warning(f"🟡 Riesgo global: {risk_pct}%")
    else:
        st.success(f"🟢 Riesgo global: {risk_pct}%")

    st.markdown("<div class='footer'>Sentinel AI © 2026</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# INICIO
# ════════════════════════════════════════════════════════════════════════════
if menu == "Inicio":

    st.markdown("<div class='section-title'>🔐 Sentinel AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Sistema inteligente de detección de comportamiento anómalo en redes</div>", unsafe_allow_html=True)

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estado del Sistema", "Activo",      delta="🟢 Online")
    col2.metric("Modelo IA",          "Isolation Forest", delta="🧠 Operativo")
    col3.metric("Registros analizados", f"{total:,}", delta=f"+{len(data[data['anomaly']==-1])} anomalías")
    col4.metric("Nivel de Riesgo",    f"{risk_pct}%", delta="Monitoreo continuo")

    st.divider()

    # Cards de características
    st.subheader("¿Qué hace Sentinel AI?")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>🧠 Detección con IA</h4>
            <p style='color:#64748B; font-size:14px; margin-top:8px;'>
            Usa Isolation Forest para aprender patrones normales y detectar comportamientos
            que se desvían estadísticamente del baseline.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>📊 Dashboard en tiempo real</h4>
            <p style='color:#64748B; font-size:14px; margin-top:8px;'>
            Visualizaciones interactivas de patrones de acceso, distribución de anomalías
            y métricas de riesgo actualizadas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='info-card'>
            <h4>⚠️ Análisis de comportamiento</h4>
            <p style='color:#64748B; font-size:14px; margin-top:8px;'>
            Evalúa hora de acceso, intentos fallidos, duración de sesión, dispositivo
            y ubicación para calcular un score de riesgo.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Cómo funciona — paso a paso
    st.subheader("Arquitectura del sistema")
    steps = [
        ("1", "Recopilación de datos", "Se capturan features del comportamiento: hora, intentos, sesión, dispositivo, ubicación."),
        ("2", "Preprocesamiento",       "Encoding de variables categóricas con LabelEncoder. Normalización del espacio de features."),
        ("3", "Isolation Forest",       "El modelo aísla muestras anómalas partiendo el espacio con árboles de decisión aleatorios."),
        ("4", "Score de anomalía",      "Cada muestra recibe un score continuo. Valores más negativos = más anómalos."),
        ("5", "Visualización",          "Los resultados se despliegan en dashboard interactivo con alertas y métricas de riesgo."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex; gap:16px; align-items:flex-start; margin-bottom:16px;'>
            <div style='background:#00D4FF15; border:1px solid #00D4FF40; color:#00D4FF;
                        border-radius:50%; width:32px; height:32px; display:flex;
                        align-items:center; justify-content:center; font-weight:700;
                        font-size:13px; flex-shrink:0;'>{num}</div>
            <div>
                <div style='font-weight:600; color:#E2E8F0; font-size:14px;'>{title}</div>
                <div style='color:#64748B; font-size:13px; margin-top:2px;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ANÁLISIS
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Análisis":
 
    st.markdown("<div class='section-title'>🔍 Análisis de Tráfico de Red</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Ingresa las características de una conexión para evaluar si es anómala</div>", unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("**Características de la conexión**")
        duration         = st.slider("Duración de la conexión (seg)", 0, 60000, 0)
        src_bytes        = st.slider("Bytes enviados por el origen", 0, 100000, 200)
        num_failed_logins = st.slider("Intentos fallidos de login", 0, 10, 0)
        count            = st.slider("Conexiones al mismo host (últimos 2 seg)", 0, 512, 5)
 
    with col2:
        st.markdown("**Contexto del protocolo**")
        protocol_type = st.selectbox("Protocolo de red", ["tcp", "udp", "icmp"])
        service       = st.selectbox("Servicio de destino", ["http", "ftp", "smtp", "ssh", "telnet", "private", "other"])
        flag          = st.selectbox("Estado de la conexión", ["SF", "REJ", "S0", "RSTO", "RSTR", "S1", "OTH"])
        logged_in     = st.selectbox("¿Sesión iniciada exitosamente?", ["Sí (1)", "No (0)"])
 
    # Expander con explicación de cada campo
    with st.expander("ℹ️ ¿Qué significa cada campo?"):
        st.markdown("""
        | Campo | Significado |
        |-------|-------------|
        | **Duración** | Cuántos segundos duró la conexión |
        | **Bytes enviados** | Cantidad de datos transferidos desde el origen |
        | **Intentos fallidos** | Cuántas veces falló el login antes de entrar |
        | **Conexiones al mismo host** | Cuántas conexiones hubo al mismo servidor en 2 segundos |
        | **Protocolo** | TCP (web/correo), UDP (streaming), ICMP (ping/diagnóstico) |
        | **Servicio** | A qué servicio se conectó (http, ftp, ssh...) |
        | **Flag** | SF = conexión completada normal. REJ = rechazada. S0 = sin respuesta |
        | **Sesión iniciada** | Si el usuario logró autenticarse exitosamente |
        """)
 
    st.markdown("---")
 
    if st.button("🔍 Analizar conexión", use_container_width=True):
 
        # ── Construir input con encoding correcto ────────────────────────────
        service_encoded = SERVICE_MAP.get(service, SERVICE_MAP.get("other", 1))
 
        input_data = [[
            duration,
            PROTOCOL_MAP[protocol_type],
            service_encoded,
            FLAG_MAP[flag],
            src_bytes,
            num_failed_logins,
            count,
            LOGGED_IN_MAP[logged_in]
        ]]
 
        prediction    = model.predict(input_data)
        anomaly_score = model.decision_function(input_data)[0]
 
        # ── Calcular riesgo ──────────────────────────────────────────────────
        risk_score = 0
        reasons    = []
 
        if num_failed_logins > 2:
            risk_score += 30
            reasons.append(("🔴", "Múltiples intentos fallidos de login", f"{num_failed_logins} intentos — umbral normal: ≤2"))
 
        if count > 100:
            risk_score += 25
            reasons.append(("🔴", "Posible ataque DoS — conexiones masivas", f"{count} conexiones en 2 seg — umbral normal: ≤100"))
 
        if flag in ["REJ", "S0", "RSTO"]:
            risk_score += 20
            reasons.append(("🟡", f"Flag sospechoso: {flag}", "SF = normal. REJ/S0/RSTO indican conexiones anómalas"))
 
        if protocol_type == "icmp" and src_bytes == 0 and count > 50:
            risk_score += 20
            reasons.append(("🔴", "Patrón de ping flood (ICMP)", "Muchos pings sin datos = posible ataque de reconocimiento"))
 
        if src_bytes > 50000 and logged_in == "No (0)":
            risk_score += 15
            reasons.append(("🟡", "Transferencia masiva sin autenticación", f"{src_bytes:,} bytes sin sesión activa"))
 
        if prediction[0] == -1:
            risk_score = max(risk_score, 70)
 
        risk_score = min(risk_score, 100)
 
        if risk_score >= 60:
            risk_level = "Alto"
        elif risk_score >= 30:
            risk_level = "Medio"
        else:
            risk_level = "Bajo"
 
        st.session_state.last_risk_level = risk_level
        st.session_state.last_risk_pct   = risk_score
 
        # ── Resultado principal ──────────────────────────────────────────────
        if prediction[0] == -1:
            st.markdown("""
            <div class='anomaly-alert'>
                <div style='font-size:18px; font-weight:700; color:#FF4D4D;'>
                    ⚠️ Conexión anómala detectada
                </div>
                <div style='color:#94A3B8; font-size:13px; margin-top:4px;'>
                    Isolation Forest clasificó esta conexión fuera del patrón normal del tráfico de red
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='safe-alert'>
                <div style='font-size:18px; font-weight:700; color:#00D4FF;'>
                    ✅ Tráfico dentro del rango normal
                </div>
                <div style='color:#94A3B8; font-size:13px; margin-top:4px;'>
                    El modelo no detectó desviaciones significativas del baseline aprendido
                </div>
            </div>
            """, unsafe_allow_html=True)
 
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clasificación",   "Anómalo" if prediction[0] == -1 else "Normal")
        col2.metric("Nivel de Riesgo", risk_level)
        col3.metric("Score de Riesgo", f"{risk_score}%")
        col4.metric("Anomaly Score",   f"{anomaly_score:.4f}",
                    help="Negativo = anómalo. Más negativo = más sospechoso. Umbral = 0")
 
        # Barra de riesgo
        bar_color = "#FF4D4D" if risk_level == "Alto" else ("#FFB300" if risk_level == "Medio" else "#00D4FF")
        st.markdown(f"""
        <div style='margin:20px 0 6px; font-size:13px; color:#64748B;'>Score de riesgo acumulado</div>
        <div style='background:#1E2A3A; border-radius:8px; height:10px; overflow:hidden;'>
            <div style='background:{bar_color}; width:{risk_score}%; height:100%; border-radius:8px;'></div>
        </div>
        <div style='display:flex; justify-content:space-between; margin-top:4px;'>
            <span style='font-size:11px; color:#334155;'>0% — Sin riesgo</span>
            <span style='font-size:12px; color:{bar_color}; font-weight:600;'>{risk_score}%</span>
            <span style='font-size:11px; color:#334155;'>100% — Crítico</span>
        </div>
        """, unsafe_allow_html=True)
 
        st.divider()
 
        # ── Radar chart vs promedio histórico ────────────────────────────────
        hist = load_data()
 
        avg_duration  = hist["duration"].mean()          / max(hist["duration"].max(), 1)
        avg_src_bytes = hist["src_bytes"].mean()          / max(hist["src_bytes"].max(), 1)
        avg_failed    = hist["num_failed_logins"].mean()  / 10
        avg_count     = hist["count"].mean()              / 512

        cur_duration  = min(duration          / max(hist["duration"].max(), 1), 1)
        cur_src_bytes = min(src_bytes         / max(hist["src_bytes"].max(), 1), 1)
        cur_failed    = min(num_failed_logins / 10, 1)
        cur_count     = min(count             / 512, 1)
 
        categorias = ["Duración", "Bytes enviados", "Intentos fallidos", "Conexiones"]
 
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[avg_duration, avg_src_bytes, avg_failed, avg_count, avg_duration],
            theta=categorias + [categorias[0]],
            fill="toself", name="Promedio histórico",
            line=dict(color="#00D4FF", width=2),
            fillcolor="rgba(0,212,255,0.1)"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[cur_duration, cur_src_bytes, cur_failed, cur_count, cur_duration],
            theta=categorias + [categorias[0]],
            fill="toself", name="Conexión analizada",
            line=dict(color="#FF4D4D" if prediction[0] == -1 else "#FFB300", width=2),
            fillcolor="rgba(255,77,77,0.15)" if prediction[0] == -1 else "rgba(255,179,0,0.15)"
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#1E2A3A", tickfont=dict(color="#334155")),
                angularaxis=dict(gridcolor="#1E2A3A", tickfont=dict(color="#94A3B8"))
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94A3B8", showlegend=True,
            legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=40, l=60, r=60)
        )
 
        col_radar, col_reasons = st.columns([1, 1])
 
        with col_radar:
            st.subheader("Comparación vs. tráfico histórico")
            st.caption("Azul = promedio del dataset NSL-KDD. Rojo/amarillo = conexión analizada.")
            st.plotly_chart(fig_radar, use_container_width=True)
 
        with col_reasons:
            st.subheader("Factores de riesgo detectados")
            if reasons:
                for emoji, titulo, detalle in reasons:
                    st.markdown(f"""
                    <div style='background:#111827; border:1px solid #1E2A3A;
                        border-left:3px solid {"#FF4D4D" if emoji == "🔴" else "#FFB300"};
                        border-radius:8px; padding:12px 14px; margin-bottom:10px;'>
                        <div style='font-size:13px; font-weight:600; color:#E2E8F0;'>{emoji} {titulo}</div>
                        <div style='font-size:12px; color:#64748B; margin-top:3px;'>{detalle}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#111827; border:1px solid #1E2A3A;
                    border-left:3px solid #00D4FF; border-radius:8px; padding:12px 14px;'>
                    <div style='font-size:13px; font-weight:600; color:#00D4FF;'>✅ Sin factores de riesgo</div>
                    <div style='font-size:12px; color:#64748B; margin-top:3px;'>
                        Todos los parámetros dentro del rango normal del tráfico histórico.
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
            st.markdown(f"""
            <div style='background:#0D1117; border:1px solid #1E2A3A; border-radius:8px;
                padding:12px 14px; margin-top:12px;'>
                <div style='font-size:11px; color:#334155; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.05em;'>
                    Nota técnica — Isolation Forest
                </div>
                <div style='font-size:12px; color:#64748B; line-height:1.6;'>
                    <code style='color:#00D4FF;'>decision_function()</code> retornó
                    <code style='color:{"#FF4D4D" if anomaly_score < 0 else "#00D4FF"};'>{anomaly_score:.4f}</code>.
                    {"Negativo → el punto se aisló con pocos cortes → región de baja densidad → anomalía." if anomaly_score < 0 else "Positivo → el punto necesitó muchos cortes para aislarse → rodeado de tráfico similar → normal."}
                </div>
            </div>
            """, unsafe_allow_html=True)
 
        # ── Guardar en historial ─────────────────────────────────────────────
        alert = {
            "Duración":     duration,
            "Protocolo":    protocol_type,
            "Servicio":     service,
            "Flag":         flag,
            "Bytes orig.":  src_bytes,
            "Login fails":  num_failed_logins,
            "Conexiones":   count,
            "Logged in":    logged_in,
            "Riesgo":       risk_level,
            "Score":        f"{risk_score}%",
            "Anomaly Score":f"{anomaly_score:.4f}",
            "Modelo":       "Anómalo" if prediction[0] == -1 else "Normal"
        }
        st.session_state.alert_history.append(alert)
 
        # ── Guardar en CSV ───────────────────────────────────────────────────
        new_row = pd.DataFrame([{
            "duration":          duration,
            "protocol_type":     PROTOCOL_MAP[protocol_type],
            "service":           service_encoded,
            "flag":              FLAG_MAP[flag],
            "src_bytes":         src_bytes,
            "num_failed_logins": num_failed_logins,
            "count":             count,
            "logged_in":         LOGGED_IN_MAP[logged_in],
            "anomaly":           prediction[0]
        }])
        updated_data = pd.concat([data, new_row], ignore_index=True)
        updated_data.to_csv(DATA_PATH, index=False)
        load_data.clear()



# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Dashboard":
 
    st.markdown("<div class='section-title'>📊 Dashboard de Seguridad</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Métricas globales del sistema en tiempo real</div>", unsafe_allow_html=True)
 
    # Recargar datos frescos
    fresh_data    = load_data()
    anomaly_count = fresh_data["anomaly"].value_counts().get(-1, 0)
    normal_count  = fresh_data["anomaly"].value_counts().get(1, 0)
    total_records = len(fresh_data)
    risk_pct_dash = round((anomaly_count / total_records) * 100, 1) if total_records > 0 else 0
 
    # ── Métricas globales ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Anomalías",        f"{anomaly_count:,}")
    col2.metric("Conexiones Normales",    f"{normal_count:,}")
    col3.metric("Registros Totales",      f"{total_records:,}")
    col4.metric("Tasa de Anomalía",       f"{risk_pct_dash}%")
 
    st.divider()
 
    # ── Gráfica 1 y 2 ────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
 
    with col_left:
        # PIE: Distribución normal vs anómalo
        st.subheader("Distribución de anomalías")
        fig_dist = px.pie(
            values=[normal_count, anomaly_count],
            names=["Normal", "Anómalo"],
            color_discrete_sequence=["#00D4FF", "#FF4D4D"],
            hole=0.55
        )
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94A3B8",
            legend=dict(font=dict(color="#94A3B8")),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
 
    with col_right:
        # BAR: Intentos fallidos de login
        st.subheader("Intentos fallidos de login")
        failed_data = fresh_data["num_failed_logins"].value_counts().reset_index()
        failed_data.columns = ["Intentos fallidos", "Cantidad"]
        failed_data = failed_data.sort_values("Intentos fallidos")
 
        fig_failed = px.bar(
            failed_data,
            x="Intentos fallidos",
            y="Cantidad",
            color="Cantidad",
            color_continuous_scale=["#00D4FF", "#FFB300", "#FF4D4D"],
        )
        fig_failed.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94A3B8",
            coloraxis_showscale=False,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_failed, use_container_width=True)
 
    # ── Gráfica 3: Accesos por tipo de actividad (normal vs anómalo) ──────────
    # ── Gráfica 3: Conexiones por protocolo (normal vs anómalo) ─────────────
    st.subheader("Conexiones por protocolo de red")
    st.caption("Compara tráfico normal vs anómalo según el protocolo usado.")

    proto_inv = {0: "icmp", 1: "tcp", 2: "udp"}
    fresh_data["protocol_nombre"] = fresh_data["protocol_type"].map(proto_inv).fillna("otro")

    proto_normal  = fresh_data[fresh_data["anomaly"] == 1]["protocol_nombre"].value_counts().reset_index()
    proto_anomaly = fresh_data[fresh_data["anomaly"] == -1]["protocol_nombre"].value_counts().reset_index()
    proto_normal.columns  = ["Protocolo", "Cantidad"]
    proto_anomaly.columns = ["Protocolo", "Cantidad"]

    fig_proto = go.Figure()
    fig_proto.add_trace(go.Bar(x=proto_normal["Protocolo"],  y=proto_normal["Cantidad"],  name="Normal",  marker_color="#00D4FF"))
    fig_proto.add_trace(go.Bar(x=proto_anomaly["Protocolo"], y=proto_anomaly["Cantidad"], name="Anómalo", marker_color="#FF4D4D"))
    fig_proto.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94A3B8",
        xaxis=dict(title="Protocolo", gridcolor="#1E2A3A"),
        yaxis=dict(title="Cantidad",  gridcolor="#1E2A3A"),
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=20, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_proto, use_container_width=True)
 
    # ── Gráfica 4: Scatter — Duración de sesión vs Conteo de accesos ─────────
    # ── Gráfica 4: Scatter — Bytes enviados vs Conexiones ────────────────────
    st.subheader("Mapa de anomalías: Bytes enviados vs. Conexiones al host")
    st.caption("Cada punto es una conexión. Rojo = anómalo detectado por Isolation Forest.")

    plot_data = fresh_data.copy()
    plot_data["src_bytes_clip"] = plot_data["src_bytes"].clip(0, 50000)
    plot_data["count_clip"]     = plot_data["count"].clip(0, 300)

    fig_scatter = px.scatter(
        plot_data,
        x="src_bytes_clip",
        y="count_clip",
        color=plot_data["anomaly"].map({1: "Normal", -1: "Anómalo"}),
        color_discrete_map={"Normal": "rgba(0,212,255,0.25)", "Anómalo": "#FF4D4D"},
        opacity=0.6,
        labels={"src_bytes_clip": "Bytes enviados (máx 50K)", "count_clip": "Conexiones al host (máx 300)", "color": "Clasificación"},
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94A3B8",
        xaxis=dict(gridcolor="#1E2A3A"),
        yaxis=dict(gridcolor="#1E2A3A"),
        margin=dict(t=20, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
 
    st.divider()
 
    # ── Nivel de Riesgo Global ───────────────────────────────────────────────
    st.subheader("Nivel de riesgo global del sistema")
 
    if risk_pct_dash >= 15:
        riesgo_label = "Alto"
        riesgo_color = "#FF4D4D"
    elif risk_pct_dash >= 8:
        riesgo_label = "Medio"
        riesgo_color = "#FFB300"
    else:
        riesgo_label = "Bajo"
        riesgo_color = "#00D4FF"
 
    st.markdown(f"""
    <div style='background:#111827; border:1px solid #1E2A3A; border-radius:12px; padding:20px;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
            <span style='font-size:16px; font-weight:600; color:#E2E8F0;'>Riesgo Global</span>
            <span class='risk-badge risk-{riesgo_label.lower()}'>{riesgo_label}</span>
        </div>
        <div style='background:#1E2A3A; border-radius:8px; height:12px; overflow:hidden;'>
            <div style='background:{riesgo_color}; width:{risk_pct_dash}%; height:100%; border-radius:8px;'></div>
        </div>
        <div style='text-align:right; font-size:13px; color:{riesgo_color}; margin-top:6px;'>
            {risk_pct_dash}% de las conexiones son anómalas
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Historial de alertas ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Historial de análisis de esta sesión")
 
    if st.session_state.alert_history:
        history_df = pd.DataFrame(st.session_state.alert_history)
        st.dataframe(history_df, use_container_width=True)
 
        if st.button("🗑️ Limpiar historial"):
            st.session_state.alert_history = []
            st.rerun()
    else:
        st.info("Aún no hay análisis registrados en esta sesión. Ve a **Análisis** para evaluar una conexión.")
 

 

# ════════════════════════════════════════════════════════════════════════════
# ACERCA DE
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Acerca de":

    st.markdown("<div class='section-title'>Acerca del Proyecto</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Sentinel AI — Sistema de detección de anomalías en redes</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>🎯 Problema que resuelve</h4>
            <p style='color:#94A3B8; font-size:14px; margin-top:8px; line-height:1.7;'>
            Las organizaciones enfrentan el reto de detectar accesos no autorizados en tiempo real.
            Los métodos basados en reglas fijas no se adaptan a comportamientos nuevos. Sentinel AI usa
            Machine Learning no supervisado para aprender patrones legítimos y detectar desviaciones
            automáticamente, sin necesidad de datos etiquetados.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <h4>⚙️ Arquitectura técnica</h4>
            <p style='color:#94A3B8; font-size:14px; margin-top:8px; line-height:1.7;'>
            <strong style='color:#00D4FF;'>Modelo:</strong> Isolation Forest (scikit-learn) — algoritmo no supervisado
            que detecta anomalías aislando muestras con particiones aleatorias en árboles de decisión.<br><br>
            <strong style='color:#00D4FF;'>Features:</strong> hora de acceso, intentos fallidos, cantidad de accesos,
            duración de sesión, ubicación geográfica, tipo de dispositivo, tipo de actividad, día de la semana.<br><br>
            <strong style='color:#00D4FF;'>Interfaz:</strong> Streamlit con visualizaciones Plotly interactivas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>🛠️ Stack tecnológico</h4>
            <div style='display:flex; flex-direction:column; gap:8px; margin-top:12px;'>
        """, unsafe_allow_html=True)

        techs = [
            ("Python 3.x",          "#3776AB"),
            ("Streamlit",           "#FF4B4B"),
            ("scikit-learn",        "#F7931E"),
            ("Pandas / NumPy",      "#150458"),
            ("Plotly",              "#3F4F75"),
            ("Isolation Forest",    "#00D4FF"),
        ]
        for tech, color in techs:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:8px; height:8px; border-radius:50%; background:{color}; flex-shrink:0;'></div>
                <span style='font-size:13px; color:#94A3B8;'>{tech}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card' style='margin-top:12px;'>
            <h4>📁 Estructura del proyecto</h4>
            <pre style='color:#64748B; font-size:12px; margin-top:8px; line-height:1.8;'>
sentinel_ai/
├── app.py
├── model/
│   └── anomaly_model.pkl
├── data/
│   ├── user_behavior.csv
│   └── processed_data.csv
├── utils/
│   └── styles.py
└── notebooks/
    └── data_generator.ipynb
            </pre>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='text-align:center; padding:20px; color:#334155;'>
        <div style='font-size:13px;'>Desarrollado por Pablo Rodriguez M — Inteligencia Artificial Aplicada</div>
        <div style='font-size:12px; margin-top:4px;'> · Sentinel AI v1.0 · 2026</div>
    </div>
    """, unsafe_allow_html=True)