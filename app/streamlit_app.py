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

# ── Mapeos para encoding (igual que en el notebook) ─────────────────────────
LOCATION_MAP   = {"CR": 0, "ES": 1, "MX": 2, "US": 3}
DEVICE_MAP     = {"desktop": 0, "mobile": 1, "tablet": 2}
ACTIVITY_MAP   = {"delete": 0, "download": 1, "login": 2, "upload": 3}
DAY_MAP        = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

# ── Carga del modelo y datos ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model/anomaly_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_data.csv")

model = load_model()
data  = load_data()

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

    st.markdown("<div class='section-title'>🔍 Análisis de comportamiento</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Ingresa los datos de un usuario para evaluar su nivel de riesgo</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Parámetros de sesión**")
        login_hour       = st.slider("Hora de acceso",          0,  23,  12)
        failed_attempts  = st.slider("Intentos fallidos",        0,  10,   0)
        access_count     = st.slider("Cantidad de accesos",      1,  20,   5)
        session_duration = st.slider("Duración de sesión (min)", 1, 120,  30)

    with col2:
        st.markdown("**Contexto del acceso**")
        location     = st.selectbox("Ubicación",   ["CR", "US", "MX", "ES"])
        device       = st.selectbox("Dispositivo", ["desktop", "mobile", "tablet"])
        activity     = st.selectbox("Tipo de actividad", ["login", "download", "upload", "delete"])
        day_of_week  = st.selectbox("Día de la semana",  ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    st.markdown("---")

    if st.button("🔍 Analizar comportamiento", use_container_width=True):

        # ── Construir input con encoding correcto ────────────────────────────
        input_data = [[
            login_hour,
            LOCATION_MAP[location],
            failed_attempts,
            access_count,
            ACTIVITY_MAP[activity],
            DEVICE_MAP[device],
            session_duration,
            DAY_MAP[day_of_week]
        ]]

        prediction   = model.predict(input_data)
        anomaly_score = model.decision_function(input_data)[0]  # Score continuo

        # ── Calcular riesgo ──────────────────────────────────────────────────
        risk_score = 0
        reasons    = []

        if failed_attempts > 3:
            risk_score += 30
            reasons.append(("🔴", "Múltiples intentos fallidos de acceso", f"{failed_attempts} intentos — umbral normal: ≤3"))

        if access_count > 12:
            risk_score += 20
            reasons.append(("🟡", "Cantidad de accesos inusualmente alta", f"{access_count} accesos — umbral normal: ≤12"))

        if login_hour < 6 or login_hour > 22:
            risk_score += 20
            reasons.append(("🟡", "Acceso en horario inusual", f"Hora {login_hour}:00 — horario normal: 6am–10pm"))

        if session_duration > 90:
            risk_score += 15
            reasons.append(("🟡", "Sesión excesivamente larga", f"{session_duration} min — umbral normal: ≤90 min"))

        if activity in ["delete", "download"] and failed_attempts > 1:
            risk_score += 15
            reasons.append(("🔴", "Actividad crítica con intentos fallidos", f"{failed_attempts} intentos previos a acción de alto impacto"))

        if prediction[0] == -1:
            risk_score = max(risk_score, 70)

        risk_score = min(risk_score, 100)

        if risk_score >= 60:
            risk_level = "Alto"
        elif risk_score >= 30:
            risk_level = "Medio"
        else:
            risk_level = "Bajo"

        # Guardar en session state para el Dashboard
        st.session_state.last_risk_level = risk_level
        st.session_state.last_risk_pct   = risk_score

        # ── Mostrar resultado ────────────────────────────────────────────────
        if prediction[0] == -1:
            st.markdown(f"""
            <div class='anomaly-alert'>
                <div style='font-size:18px; font-weight:700; color:#FF4D4D;'>
                    ⚠️ Comportamiento anómalo detectado
                </div>
                <div style='color:#94A3B8; font-size:13px; margin-top:4px;'>
                    El modelo clasificó esta sesión como sospechosa
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='safe-alert'>
                <div style='font-size:18px; font-weight:700; color:#00D4FF;'>
                    ✅ Comportamiento dentro del rango normal
                </div>
                <div style='color:#94A3B8; font-size:13px; margin-top:4px;'>
                    El modelo no detectó patrones anómalos en esta sesión
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Métricas del análisis
        col1, col2, col3 = st.columns(3)
        col1.metric("Nivel de Riesgo",    risk_level)
        col2.metric("Score de Riesgo",    f"{risk_score}%")
        col3.metric("Score del Modelo",   f"{anomaly_score:.4f}")

        # Barra de riesgo con color dinámico
        bar_color = "#FF4D4D" if risk_level == "Alto" else ("#FFB300" if risk_level == "Medio" else "#00D4FF")
        st.markdown(f"""
        <div style='margin:16px 0 8px; font-size:13px; color:#64748B;'>Score de riesgo acumulado</div>
        <div style='background:#1E2A3A; border-radius:8px; height:10px; overflow:hidden;'>
            <div style='background:{bar_color}; width:{risk_score}%; height:100%;
                        border-radius:8px; transition:width 0.5s;'></div>
        </div>
        <div style='text-align:right; font-size:12px; color:{bar_color}; margin-top:4px;'>{risk_score}%</div>
        """, unsafe_allow_html=True)

        
        # ── Comparación vs. promedio histórico (radar chart) ─────────────────
        hist = load_data()
 
        # Promedios del dataset (comportamiento histórico normalizado a escala 0-1)
        avg_login_hour       = hist["login_hour"].mean() / 23
        avg_failed_attempts  = hist["failed_attempts"].mean() / 10
        avg_access_count     = hist["access_count"].mean() / 20
        avg_session_duration = hist["session_duration"].mean() / 120
 
        # Valores actuales normalizados
        cur_login_hour       = login_hour / 23
        cur_failed_attempts  = failed_attempts / 10
        cur_access_count     = access_count / 20
        cur_session_duration = session_duration / 120
 
        categorias = ["Hora de acceso", "Intentos fallidos", "Cant. accesos", "Duración sesión"]
 
        fig_radar = go.Figure()
 
        fig_radar.add_trace(go.Scatterpolar(
            r=[avg_login_hour, avg_failed_attempts, avg_access_count, avg_session_duration, avg_login_hour],
            theta=categorias + [categorias[0]],
            fill="toself",
            name="Promedio histórico",
            line=dict(color="#00D4FF", width=2),
            fillcolor="rgba(0,212,255,0.1)"
        ))
 
        fig_radar.add_trace(go.Scatterpolar(
            r=[cur_login_hour, cur_failed_attempts, cur_access_count, cur_session_duration, cur_login_hour],
            theta=categorias + [categorias[0]],
            fill="toself",
            name="Sesión analizada",
            line=dict(color="#FF4D4D" if prediction[0] == -1 else "#FFB300", width=2),
            fillcolor="rgba(255,77,77,0.15)" if prediction[0] == -1 else "rgba(255,179,0,0.15)"
        ))
 
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1E2A3A", tickfont=dict(color="#334155")),
                angularaxis=dict(gridcolor="#1E2A3A", tickfont=dict(color="#94A3B8"))
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94A3B8",
            legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=40, l=60, r=60),
            showlegend=True
        )
 
        col_radar, col_reasons = st.columns([1, 1])
 
        with col_radar:
            st.subheader("Comparación vs. comportamiento histórico")
            st.caption("Valores normalizados 0–1. Azul = promedio del dataset. Rojo/amarillo = sesión actual.")
            st.plotly_chart(fig_radar, use_container_width=True)
 
        with col_reasons:
            st.subheader("Factores de riesgo detectados")
            st.caption("Reglas disparadas durante el análisis de esta sesión.")
 
            if reasons:
                for emoji, titulo, detalle in reasons:
                    st.markdown(f"""
                    <div style='background:#111827; border:1px solid #1E2A3A; border-left:3px solid
                        {"#FF4D4D" if emoji == "🔴" else "#FFB300"};
                        border-radius:8px; padding:12px 14px; margin-bottom:10px;'>
                        <div style='font-size:13px; font-weight:600; color:#E2E8F0;'>{emoji} {titulo}</div>
                        <div style='font-size:12px; color:#64748B; margin-top:3px;'>{detalle}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#111827; border:1px solid #1E2A3A; border-left:3px solid #00D4FF;
                    border-radius:8px; padding:12px 14px;'>
                    <div style='font-size:13px; font-weight:600; color:#00D4FF;'>✅ Sin factores de riesgo</div>
                    <div style='font-size:12px; color:#64748B; margin-top:3px;'>
                        Todos los parámetros están dentro del rango normal del baseline histórico.
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
            # Nota técnica sobre el score del modelo
            st.markdown(f"""
            <div style='background:#0D1117; border:1px solid #1E2A3A; border-radius:8px;
                padding:12px 14px; margin-top:12px;'>
                <div style='font-size:11px; color:#334155; margin-bottom:4px; text-transform:uppercase;
                    letter-spacing:0.05em;'>Nota técnica — Isolation Forest</div>
                <div style='font-size:12px; color:#64748B; line-height:1.6;'>
                    <code style='color:#00D4FF;'>decision_function()</code> retornó
                    <code style='color:{"#FF4D4D" if anomaly_score < 0 else "#00D4FF"};'>{anomaly_score:.4f}</code>.
                    Valores {'negativos indican que el punto está en una región de baja densidad del árbol — característica de las anomalías.' if anomaly_score < 0 else 'positivos indican que el punto está en la región densa del espacio de features — comportamiento típico.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Guardar alerta en historial ──────────────────────────────────────
        alert = {
            "Hora":            login_hour,
            "Ubicación":       location,
            "Dispositivo":     device,
            "Actividad":       activity,
            "Intentos":        failed_attempts,
            "Accesos":         access_count,
            "Sesión (min)":    session_duration,
            "Riesgo":          risk_level,
            "Score":           f"{risk_score}%",
            "Modelo":          "Anómalo" if prediction[0] == -1 else "Normal"
        }
        st.session_state.alert_history.append(alert)

        # ── Guardar nuevo registro en CSV ────────────────────────────────────
        new_row = pd.DataFrame([{
            "login_hour":       login_hour,
            "location":         LOCATION_MAP[location],
            "failed_attempts":  failed_attempts,
            "access_count":     access_count,
            "activity_type":    ACTIVITY_MAP[activity],
            "device_type":      DEVICE_MAP[device],
            "session_duration": session_duration,
            "day_of_week":      DAY_MAP[day_of_week],
            "anomaly":          prediction[0]
        }])

        updated_data = pd.concat([data, new_row], ignore_index=True)
        updated_data.to_csv("data/processed_data.csv", index=False)
        # Limpiar el cache de datos para que el Dashboard refleje el nuevo registro
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
    col1.metric("Total Anomalías",    f"{anomaly_count:,}")
    col2.metric("Comportamientos Normales", f"{normal_count:,}")
    col3.metric("Registros Totales",  f"{total_records:,}")
    col4.metric("Tasa de Anomalía",   f"{risk_pct_dash}%")
 
    st.divider()
 
    # ── Gráficas ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
 
    with col_left:
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
        st.subheader("Intentos fallidos por frecuencia")
        failed_data = fresh_data["failed_attempts"].value_counts().reset_index()
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
 
    # ── Accesos por hora ─────────────────────────────────────────────────────
    st.subheader("Patrón de accesos por hora del día")
 
    # Separar normales y anómalos por hora
    hour_normal  = fresh_data[fresh_data["anomaly"] == 1]["login_hour"].value_counts().reset_index()
    hour_anomaly = fresh_data[fresh_data["anomaly"] == -1]["login_hour"].value_counts().reset_index()
    hour_normal.columns  = ["Hora", "Cantidad"]
    hour_anomaly.columns = ["Hora", "Cantidad"]
 
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(
        x=hour_normal.sort_values("Hora")["Hora"],
        y=hour_normal.sort_values("Hora")["Cantidad"],
        name="Normal", mode="lines+markers",
        line=dict(color="#00D4FF", width=2),
        marker=dict(size=6)
    ))
    fig_hour.add_trace(go.Scatter(
        x=hour_anomaly.sort_values("Hora")["Hora"],
        y=hour_anomaly.sort_values("Hora")["Cantidad"],
        name="Anómalo", mode="lines+markers",
        line=dict(color="#FF4D4D", width=2, dash="dot"),
        marker=dict(size=6)
    ))
    fig_hour.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94A3B8",
        xaxis=dict(title="Hora del día (0–23)", gridcolor="#1E2A3A"),
        yaxis=dict(title="Cantidad de accesos", gridcolor="#1E2A3A"),
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=20, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_hour, use_container_width=True)
 
    # ── Scatter: Hora vs Intentos fallidos (coloreado por anomalía) ──────────
    st.subheader("Mapa de calor: Hora de acceso vs. Intentos fallidos")
    st.caption("Cada punto es un registro. Rojo = comportamiento anómalo detectado por el modelo.")
 
    fig_scatter = px.scatter(
        fresh_data,
        x="login_hour",
        y="failed_attempts",
        color=fresh_data["anomaly"].map({1: "Normal", -1: "Anómalo"}),
        color_discrete_map={"Normal": "rgba(0,212,255,0.25)", "Anómalo": "#FF4D4D"},
        opacity=0.7,
        labels={"login_hour": "Hora de acceso", "failed_attempts": "Intentos fallidos", "color": "Clasificación"},
        size_max=8
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94A3B8",
        xaxis=dict(gridcolor="#1E2A3A"),
        yaxis=dict(gridcolor="#1E2A3A"),
        margin=dict(t=20, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
 
    st.divider()
 
    # ── Nivel de Riesgo Global ───────────────────────────────────────────────
    st.subheader("Nivel de riesgo global del sistema")
 
    # Usar el riesgo global basado en datos, no en el último análisis
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
            {risk_pct_dash}% de los registros son anómalos
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
        st.info("Aún no hay análisis registrados en esta sesión. Ve a **Análisis** para evaluar un comportamiento.")
 
 

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