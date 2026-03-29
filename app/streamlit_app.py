import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🔐",
    layout="wide"
)

# Título principal
st.title("🔐 Sentinel AI")
st.subheader("Sistema de detección de comportamiento anómalo de usuarios")

st.divider()

# Sidebar
st.sidebar.title("Sentinel AI")
st.sidebar.markdown("Sistema de detección de amenazas")

# Menú
menu = st.sidebar.selectbox(
    "Navegación",
    ["Inicio", "Análisis", "Dashboard", "Acerca de"]
)

# Inicio
if menu == "Inicio":
    st.header("Bienvenido a Sentinel AI")
    st.write("Este sistema detecta comportamientos anómalos en usuarios para prevenir riesgos de seguridad.")

# Análisis
elif menu == "Análisis":
    st.header("Análisis de Usuario")
    st.write("Aquí se analizará el comportamiento del usuario.")

# Dashboard
elif menu == "Dashboard":
    st.header("Dashboard")
    st.write("Visualización de datos y resultados.")

# Acerca de
elif menu == "Acerca de":
    st.header("Acerca del Proyecto")
    st.write("Proyecto desarrollado con Inteligencia Artificial.")