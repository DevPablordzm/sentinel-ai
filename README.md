# Sentinel AI — Sistema de Detección de Anomalías en Redes

> Sistema inteligente de monitoreo que detecta comportamientos sospechosos de usuarios usando Machine Learning no supervisado.

---

## ¿Qué problema resuelve?

Las organizaciones necesitan detectar accesos no autorizados o comportamientos inusuales en sus sistemas. Los métodos tradicionales basados en reglas fijas no se adaptan a patrones nuevos.

**Sentinel AI** usa Isolation Forest para aprender el comportamiento normal de los usuarios y detectar automáticamente cualquier desviación, sin necesidad de datos etiquetados previamente.

---


## ¿Cómo funciona?

```
Datos de sesión → Preprocesamiento → Isolation Forest → Score de anomalía → Dashboard
```

1. Se capturan 8 features del comportamiento del usuario
2. Las variables categóricas se convierten a números con `LabelEncoder`
3. El modelo calcula un score continuo con `decision_function()`
4. Si el score es negativo → comportamiento anómalo
5. Los resultados se visualizan en el dashboard interactivo

---

## Modelo de IA

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Algoritmo | Isolation Forest | Detección de anomalías no supervisada |
| `contamination` | 0.05 | 5% de anomalías esperadas en el dataset |
| `n_estimators` | 100 | Número de árboles en el ensemble |
| `random_state` | 42 | Reproducibilidad de resultados |

**¿Por qué Isolation Forest?**  
Porque las anomalías son más fáciles de aislar que los puntos normales. El algoritmo construye árboles aleatorios y mide cuántos cortes necesita para aislar cada punto — pocos cortes = anomalía.

---

## Features del modelo

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `login_hour` | Numérica | Hora del acceso (0–23) |
| `location` | Categórica | País de origen (CR, US, MX, ES) |
| `failed_attempts` | Numérica | Intentos fallidos de login |
| `access_count` | Numérica | Recursos accedidos en la sesión |
| `activity_type` | Categórica | Tipo de actividad (login, download, upload, delete) |
| `device_type` | Categórica | Dispositivo usado |
| `session_duration` | Numérica | Duración en minutos |
| `day_of_week` | Categórica | Día de la semana |

---

## Stack tecnológico

- **Python 3.x**
- **Streamlit** — interfaz web interactiva
- **scikit-learn** — modelo Isolation Forest
- **Pandas / NumPy** — manipulación de datos
- **Plotly** — visualizaciones interactivas

---

## Estructura del proyecto

```
sentinel_ai/
├── app.py/
│   └── streamlit_app.py    # Aplicación principal
├── docs/
│   └── requerimets.txt    # Lista de dependencias
├   └── README.md
│   └── plan.md            
├── model/
│   └── anomaly_model.pkl   # Modelo entrenado
├── data/
│   ├── user_behavior.csv   # Dataset original
│   └── processed_data.csv  # Dataset con predicciones
├── utils/
│   └── styles.py           # Estilos CSS
└── notebooks/
    └── dataset_generator.ipynb  # Entrenamiento documentado
```

---

## Instalación y uso local

```bash
# 1. Clonar el repositorio
git clone https://github.com/DevPablordzm/sentinel-ai.git
cd sentinel-ai

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Generar el modelo (correr el notebook completo)
jupyter notebook notebooks/dataset_generator.ipynb

# 4. Correr la aplicación
streamlit run streamlit_app.py
```

---

## Contexto académico

Proyecto final — **Inteligencia Artificial Aplicada**  
Ingeniería en Sistemas — 2026