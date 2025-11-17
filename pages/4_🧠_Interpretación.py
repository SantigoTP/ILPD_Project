import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide", 
    page_title="Explicación del Modelo de Predicción Hepática",
    initial_sidebar_state="expanded"
)

st.title("🔬 Explicación de las Decisiones del Modelo de Clasificación")
st.markdown("---")

st.write("Esta sección detalla los mecanismos internos por los cuales el modelo de *Machine Learning* clasifica a los pacientes, basándose en el **Dataset de Pacientes con Enfermedad Hepática (ILPD)**.")



st.header("1. Importancia de Variables Predictoras")
st.markdown("""
La **Importancia de Variables (Feature Importance)** es crucial para la interpretación. Muestra qué características de los pacientes son más influyentes en el resultado predictivo del modelo. Las variables con mayor puntaje son las que tienen el impacto más significativo en la probabilidad de clasificación.
""")

st.subheader("Variables clave del Modelo:")
st.markdown("""
- **Bilirrubina Total y Directa:** Son los indicadores primarios de disfunción hepática y, por lo tanto, suelen ser las variables más importantes en cualquier modelo de este tipo.
- **Enzimas Hepáticas (ALT, AST, ALP):** Estos marcadores son esenciales para detectar daño o inflamación celular en el hígado.
- **Relación Albúmina-Globulina:** Un fuerte indicador de enfermedad hepática crónica. Una disminución o inversión de esta relación (valor < 1) es un factor de riesgo elevado.
""")

st.subheader("Visualización de la Importancia")
st.image("/workspaces/ILPD_Project/src/feature_importance_plot.png", caption="Importáncia de variables")

st.markdown("---")


st.header("2. Umbrales de Clasificación (Thresholds)")
st.markdown("""El **Umbral ($\Theta$)** es el punto de corte de probabilidad que el modelo utiliza para convertir la predicción continua (probabilidad de ser Clase 1) en una clasificación binaria ('Enfermo' vs. 'No Enfermo').
""")

st.subheader("Umbral Estándar")
st.markdown(r"""
El umbral por defecto es $\Theta = 0.5$. La regla de decisión es:
$$
\text{Clase} = 
\begin{cases} 
\text{1 (Enfermo)} & \text{si } P(\text{Enfermo}) \ge 0.5 \\
\text{2 (No Enfermo)} & \text{si } P(\text{Enfermo}) < 0.5 
\end{cases}
$$
""")

st.subheader("Consideraciones para el Contexto Médico")
st.markdown("""
En contextos diagnósticos, el costo de un **Falso Negativo (FN)** (no diagnosticar a un enfermo) es mucho mayor que el de un **Falso Positivo (FP)**.
* **Ajuste:** Para maximizar la detección de casos reales (aumentar la **Sensibilidad** o *Recall*), el umbral se puede **reducir** (ej., a 0.4). Este ajuste minimiza el riesgo de FN, pero incrementa el número de FP.
""")

st.markdown("---")


st.header("3. Riesgos y Matriz de Confusión")
st.markdown("""
Los riesgos del modelo se cuantifican mediante el análisis de la **Matriz de Confusión**, la cual desglosa los tipos de aciertos y errores.
""")

st.subheader("Matriz de Confusión")
st.markdown("""
| Predicción | Real: Clase 1 (Enfermo) | Real: Clase 2 (No Enfermo) |
| :---: | :---: | :---: |
| **Predicho: Clase 1** | **Verdadero Positivo (VP)** | **Falso Positivo (FP)** |
| **Predicho: Clase 2** | **Falso Negativo (FN)** | **Verdadero Negativo (VN)** |
""")

st.subheader("Análisis de Riesgos Críticos")
st.markdown(r"""
1.  **Riesgo de Falsos Negativos (FN):**
    * **Consecuencia:** El paciente enfermo es clasificado como sano y, por ende, puede no recibir el tratamiento oportuno.
    * **Métrica asociada (Importante):** **Sensibilidad (Recall)** $\left( \frac{VP}{VP + FN} \right)$.
2.  **Riesgo de Falsos Positivos (FP):**
    * **Consecuencia:** El paciente sano es clasificado como enfermo, lo que provoca estrés, ansiedad y posibles costos innecesarios por pruebas confirmatorias.
    * **Métrica asociada:** **Especificidad** $\left( \frac{VN}{VN + FP} \right)$.
""")

st.subheader("Visualización de la Matriz de Confusión")
st.image("/workspaces/ILPD_Project/src/confusion_matrix.png", caption="Confusion Matrix")

st.markdown("---")


st.header("4. Limitaciones y Desafíos del Modelo")
st.markdown("""
El desarrollo del modelo estuvo sujeto a restricciones inherentes al *dataset* y a la simplificación del problema clínico:
""")

st.subheader("Restricciones del Dataset y el Preprocesamiento")
st.markdown("""
1.  **Imputación de Valores Faltantes:** El *dataset* ILPD original contenía valores faltantes, particularmente en la variable **Relación Albúmina-Globulina**. La técnica de imputación utilizada (ej. media o moda) introduce **ruido** en el modelo y puede sesgar la importancia de esta variable.
2.  **Desbalance de Clases:** El *dataset* presenta un **desequilibrio** en la distribución de la variable objetivo (Clase 1 vs. Clase 2). Esto puede llevar a que el modelo favorezca a la clase mayoritaria, resultando en una baja **Sensibilidad** (Falsos Negativos altos), a pesar de tener una alta precisión global.
""")

st.subheader("Restricciones Clínicas y de Generalización")
st.markdown("""
3.  **Generalización Geográfica y Étnica:** Al ser un conjunto de datos específico de la India (*Indian Liver Patient Dataset*), el modelo podría **no generalizar adecuadamente** a poblaciones de otras regiones del mundo con diferentes factores genéticos, dietéticos o patrones de enfermedad.
4.  **Simplificación Binaria:** El modelo solo predice la **presencia o ausencia** de enfermedad hepática (Clase 1 o 2). **Ignora la gravedad** o el tipo específico de la patología subyacente (ej. cirrosis, hepatitis viral, etc.), lo cual es vital para el manejo clínico real.
5.  **Correlación vs. Causalidad:** El modelo se basa en **correlaciones** estadísticas. Los cambios en los biomarcadores son solo síntomas. El modelo no puede identificar la **causa** raíz, por lo que su resultado debe ser siempre validado con una historia clínica completa.
""")