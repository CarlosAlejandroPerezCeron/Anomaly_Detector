import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Configuración del título del dashboard
st.title("Dashboard de Detección de Anomalías en Logs")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV con los logs", type="csv")

if uploaded_file:
    # Cargar datos
    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())

    # Normalizar los datos
    scaler = MinMaxScaler()
    features = ["cpu_usage", "memory_usage", "disk_io"]
    data_normalized = scaler.fit_transform(data[features])

    # Cargar el modelo entrenado
    model = load_model("anomaly_detector_model.h5")

    # Reconstrucciones y errores
    reconstructions = model.predict(data_normalized)
    reconstruction_errors = np.mean(np.square(data_normalized - reconstructions), axis=1)

    # Calcular umbral para anomalías
    threshold = reconstruction_errors.mean() + 3 * reconstruction_errors.std()

    # Identificar anomalías
    data["anomaly"] = reconstruction_errors > threshold
    st.write("Datos con detección de anomalías:")
    st.dataframe(data)

    # Visualizar anomalías
    st.subheader("Visualización de anomalías")
    fig, ax = plt.subplots()
    ax.plot(data["cpu_usage"], label="Uso de CPU")
    ax.scatter(data.index[data["anomaly"]], data["cpu_usage"][data["anomaly"]], color="red", label="Anomalías")
    ax.set_title("Detección de anomalías en Uso de CPU")
    ax.legend()
    st.pyplot(fig)

    # Descargar resultados como CSV
    st.download_button(
        label="Descargar resultados como CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="logs_with_anomalies.csv",
        mime="text/csv"
    )

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Configuración del título del dashboard
st.title("Dashboard de Detección de Anomalías en Tiempo Real")

# Cargar el modelo entrenado
model = load_model("anomaly_detector_model.h5")

# Umbral para anomalías
def calcular_umbral(data_normalized, model):
    reconstructions = model.predict(data_normalized)
    reconstruction_errors = np.mean(np.square(data_normalized - reconstructions), axis=1)
    return reconstruction_errors.mean() + 3 * reconstruction_errors.std()

# Configuración de procesamiento en tiempo real
st.subheader("Logs en tiempo real")

placeholder = st.empty()
umbral = None

while True:
    # Leer el archivo de logs en tiempo real
    try:
        data = pd.read_csv("real_time_logs.csv")
        if len(data) > 0:
            # Normalizar los datos
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data[["cpu_usage", "memory_usage", "disk_io"]])

            # Calcular el umbral una sola vez
            if umbral is None:
                umbral = calcular_umbral(data_normalized, model)

            # Identificar anomalías
            reconstructions = model.predict(data_normalized)
            reconstruction_errors = np.mean(np.square(data_normalized - reconstructions), axis=1)
            data["anomaly"] = reconstruction_errors > umbral

            # Mostrar datos y gráficos
            with placeholder.container():
                st.write("Datos procesados:")
                st.dataframe(data.tail(10))

                st.subheader("Visualización de anomalías")
                fig, ax = plt.subplots()
                ax.plot(data["cpu_usage"], label="Uso de CPU")
                ax.scatter(data.index[data["anomaly"]], data["cpu_usage"][data["anomaly"]], color="red", label="Anomalías")
                ax.legend()
                st.pyplot(fig)

            st.sleep(2)  # Actualizar cada 2 segundos
    except Exception as e:
        st.warning(f"Error procesando los datos: {e}")
