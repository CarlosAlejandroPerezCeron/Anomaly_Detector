import pandas as pd
import numpy as np

# Generar datos simulados
np.random.seed(42)
n_samples = 10000

# Datos normales
normal_data = pd.DataFrame({
    "timestamp": pd.date_range(start="2023-01-01", periods=n_samples, freq="T"),
    "cpu_usage": np.random.normal(50, 10, n_samples),
    "memory_usage": np.random.normal(30, 5, n_samples),
    "disk_io": np.random.normal(70, 15, n_samples),
    "anomaly": 0
})

# Datos anómalos
anomalous_data = pd.DataFrame({
    "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="T"),
    "cpu_usage": np.random.normal(90, 5, 100),
    "memory_usage": np.random.normal(80, 10, 100),
    "disk_io": np.random.normal(120, 10, 100),
    "anomaly": 1
})

# Combinar y guardar
data = pd.concat([normal_data, anomalous_data]).sample(frac=1).reset_index(drop=True)
data.to_csv("logs.csv", index=False)

print("Dataset generado y guardado como 'logs.csv'.")
