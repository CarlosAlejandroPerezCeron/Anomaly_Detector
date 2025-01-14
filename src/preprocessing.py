import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Cargar datos
data = pd.read_csv("logs.csv")

# Seleccionar características y etiquetas
X = data[["cpu_usage", "memory_usage", "disk_io"]]
y = data["anomaly"]

# Normalizar los datos
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

print("Datos procesados y listos para el modelo.")
