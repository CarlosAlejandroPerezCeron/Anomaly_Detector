import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Cargar datos procesados
from preprocessing import X_train, X_test

# Crear el modelo autoencoder
autoencoder = Sequential([
    Dense(32, activation='relu', input_shape=(3,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Guardar el modelo
autoencoder.save("anomaly_detector_model.h5")

# Graficar la pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title("Pérdida del Modelo")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()

print("Modelo entrenado y guardado como 'anomaly_detector_model.h5'.")
