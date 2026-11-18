import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

keras.utils.set_random_seed(42)

print("Cargando el dataset de dígitos...")
digits = load_digits()

X = digits.data.astype("float32")
y = digits.target

X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

escalador = StandardScaler()
X_entrena = escalador.fit_transform(X_entrena)
X_prueba = escalador.transform(X_prueba)

joblib.dump(escalador, "escalador_digitos.pkl")

modelo = keras.Sequential([
    keras.Input(shape=(64,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")
])

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nEntrenando la red neuronal...")
modelo.fit(
    X_entrena, y_entrena,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

modelo.save("modelo_digitos.keras")
print("\n¡Modelo guardado exitosamente como 'modelo_digitos.keras'!")