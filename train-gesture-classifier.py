import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Cargar datos ---
X = np.load("./rps_dataset.npy")   # (N, 42)
y = np.load("./rps_labels.npy")    # (N,)

# --- Preprocesamiento ---
X = np.array(X).astype("float32")
y = to_categorical(y, num_classes=3)  # one-hot encoding - rari no fue hecho por mi esta linea jejex

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Definir el modelo ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(42,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 clases: piedra, papel, tijera
])

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

# --- Entrenamiento ---
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_val, y_val)
)

# --- Guardar el modelo ---
model.save("./rps_model.h5")
print("✅ Modelo guardado como rps_model.h5")
