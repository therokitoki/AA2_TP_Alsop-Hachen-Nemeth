############################################################################################
#                                                                                          #
#                               APRENDIZAJE AUTOMÁTICO 2                                   #
#                                 TRABAJO PRÁCTICO N°1                                     #
#                                                                                          #
#                GRUPO N°12: Ulises Nemeth, Agustín Alsop, Rocío Hachen                    #
#                                                                                          #
#                          Problema 2 - Piedra, Papel y Tijera                             #
#                                                                                          #
#                                 Generación de modelo                                     #
############################################################################################

# --- LIBRERÍAS ---
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- CARGA DE DATOS Y LABELS ---
X = np.load("./dataset/rps_dataset.npy")   # (N, 42)
y = np.load("./dataset/rps_labels.npy")    # (N,)

# --- PROCESAMIENTO ---
X = np.array(X).astype("float32")
y = to_categorical(y, num_classes=3)  # one-hot encoding

# --- DIVISIÓN DE DATOS EN TEST Y TRAIN ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- DEFINICIÓN DE MODELO ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 clases: piedra, papel, tijera
])

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

# --- ENTRENAMIENTO ---
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val)
)

# --- GUARDAR EL MODELO ---
model.save("./modelo/rps_model.h5")
print("Modelo guardado como rps_model.h5")
