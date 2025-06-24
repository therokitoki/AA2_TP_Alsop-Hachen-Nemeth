import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Constantes y Configuración ---
MODEL_SAVE_PATH = 'src/flappy_q_nn_model.h5'
NUM_ACTIONS =  2 # Subir, bajar es no hacer nada

# Hiperparámetros de entrenamiento
EPOCHS = 500
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42 # Para reproducibilidad en shuffle y train_test_split

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'src/flappy_birds_q_table_final.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)

X = np.array(X)
y = np.array(y)

# --- Dividir en entrenamiento y validación ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)


# --- Definir la red neuronal ---
# model = keras.Sequential([
#     layers.Input(shape=(X.shape[1],)),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(NUM_ACTIONS)
# ])
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_ACTIONS)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# --- Callbacks para early stopping ---
#early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


# --- Entrenar la red neuronal ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    #callbacks=[early_stop],
    verbose=1
)

# --- Imprimir métricas finales ---
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]

print("\n--- Métricas finales ---")
print(f"Loss (entrenamiento): {final_loss:.4f}")
print(f"Loss (validación):    {final_val_loss:.4f}")
print(f"MAE (entrenamiento):  {final_mae:.4f}")
print(f"MAE (validación):     {final_val_mae:.4f}")

# --- Graficar métricas ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Loss (MSE)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE entrenamiento')
plt.plot(history.history['val_mae'], label='MAE validación')
plt.title('Error Absoluto Medio (MAE)')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas
model.save(MODEL_SAVE_PATH)
print(f'Modelo guardado como TensorFlow SavedModel en {MODEL_SAVE_PATH}')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.

