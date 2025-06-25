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
NUM_ACTIONS =  2

# Hiperparámetros de entrenamiento
EPOCHS = 500
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'src/flappy_birds_q_table_final.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Parámetros de discretización (ajusta si es necesario) ---
bins_a = 31
bins_b = 16  # Cambié de 15 a 16 para incluir índice 15 si existe
bins_c = 16

# --- Verificar valores máximos ---
states_raw = np.array(list(q_table.keys()))
print("Máximo valor de a:", np.max(states_raw[:, 0]))
print("Máximo valor de b:", np.max(states_raw[:, 1]))
print("Máximo valor de c:", np.max(states_raw[:, 2]))

def one_hot_encode_state(state):
    a, b, c = state
    assert 0 <= a < bins_a, f"Valor fuera de rango para a: {a}"
    assert 0 <= b < bins_b, f"Valor fuera de rango para b: {b}"
    assert 0 <= c < bins_c, f"Valor fuera de rango para c: {c}"

    a_vec = np.zeros(bins_a)
    b_vec = np.zeros(bins_b)
    c_vec = np.zeros(bins_c)
    a_vec[a] = 1
    b_vec[b] = 1
    c_vec[c] = 1
    return np.concatenate([a_vec, b_vec, c_vec])

# --- Preparar datos ---
X = np.array([one_hot_encode_state(state) for state in q_table.keys()])
y = np.array(list(q_table.values()))


# --- Dividir en entrenamiento y validación ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)


# --- Modelo ---
model = keras.Sequential([
    layers.Input(shape=(bins_a, bins_b, bins_c,)),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_ACTIONS)  # Salida Q-values por acción
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# --- Callbacks para early stopping ---
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

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

