import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Constantes y Configuración ---
DATASET_FILE = 'pong_q_expert_dataset.npy'
MODEL_SAVE_PATH = 'qtable_nn.h5'
NUM_ACTIONS =  2 # Subir, bajar es no hacer nada

# Hiperparámetros de entrenamiento
EPOCHS = 1000
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42 # Para reproducibilidad en shuffle y train_test_split

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table_final_test_v1.pkl'  # Cambia el path si es necesario
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

# --- Definir la red neuronal ---
model = keras.Sequential([
    # COMPLETAR: Definir la arquitectura de la red neuronal
    layers.Input(shape=(3,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense (NUM_ACTIONS)
])

model.compile(optimizer='adam', loss='mse')

# --- Entrenar la red neuronal ---
# COMPLETAR: Ajustar hiperparámetros según sea necesario
#model.fit(X, y, )
model.fit(X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE)
# --- Mostrar resultados del entrenamiento ---
# Completar: Imprimir métricas de entrenamiento

# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas
model.save('flappy_q_nn_model.h5')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.

