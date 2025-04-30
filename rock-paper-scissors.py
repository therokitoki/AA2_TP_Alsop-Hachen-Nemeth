import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- CONFIGURACIÓN ---
model = load_model("rps_model.h5")
RESIZE_DIM = (320, 240)
label_map = {'r': 0, 'p': 1, 's': 2}
labels = ['Piedra', 'Papel', 'Tijera']
# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=0
)
mp_drawing = mp.solutions.drawing_utils

# Captura
cap = cv2.VideoCapture(0)

print("ESC para salir y guardar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen.")
        break

    small = cv2.resize(frame, RESIZE_DIM)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # Dibuja los puntos de la mano si están presentes
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        vec = [coord for p in lm for coord in (p.x, p.y)]
        #vec son los 42 point
        X = np.array([vec])  # forma (1, 42)
        pred = model.predict(X, verbose=0)[0]
        pred_label = labels[np.argmax(pred)]
        confidence = np.max(pred)
        cv2.putText(frame,
            f"{pred_label} ({confidence:.2f})",  # texto con clase + confianza
            (10, 30),                            # posición (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,           # tipo de fuente
            1,                                   # escala
            (0, 255, 0),                         # color (verde)
            2)                                   # grosor
        print(pred_label)
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #print(res.multi_hand_landmarks)

    # Mostrar ventana
    cv2.imshow("RPS Fast Collector", frame)


    key = cv2.waitKey(1) & 0xFF


    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

