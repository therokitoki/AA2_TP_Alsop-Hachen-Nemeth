import cv2
import time
import numpy as np
import mediapipe as mp

# --- CONFIGURACIÓN ---
BATCH_SIZE = 50
RESIZE_DIM = (320, 240)
INTERVAL = 0.1
DATA_FILE = "rps_dataset.npy"
LABEL_FILE = "rps_labels.npy"
label_map = {'r': 0, 'p': 1, 's': 2}

# Almacenamiento de datos
data, labels = [], []

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
print("Presioná 'r' (piedra), 'p' (papel), o 's' (tijera) para comenzar a capturar.")
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
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar ventana
    cv2.imshow("RPS Fast Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Si es una tecla válida de clase
    if key in [ord(k) for k in label_map]:
        cls = chr(key)
        cls_id = label_map[cls]
        print(f"[{cls.upper()}] Capturando {BATCH_SIZE} muestras…")

        count = 0
        start = time.time()
        while count < BATCH_SIZE:
            ret, frame = cap.read()
            if not ret:
                break

            small = cv2.resize(frame, RESIZE_DIM)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                vec = [coord for p in lm for coord in (p.x, p.y)]
                data.append(vec)
                labels.append(cls_id)
                count += 1

            # Dibuja landmarks y progreso
            if res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame,
                        f"{cls.upper()} {count}/{BATCH_SIZE}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("RPS Fast Collector", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC para cancelar en medio
                print("Captura cancelada.")
                break

            elapsed = time.time() - start
            target = count * INTERVAL
            if target > elapsed:
                time.sleep(target - elapsed)

        print(f"[{cls.upper()}] Captura completada.")

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# Guardar solo si hay datos
if data:
    print("Guardando archivos .npy…")
    np.save(DATA_FILE, np.array(data))
    np.save(LABEL_FILE, np.array(labels))
    print(f"Guardado:\n • {DATA_FILE} ({len(data)}×42)\n • {LABEL_FILE} ({len(labels)})")
else:
    print("No se capturaron datos. No se guardaron archivos.")
