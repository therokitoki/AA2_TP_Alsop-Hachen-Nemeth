import cv2
import time
import numpy as np
import mediapipe as mp

# pip install opencv-python
# pip install mediapipe

###
# r para capturar landmarks de piedra
# p para capturar landmarks de papel
# s para capturar landmarks de tijera
#
# Mantener una sola mano dentro del cuadro
# 
# ESC para obtener los archivos npy
###

# --- CONFIG ---
BATCH_SIZE = 50          # cantidad de datos por clase
RESIZE_DIM = (320, 240)  # menor resolucion para mayor velocidad
INTERVAL = 0.1          # cantidad de segundos entre capturas
DATA_FILE = "rps_dataset.npy"
LABEL_FILE = "rps_labels.npy"

# label to integer
label_map = {'r': 0, 'p': 1, 's': 2}

# storage
data, labels = [], []

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=0
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("RPS Fast Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord(k) for k in label_map]:
        cls = chr(key)
        cls_id = label_map[cls]
        print(f"[{cls.upper()}] Capturing {BATCH_SIZE} samples…")

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
                vec = []
                for p in lm:
                    vec += [p.x, p.y]
                data.append(vec)
                labels.append(cls_id)
                count += 1

            cv2.putText(frame,
                        f"{cls.upper()} {count}/{BATCH_SIZE}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("RPS Fast Collector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            elapsed = time.time() - start
            target = count * INTERVAL
            if target > elapsed:
                time.sleep(target - elapsed)

        print(f"[{cls.upper()}] Done.")

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Saving .npy files…")
np.save(DATA_FILE, np.array(data))
np.save(LABEL_FILE, np.array(labels))
print(f"Saved:\n • {DATA_FILE} ({len(data)}×42)\n • {LABEL_FILE} ({len(labels)})")
