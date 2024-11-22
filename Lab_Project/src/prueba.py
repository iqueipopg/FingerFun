import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import funciones as f

# Configurar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Lista para almacenar la trayectoria
trajectory = []

kalman = cv2.KalmanFilter(4, 2)  # 4 estados (x, y, dx, dy), 2 mediciones (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

cap = cv2.VideoCapture(0)
cont = 0
# Definir los límites para el azul puro en HSV
lower_blue = np.array([110, 150, 150])  # H:110, S:150, V:150
upper_blue = np.array([130, 255, 255])  # H:130, S:255, V:255

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a RGB para MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener la yema del dedo índice
            index_finger_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            h, w, _ = frame.shape
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)

            # Actualizar el filtro de Kalman
            kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))
            prediction = kalman.predict()

            # Dibujar la posición predicha
            pred_x, pred_y = int(prediction[0]), int(prediction[1])
            trajectory.append((pred_x, pred_y))
            cv2.circle(frame, (pred_x, pred_y), 10, (0, 255, 0), -1)

            if len(trajectory) > 100:
                trajectory.pop(0)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

    # Mostrar el frame
    cv2.imshow("Shape Detection", frame)

    # Detectar teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Salir con 'q'
        break
    elif key == 32:  # Borrar trayectoria con espacio (ASCII 32)
        print("Trayectoria borrada")
        trajectory.clear()
    elif key == 13:
        print("Guardando y borrando trayectoria")
        time.sleep(0.5)
        cont += 1

        # Dibujar la trayectoria antes de guardar
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

        # Crear carpeta si no existe
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        if not os.path.exists("masks"):
            os.makedirs("masks")
        # Guardar la imagen con trayectoria
        cv2.imwrite(f"screenshots/screenshot{cont}.png", frame)
        cv2.imwrite(
            f"masks/mask{cont}.png",
            f.create_mask(frame, lower_blue, upper_blue),
        )

        # Limpiar la trayectoria
        trajectory.clear()

cap.release()
cv2.destroyAllWindows()
hands.close()
