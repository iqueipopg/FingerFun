import cv2
import mediapipe as mp
import numpy as np
import libreria.funciones as f

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
kalman.processNoiseCov = (
    np.eye(4, dtype=np.float32) * 0.03
)  # Incrementar para suavizar más
kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * 0.1
)  # Reducir para confiar más en predicciones

cap = cv2.VideoCapture(0)
cont = 0
# Definir los límites para el azul puro en HSV
lower_blue = np.array([110, 150, 150])  # H:110, S:150, V:150
upper_blue = np.array([130, 255, 255])  # H:130, S:255, V:255

image_folder = "figures"

opencv_images = f.load_images_from_folder(image_folder)

global nivel
nivel = 0
