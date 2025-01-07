import cv2
import mediapipe as mp
import numpy as np
import os

def load_images_from_folder(folder):
    folder = os.path.join(".", "images", folder)
    images = []
    names = []
    for filename in os.listdir(folder):
        name = filename[:-4]
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            names.append(name)
    return images, names


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
password_folder = "passwords"
calibration_folder = "calibration"

opencv_images, names = load_images_from_folder(image_folder)
passwords, _ = load_images_from_folder(password_folder)
calibration, _ = load_images_from_folder(calibration_folder)

mapeo_nombres = {"BSC": "BELTRÁN SÁNCHEZ", "IQL": "IGNACIO QUEIPO", "EVS": "ERIK VELASCO"}

global nivel
nivel = 0
puntuacion_maxima = 0
rondas_jugadas = 0
