import cv2
import mediapipe as mp


def create_mask(img, lower, upper):
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Crear una máscara con los colores en el rango
    mask = cv2.inRange(hsv, lower, upper)
    return mask


# Función para verificar si la mano está cerrada
def is_fist(hand_landmarks, mp_hands, threshold=0.1):
    # Calcular si todos los dedos están doblados (distancias entre la yema y la base del dedo)
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    closed_fingers = 0
    for tip, pip in zip(finger_tips, finger_pips):
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y
        if tip_y > pip_y + threshold:  # Si la yema está más cerca de la palma
            closed_fingers += 1

    # Considerar que el puño está cerrado si 4 dedos están doblados
    return closed_fingers >= 4
