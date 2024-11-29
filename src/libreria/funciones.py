import cv2
import mediapipe as mp
import os
import random


# # Cargar las imágenes de referencia
# triangle_img = cv2.imread("triangle.png")  # Cargar la imagen del triángulo
# circle_img = cv2.imread("circle.png")  # Cargar la imagen del círculo
# rectangle_img = cv2.imread("rectangle.png")  # Cargar la imagen del rectángulo
# square_img = cv2.imread("square.png")  # Cargar la imagen del cuadrado

# # Almacenar las imágenes de referencia en un diccionario
# reference_images = {
#     "Triangulo": triangle_img,
#     "Circulo": circle_img,
#     "Rectangulo": rectangle_img,
#     "Cuadrado": square_img,
# }


def create_mask(img, lower, upper):
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Crear una máscara con los colores en el rango
    mask = cv2.inRange(hsv, lower, upper)
    return mask


# Función para detectar y clasificar figuras geométricas
def detect_geometric_shapes(img):
    # Encontrar los contornos en la imagen binaria
    fig = "No figure detected"
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Aproximar el contorno a un polígono
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Obtener el centro del contorno para el texto
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Clasificación de la figura según el número de vértices
        if len(approx) == 3:
            # Triángulo
            cv2.drawContours(img, [approx], -1, (0, 255, 0), -1)
            cv2.putText(
                img, "Triangulo", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
            fig = "Triangle"
        elif len(approx) == 4:
            # Cuadrado o Rectángulo
            # Verificar si el contorno es cuadrado o rectangular
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                cv2.drawContours(img, [approx], -1, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    "Cuadrado",
                    (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
                fig = "Square"
            else:
                cv2.drawContours(img, [approx], -1, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    "Rectangulo",
                    (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
                fig = "Rectangle"
        elif len(approx) > 4:
            # Círculo
            # Aproximar la forma a un círculo
            (x, y), radius = cv2.minEnclosingCircle(contour)
            cv2.drawContours(img, [approx], -1, (0, 255, 0), -1)
            cv2.putText(
                img, "Circulo", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
            fig = "Circle"

    return img, fig


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


# GESTIONAR NIVELES


def gestionar_niveles(nivel, opencv_images):
    if nivel <= 6:
        # Generar una lista de imágenes aleatorias
        lista = generar_imagenes_random(nivel, opencv_images)
        show_image(lista, len(lista))
        print("Repite las imágenes mostradas para continuar al siguiente nivel.")
    else:
        print("¡Felicidades! Has completado el juego.")


def generar_imagenes_random(nivel, opencv_images):
    lista = []
    for i in range(nivel):
        num = random.randint(0, len(opencv_images) - 1)
        imagen = opencv_images[num]
        lista.append(imagen)
    return lista


def show_image(imgs, num=None):
    if num is not None:
        imgs = imgs[:num]

    for i, img in enumerate(imgs):
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_copy = img.copy()
        cv2.putText(
            img_copy,
            f"Figura: {i + 1}",
            (5, 25),
            font,
            1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow(f"Imagen {i + 1}", img_copy)
        cv2.waitKey(1500)
        cv2.destroyWindow(f"Imagen {i + 1}")


def load_images_from_folder(folder):
    folder = os.path.join(".", "images", "figures")
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images
