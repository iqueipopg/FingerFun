import cv2
import mediapipe as mp
import os
import random
import shutil
import numpy as np
from libreria.constants import *

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


def eliminar_readonly(func, path, excinfo):
    os.chmod(path, 0o777)  # Cambia los permisos a escritura
    func(path)  # Reintenta eliminar


def create_mask(img, lower, upper):
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Crear una máscara con los colores en el rango
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def detect_geometric_shapes2(img, opencv_images, names):
    """
    Detecta y clasifica las figuras geométricas en una imagen usando template matching.

    :param img: Imagen de entrada (en blanco y negro) que contiene la figura a detectar.
    :param opencv_images: Lista de imágenes de referencia de las figuras.
    :param names: Nombres de las figuras correspondientes a las imágenes en opencv_images.

    :return: Imagen con la figura detectada y su nombre.
    """
    # Convertir la imagen de entrada a escala de grises y asegurarse de que es uint8
    if len(img.shape) == 3:  # Si la imagen tiene más de un canal (RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = np.uint8(img)  # Convertir a uint8 si ya está en escala de grises

    # Inicializar la variable que contendrá la figura detectada
    detected_figure = "No figure detected"
    max_match_value = 0  # Valor máximo de coincidencia (inicialmente 0)

    for i, template in enumerate(opencv_images):
        # Asegurarse de que las plantillas están en formato adecuado (escala de grises y uint8)
        if len(template.shape) == 3:  # Si la plantilla tiene más de un canal (RGB)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = np.uint8(
                template
            )  # Convertir a uint8 si ya está en escala de grises

        # Verificar el tamaño de la plantilla y redimensionarla si es necesario
        if (
            template_gray.shape[0] > img_gray.shape[0]
            or template_gray.shape[1] > img_gray.shape[1]
        ):
            template_gray = cv2.resize(
                template_gray, (img_gray.shape[1], img_gray.shape[0])
            )

        # Aplicar template matching
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Obtener el valor máximo de coincidencia y la ubicación
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Si la coincidencia es mejor que la anterior, actualizamos la figura detectada
        if max_val > max_match_value:
            max_match_value = max_val
            detected_figure = names[i]

            # Dibujar un rectángulo alrededor de la coincidencia
            h, w = template.shape[:2]
            cv2.rectangle(
                img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2
            )

    # Especificar el texto con el nombre de la figura detectada
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img, detected_figure, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA
    )

    return img, detected_figure


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
        if len(approx) == 4:
            # Cuadrado o Rombo
            # Verificar si el contorno es un cuadrado o rombo
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.95 <= aspect_ratio <= 1.05:
                # Es un cuadrado (relación de aspecto cercana a 1)
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
                # Es un rombo (si no es un cuadrado)
                cv2.drawContours(img, [approx], -1, (0, 255, 0), -1)
                cv2.putText(
                    img, "Rombo", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                )
                fig = "Diamond"

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


# Función para calcular el ángulo entre tres puntos
def calculate_angle(p1, p2, p3):
    # Calcular los vectores
    v1 = np.array([p1[0][0] - p2[0][0], p1[0][1] - p2[0][1]])
    v2 = np.array([p3[0][0] - p2[0][0], p3[0][1] - p2[0][1]])

    # Calcular el ángulo entre los dos vectores
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calcular el ángulo en radianes y convertirlo a grados
    angle = np.degrees(np.arccos(dot_product / (magnitude_v1 * magnitude_v2)))
    return angle


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


def gestionar_niveles(nivel, opencv_images, names):
    if nivel <= 6:
        # Generar una lista de imágenes aleatorias
        lista, nombres = generar_imagenes_random(nivel, opencv_images, names)
        show_image(lista, len(lista))
        print("Repite las imágenes mostradas para continuar al siguiente nivel.")
        return nombres
    else:
        print("¡Felicidades! Has completado el juego.")
        exit()


def generar_imagenes_random(nivel, opencv_images, names):
    lista = []
    nombres = []
    for i in range(nivel):
        num = random.randint(0, len(opencv_images) - 1)
        imagen = opencv_images[num]
        lista.append(imagen)  # imágenes
        nombres.append(names[num])  # nombres de imagenes
    return lista, nombres


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


def mostrar_menu(puntuacion_maxima, rondas_jugadas, inicio=False):
    os.system("cls" if os.name == "nt" else "clear")  # Limpia la terminal

    ancho = 52  # Ancho total del marco

    # Encabezado
    print("╔" + "═" * ancho + "╗")
    print("║{:^{}}║".format("🎮 Bienvenido a 🎮", ancho - 2))
    print("║{:^{}}║".format("FINGERFUN", ancho))
    print("╚" + "═" * ancho + "╝")

    # Información del juego
    print("╔" + "═" * ancho + "╗")
    print("║{:^{}}║".format("Información del Juego", ancho))
    print("╟" + "─" * ancho + "╢")
    print(f"║ {'Creadores: Ignacio Queipo | Beltrán Sánchez':<{ancho - 1}}║")
    print(f"║ {'Puntuación Máxima: ' + str(puntuacion_maxima):<{ancho - 1}}║")
    print(f"║ {'Rondas Jugadas: ' + str(rondas_jugadas):<{ancho - 1}}║")
    print("╚" + "═" * ancho + "╝")

    # Opciones
    print("╔" + "═" * ancho + "╗")
    print("║{:^{}}║".format("Opciones", ancho))
    print("╟" + "─" * ancho + "╢")
    if inicio:
        print(f"║ {'1. Jugar':<{ancho - 1}}║")
    else:
        print(f"║ {'1. Volver a jugar':<{ancho - 1}}║")
    print(f"║ {'2. Salir':<{ancho - 1}}║")
    print("╚" + "═" * ancho + "╝")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 49:  # Si presiona la tecla 1
            opcion = 1
            if inicio:
                print("Iniciando juego...")
            else:
                print("¡Volviendo a jugar!")
            return opcion
        elif key == 50:  # Si presiona la tecla 2
            opcion = 2
            print("¡Gracias por jugar!")
            return opcion
        else:
            print("Por favor, presiona 1 o 2.")


def superponer_imagen_fullscreen(cap, overlay_path):
    """
    Superpone una imagen en pantalla completa sobre el frame de la cámara en la misma ventana.

    :param cap: Objeto cv2.VideoCapture (captura de cámara).
    :param overlay_path: Ruta de la imagen a superponer.
    """
    # Carga la imagen a superponer
    overlay_image = cv2.imread(
        overlay_path, cv2.IMREAD_UNCHANGED
    )  # Leer con canal alfa
    if overlay_image is None:
        print(f"No se pudo cargar la imagen de superposición: {overlay_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede recibir el frame. Saliendo...")
            break

        # Redimensionar la imagen de superposición al tamaño del frame
        frame_height, frame_width, _ = frame.shape
        overlay_resized = cv2.resize(overlay_image, (frame_width, frame_height))

        # Separar los canales BGR y Alfa de la imagen superpuesta
        if overlay_resized.shape[2] == 4:  # Si la imagen tiene canal alfa
            b, g, r, a = cv2.split(overlay_resized)
            alpha = a / 255.0  # Normalizar canal alfa
            for c in range(0, 3):  # Aplicar en los canales BGR
                frame[:, :, c] = (
                    alpha * overlay_resized[:, :, c] + (1 - alpha) * frame[:, :, c]
                )
        else:
            frame = overlay_resized  # Si no hay canal alfa, reemplaza directamente

        # Mostrar el frame original con la imagen superpuesta
        cv2.imshow("Camara con Superposición", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()


def limpiar_imagenes():
    directorio = "images/output"
    if os.path.exists(directorio) and os.path.isdir(directorio):
        try:
            shutil.rmtree(directorio, onerror=eliminar_readonly)
            print(f"El directorio {directorio} ha sido eliminado.")
        except OSError as e:
            print(f"Error al eliminar el directorio: {e}")
    else:
        print(f"El directorio {directorio} no existe o no es un directorio.")


def inicio_partida(running, opencv_images, names, puntuacion_maxima, rondas_jugadas):
    if running == False:
        opcion = mostrar_menu(puntuacion_maxima, rondas_jugadas, inicio=False)
        running = True
    else:
        opcion = mostrar_menu(puntuacion_maxima, rondas_jugadas, inicio=True)

    if opcion == 1:
        nivel = 1
        print(f"Pasando al nivel: {nivel}")
        secuencia = gestionar_niveles(nivel, opencv_images, names)
        print(secuencia)
        nivel += 1

        return secuencia
    else:
        print("Saliendo del juego...")
        return False


def iniciar_partida():
    """Inicializa las variables y secuencia para una nueva partida"""
    return (
        0,
        0,
        1,
        True,
        [],
    )  # contador, rondas_jugadas, nivel, primera_partida, secuencia


def gestionar_rondas(figura_detectada, secuencia, contador, nivel):
    """Lógica para gestionar las rondas y niveles"""
    if figura_detectada == secuencia[contador]:
        print(f"Figura correcta: {figura_detectada}")
        contador += 1
        if contador == nivel:  # Se completó el nivel
            print(f"Nivel {nivel} completado")
            nivel += 1  # Avanza al siguiente nivel
            contador = 0  # Reinicia el contador para el nivel siguiente
            secuencia = gestionar_niveles(
                nivel, opencv_images, names
            )  # Nuevas figuras para el siguiente nivel
            print(f"Pasando al siguiente nivel: {nivel}")
    else:
        print(f"Figura incorrecta: {figura_detectada}")
        print("DERROTA")
        return (
            False,
            nivel,
            contador,
            secuencia,
        )  # Si la figura es incorrecta, termina el juego

    return True, nivel, contador, secuencia


def guardar_resultados(frame, cont, secuencia, contador, nivel):
    """Guarda la imagen y los resultados después de cada ronda"""
    if not os.path.exists("images/output"):
        os.makedirs("images/output/screenshots")
        os.makedirs("images/output/masks")
        os.makedirs("images/output/shapes")

    # Guardar la imagen con trayectoria
    cv2.imwrite(f"images/output/screenshots/screenshot{cont}.png", frame)
    mask = create_mask(frame, lower_blue, upper_blue)
    cv2.imwrite(f"images/output/masks/mask{cont}.png", mask)

    # Detectar las formas geométricas en la imagen guardada
    img_with_shapes, fig = detect_geometric_shapes2(mask, opencv_images, names)
    cv2.imwrite(f"images/output/shapes/shape{cont}.png", img_with_shapes)

    print(f"Figura detectada: {fig}, Secuencia: {secuencia[contador]}")
    return fig


# Función para dibujar el corazón en la pantalla
def draw_heart(frame, x, y, scale=0.1):
    # Cargar la imagen del corazón
    heart_image = cv2.imread(
        "images/heart.png", cv2.IMREAD_UNCHANGED
    )  # Asegúrate de que el archivo heart.png esté en el directorio

    if heart_image is None:
        print("Error: No se pudo cargar la imagen del corazón.")
        exit()
    heart_resized = cv2.resize(heart_image, None, fx=scale, fy=scale)
    heart_h, heart_w, _ = heart_resized.shape
    roi = frame[y : y + heart_h, x : x + heart_w]

    # Crear una máscara para el corazón con transparencia
    mask = heart_resized[:, :, 3]
    heart_resized_rgb = heart_resized[:, :, :3]

    # Colocar el corazón sobre el fondo (con máscara)
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask / 255.0) + heart_resized_rgb[
            :, :, c
        ] * (mask / 255.0)

    frame[y : y + heart_h, x : x + heart_w] = roi
    return frame
