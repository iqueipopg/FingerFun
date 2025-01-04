import cv2
import mediapipe as mp
import os
import random
import shutil

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


def gestionar_niveles(nivel, opencv_images, names):
    if nivel <= 6:
        # Generar una lista de imágenes aleatorias
        lista, nombres = generar_imagenes_random(nivel, opencv_images, names)
        show_image(lista, len(lista))
        print("Repite las imágenes mostradas para continuar al siguiente nivel.")
    else:
        print("¡Felicidades! Has completado el juego.")
    
    return nombres


def generar_imagenes_random(nivel, opencv_images, names):
    lista = []
    nombres = []
    for i in range(nivel):
        num = random.randint(0, len(opencv_images) - 1)
        imagen = opencv_images[num]
        lista.append(imagen) # imágenes 
        nombres.append(names[num]) # nombres de imagenes
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


def load_images_from_folder(folder):
    folder = os.path.join(".", "images", "figures")
    images = []
    names = []
    for filename in os.listdir(folder):
        name = filename[:-4] 
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            names.append(name)
    return images, names

def mostrar_menu(puntuacion_maxima, rondas_jugadas, inicio = False):
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpia la terminal

    ancho = 52  # Ancho total del marco

    # Encabezado
    print("╔" + "═" * ancho + "╗")
    print("║{:^{}}║".format("🎮 Bienvenido a 🎮", ancho -2))
    print("║{:^{}}║".format('FINGERFUN', ancho))
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
        key = cv2.waitKey(0) & 0xFF  # Espera hasta que se presione una tecla
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
    overlay_image = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # Leer con canal alfa
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
        opcion = mostrar_menu(puntuacion_maxima, rondas_jugadas, inicio = False)
        running = True
    else:
        opcion = mostrar_menu(puntuacion_maxima, rondas_jugadas, inicio = True)

    if opcion == 1:
        nivel = 1
        print(f"Pasando al nivel: {nivel}")
        secuencia = gestionar_niveles(nivel, opencv_images, names)
        print(secuencia)

        return secuencia
    else:
        print("Saliendo del juego...")
        return False