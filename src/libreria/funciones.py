import cv2
import mediapipe as mp
import os
import random
import shutil
import numpy as np
import copy
import matplotlib.pyplot as plt
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


def write_image(imgs):
    for i, img in enumerate(imgs):
        cv2.imwrite(f"corner_{i}.jpg", img)


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


def obtener_paths(carpeta):
    carpeta = os.path.join(".", "images", carpeta)
    archivos = []
    for archivo in os.listdir(carpeta):
        # Crear el path completo y verificar que sea un archivo
        path_completo = os.path.join(carpeta, archivo)
        if os.path.isfile(path_completo):
            archivos.append(path_completo)
    return archivos


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


def detectar_contraseñas(plantillas, imagen_capturada, umbral=50):
    imagen_capturada = cv2.flip(imagen_capturada, 1)
    imagen_gris = cv2.cvtColor(imagen_capturada, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp_captura, des_captura = orb.detectAndCompute(imagen_gris, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for plantilla_path in plantillas:
        plantilla = cv2.imread(plantilla_path, cv2.IMREAD_GRAYSCALE)

        kp_plantilla, des_plantilla = orb.detectAndCompute(plantilla, None)

        # Comparar características si existen descriptores
        if des_captura is not None and des_plantilla is not None:
            matches = bf.match(des_captura, des_plantilla)
            matches = sorted(matches, key=lambda x: x.distance)

            umbral_buenas = umbral
            buenas_coincidencias = [m for m in matches if m.distance < 50]

            nombre = plantilla_path.split("\\")[-1].split(".")[0].split("_")[-1].upper()
            print(
                f"Comparando con {nombre}: {len(buenas_coincidencias)} buenas coincidencias"
            )

            umbral_buenas = 110 if ("B" in nombre or "S" in nombre) else umbral
            if len(buenas_coincidencias) > umbral_buenas:
                # print(f"Contraseña correcta. Coincide con {nombre}")
                return plantilla_path.split("_")[-1].split(".")[0].upper()

    return None


def get_chessboard_points(chessboard_shape, dx, dy):
    chessboard_points = np.zeros(
        (chessboard_shape[0] * chessboard_shape[1], 3), np.float32
    )
    chessboard_points[:, :2] = np.mgrid[
        0 : chessboard_shape[0], 0 : chessboard_shape[1]
    ].T.reshape(-1, 2)
    chessboard_points[:, 0] *= dx
    chessboard_points[:, 1] *= dy
    return chessboard_points


def calibrar_camara(calibration):
    imgs = calibration
    corners = []
    ret_list = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for img in imgs:
        cor = cv2.findChessboardCorners(img, (8, 6))
        corners.append(cor)

    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    corners_refined = [
        cv2.cornerSubPix(i, cor[1], (8, 6), (-1, -1), criteria) if cor[0] else []
        for i, cor in zip(imgs_gray, corners_copy)
    ]

    imgs_copy = copy.deepcopy(imgs)

    imgs_corners = [
        cv2.drawChessboardCorners(img, (8, 6), cor[1], cor[0]) if cor[0] else img
        for img, cor in zip(imgs_copy, corners)
    ]
    os.makedirs("images/calibration/corners", exist_ok=True)
    for i in range(len(imgs_corners)):
        cv2.imwrite(f"images/calibration/corners/corners_{i}.jpg", imgs_corners[i])

    chessboard_points = get_chessboard_points((8, 6), 30, 30)

    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [chessboard_points] * len(valid_corners), valid_corners, (8, 6), None, None
    )
    # extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    return intrinsics, dist_coeffs, rms, corners, chessboard_points


def numero_imgs_optimo(corners, chessboard_points):
    # Calculate RMS for number of images
    image_counts = range(2, len(corners) + 1)
    rms_errors = []
    for i in image_counts:
        valid_corners = [cor[1] for cor in corners[:i] if cor[0]]
        valid_corners = np.asarray(valid_corners, dtype=np.float32)
        rms, _, _, _, _ = cv2.calibrateCamera(
            [chessboard_points] * len(valid_corners), valid_corners, (8, 6), None, None
        )
        rms_errors.append(rms)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(image_counts, rms_errors, marker="o")
    plt.title("RMS Reprojection Error vs. Number of Images")
    plt.xlabel("Number of Images Used")
    plt.ylabel("RMS Reprojection Error")
    plt.grid(True)
    plt.show()
