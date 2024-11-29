import cv2
import time
import os
import shutil
import numpy as np
import libreria.funciones as f
from libreria.constants import *


if __name__ == "__main__":

    # opencv_images = f.load_images_from_folder("figures")
    directorio = "images/output"
    if os.path.exists(directorio) and os.path.isdir(directorio):
        try:
            shutil.rmtree(directorio)
            print(f"El directorio {directorio} ha sido eliminado.")
        except OSError as e:
            print(f"Error al eliminar el directorio: {e}")
    else:
        print(f"El directorio {directorio} no existe o no es un directorio.")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
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
        elif key == ord("p"):  # que coincida
            print(f"Pasando al siguiente nivel: {nivel + 1}")
            nivel += 1
            f.gestionar_niveles(nivel, opencv_images)
        elif key == 13:
            print("Guardando y borrando trayectoria")
            time.sleep(0.5)
            cont += 1

            # Dibujar la trayectoria antes de guardar
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

            # path = os.getcwd()
            # dest = os.path.join(path, '..', 'images')

            if not os.path.exists("images/output"):
                os.makedirs("images/output/screenshots")
                os.makedirs("images/output/masks")
                os.makedirs("images/output/shapes")
            # Guardar la imagen con trayectoria

            cv2.imwrite(f"images/output/screenshots/screenshot{cont}.png", frame)
            mask = f.create_mask(frame, lower_blue, upper_blue)
            cv2.imwrite(
                f"images/output/masks/mask{cont}.png",
                mask,
            )

            # Detectar las formas geométricas en la imagen guardada
            img_with_shapes, fig = f.detect_geometric_shapes(mask)
            cv2.imwrite(f"images/output/shapes/shape{cont}.png", img_with_shapes)

            # Limpiar la trayectoria
            trajectory.clear()
            print(f"Figura detectada: {fig}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
