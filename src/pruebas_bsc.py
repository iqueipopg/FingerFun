import cv2
import time
import os
import shutil
import numpy as np
import libreria.funciones as f
from libreria.constants import *


if __name__ == "__main__":
    puntuacion_maxima = 0
    rondas_jugadas = -1
    running = True
    primera_partida = True
    contador = 0
    nivel = 1

    # opencv_images = f.load_images_from_folder("figures")
    f.limpiar_imagenes()    
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

                if len(trajectory) > 125:
                    trajectory.pop(0)

            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

        # Mostrar el frame
        cv2.imshow("Shape Detection", frame)




        # Detectar teclas
        key = cv2.waitKey(1) & 0xFF

        # INICIAR PARTIDA
        if key == ord("a") and primera_partida == True: # Iniciar partida con 'a'
            secuencia = f.inicio_partida(running, opencv_images, names, puntuacion_maxima, rondas_jugadas)
            contador = 0
            primera_partida = False
            if secuencia == False:
                break 

        # SALIR DEL JUEGO
        if key == ord("q"): 
            break

        # BORAR TRAYECTORIA
        elif key == 32: 
            print("Trayectoria borrada")
            trajectory.clear()
        
        # PARCHE PARA PASAR DE NIVEL
        # elif key == ord("p"):  # que coincida
        #     print(f"Pasando al siguiente nivel: {nivel + 1}")
        #     nivel += 1
        #     secuencia = f.gestionar_niveles(nivel, opencv_images, names)
        #     print(secuencia)


        # VALIDAR FIGURA
        elif key == 13: # Guardar trayectoria con Enter
            print("Guardando y borrando trayectoria")
            time.sleep(0.5)
            cont += 1

            # Dibujar la trayectoria antes de guardar
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

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

            print(fig, secuencia[contador])
            print('nivel', nivel, 'contador', contador)

            if fig == secuencia[contador]:
                print((fig, secuencia[contador]), 'OK')
                contador += 1
            else:
                print((fig, secuencia[contador]), 'FALSE')
                print('DERROTA')
                if nivel > puntuacion_maxima:
                    puntuacion_maxima = nivel
                rondas_jugadas += 1

                # f.superponer_imagen_fullscreen(cap,"images/defeat.jpg") 
                time.sleep(2)
                running = False


            if contador + 1 == nivel:
                print(f'Nivel {nivel} finalizado.')
                nivel += 1
                print(f"Pasando al siguiente nivel: {nivel}")
                secuencia = f.gestionar_niveles(nivel, opencv_images, names)
                contador = 0
                print(secuencia)
                if nivel > puntuacion_maxima:
                    puntuacion_maxima = nivel
                rondas_jugadas += 1

            # Limpiar la trayectoria
            trajectory.clear()
            print(f"Figura detectada: {fig}")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
