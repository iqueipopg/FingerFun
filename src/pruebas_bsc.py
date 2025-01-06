import cv2
import time
import os
import shutil
import numpy as np
import libreria.funciones as f
from libreria.constants import *

# Asegúrate de que la cámara esté inicializada
cap = cv2.VideoCapture(0)  # Usar la cámara por defecto
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()


if __name__ == "__main__":
    puntuacion_maxima = 0
    rondas_jugadas = -1
    desbloquedao = False
    vidas = 5  # Número de vidas
    running = True
    primera_partida = True
    contador = 0
    nivel = 1
    secuencia = []  # Para almacenar las secuencias de figuras a detectar

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

        # Mostrar el nivel en la parte derecha de la pantalla
        nivel_texto = f"Nivel: {nivel}"
        (w, h), _ = cv2.getTextSize(nivel_texto, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(
            frame,
            nivel_texto,
            (frame.shape[1] - w - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Mostrar los corazones (vidas) en la parte superior izquierda
        for i in range(vidas):
            f.draw_heart(
                frame, 10 + (i * 40), 10
            )  # Posición ajustada a la esquina superior izquierda

        # Mostrar el frame con el nivel y las vidas
        cv2.imshow("Shape Detection", frame)

        # Detectar teclas
        key = cv2.waitKey(1) & 0xFF

        # INICIAR PARTIDA
        if key == ord("a") and primera_partida == True and desbloquedao == True:  # Iniciar partida con 'a'
            secuencia = f.inicio_partida(
                running, opencv_images, names, puntuacion_maxima, rondas_jugadas
            )
            contador = 0
            primera_partida = False
            if secuencia == False:
                break

        # SALIR DEL JUEGO
        if key == ord("q"):
            break

        # DESBLOQUEAR JUEGO
        if key == ord("b"):
            
            passwords = f.obtener_paths("passwords")
            print(passwords)
            usuario = f.detectar_contraseñas(passwords, frame)
            if usuario != None:
                print(f'¡Bienvenido a FingerFun {mapeo_nombres[usuario]}!')
                desbloquedao = True
            else:
                print("Contraseña incorrecta. ")

        # BORAR TRAYECTORIA
        elif key == 32 and desbloquedao == True:  # Borrar trayectoria con espacio
            print("Trayectoria borrada")
            trajectory.clear()

        elif key == 13 and desbloquedao == True:  # Guardar trayectoria con Enter
            print("Guardando y borrando trayectoria")
            time.sleep(0.5)

            cont = rondas_jugadas + 1
            figura_detectada = f.guardar_resultados(
                frame, cont, secuencia, contador, nivel
            )

            # Validar la figura
            continuar, nivel, contador, secuencia = f.gestionar_rondas(
                figura_detectada, secuencia, contador, nivel
            )

            if not continuar:  # Si el jugador falla
                vidas -= 1  # Restar una vida
                print(f"Has fallado. Te quedan {vidas} vidas.")

                # Si el jugador se queda sin vidas
                if vidas == 0:
                    print("¡Has perdido todas las vidas! Game Over.")
                    break  # Terminar el juego

            # Limpiar la trayectoria
            trajectory.clear()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
