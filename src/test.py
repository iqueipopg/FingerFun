import cv2
import os 

# FICHERO PARA GUARDAR IMÁGENES PARA CALIBRACIÓN DE CÁMARA

def main():
    cap = cv2.VideoCapture(0)
    contador = 0

    while True:
        ret, frame = cap.read()

        cv2.imshow('Camara', frame)

        key = cv2.waitKey(1)
    
        if key == 13:  
            os.makedirs('capturas', exist_ok=True)
            cv2.imwrite(f'capturas/imagen_{contador}.jpg', frame)
            contador += 1


        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
