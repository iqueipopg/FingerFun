import cv2

cap = cv2.VideoCapture(0)  # Usa la cámara por defecto
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Presiona 'q' para salir
        break
cap.release()
cv2.destroyAllWindows()
