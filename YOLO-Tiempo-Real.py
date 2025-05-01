# YOLO-Tiempo-Real.py

from ultralytics import YOLO
import cv2
import math

# Cargamos el modelo YOLO pre-entrenado
model = YOLO("yolo-Weights/yolov8n.pt")

# Clases del modelo COCO (parcial)
classNames = model.names

# Inicializar cámara
captura = cv2.VideoCapture(0)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not captura.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

while True:
    success, img = captura.read()
    if not success:
        print("⚠️ Error al capturar el frame.")
        break

    # Inference sin stream para evitar errores múltiples de ventana
    results = model(img, stream=False)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            class_name = classNames[cls] if cls < len(classNames) else f"id:{cls}"
            label = f"{class_name} ({conf:.2f})"

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Detección en Vivo - YOLOv8", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
