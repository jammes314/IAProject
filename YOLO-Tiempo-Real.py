# YOLO-Tiempo-Real.py

from ultralytics import YOLO
import cv2
import math

# Cargamos el modelo YOLO pre-entrenado (versión ligera)
model = YOLO("yolo-Weights/yolov8n.pt")

# Clases del modelo COCO (algunas)
classNames = ["person", "bicycle", "car", "motorbike", "fire hydrant", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Abrimos la cámara (índice 0)
captura = cv2.VideoCapture(0)
captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Verificamos si la cámara está abierta correctamente
if not captura.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

# Creamos una ventana solo una vez
cv2.namedWindow("Detección en Vivo - YOLOv8", cv2.WINDOW_NORMAL)

while True:
    success, img = captura.read()
    if not success:
        print("No se pudo acceder a la cámara.")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else f"id:{cls}"
            print(f"Confianza: {confidence}, Clase: {class_name}")

            cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

    cv2.imshow("Detección en Vivo - YOLOv8", img)

    # Salimos del bucle si se presiona la tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberamos los recursos
captura.release()
cv2.destroyAllWindows()
