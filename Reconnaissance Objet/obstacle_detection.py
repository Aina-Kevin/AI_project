# from ultralytics import YOLOWorld
# # Initialiser un modèle YOLO-World
# model = YOLOWorld("yolov8s-world.pt")  # ou choisir yolov8m/l-world.pt pour différentes tailles


# from ultralytics import YOLO
# # Charger votre modèle
# model = YOLO("yolo11n.pt")
# # Exporter en format NCNN
# model.export(format="ncnn")  # crée 'yolo11n_ncnn_model'

# from ultralytics import YOLO
# Charger un modèle YOLO11n pré-entraîné sur COCO
# model = YOLO("yolo11n.pt")
# Exécuter l'inférence sur une image
# results = model("path/to/bus.jpg")

import cv2
import torch
from ultralytics import YOLOWorld

# Charger le modèle YOLO-World
model = YOLOWorld("yolov8s-world.pt")  # 's', 'm', ou 'l' selon la performance souhaitée

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    # Convertir l'image en format compatible avec PyTorch
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des objets
    results = model.predict(img_rgb)

    # Parcourir les détections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées de la boîte
            conf = box.conf[0].item()  # Confiance de la détection
            label = result.names[box.cls[0].item()]  # Nom de l'objet

            # Dessiner le rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher la vidéo
    cv2.imshow("YOLO-World Detection", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
