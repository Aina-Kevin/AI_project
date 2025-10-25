# import cv2 
# from ultralytics import YOLO
# model = YOLO('yolov10n.pt')
# coord = {
#     'gauche':[0,200],
#     'milieu':[200,400],
#     'droite':[400,600]
# }
# cap=cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Erreur : Impossible d'accéder à la caméra.")
#     exit()
# while True :
#     succes, imagenorme  = cap.read()
#     image=cv2.resize(imagenorme,(600,600))
#     image=cv2.flip(image,1)
#     results = model.track(image,persist=True)
#     print(results)
#     frame_ = results[0].plot()
#     # cv2.line(frame_,(350,0),(350,700),(0,0,255),0.5)
#     # cv2.line(frame_,(350,0),(350,700),(0,0,255),0.5)
#     cv2.imshow("Image",frame_)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()   


# # def postion(image):
    


# import cv2
# from ultralytics import YOLO

# # Charger le modèle YOLOv8 Nano (ou YOLOv10 si disponible)
# model = YOLO('yolov10n.pt')  # Assure-toi que ce modèle est correct

# # Définir les zones pour la position des objets
# coord = {
#     'gauche': [0, 200],
#     'milieu': [200, 400],
#     'droite': [400, 600]
# }

# # Accéder à la webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Erreur : Impossible d'accéder à la caméra.")
#     exit()

# while True:
#     success, imagenorme = cap.read()
#     if not success:
#         break  # Arrêter si la caméra ne fonctionne pas

#     # Redimensionner et retourner l'image pour correspondre aux coordonnées
#     image = cv2.resize(imagenorme, (600, 600))
#     image = cv2.flip(image, 1)

#     # Détection et tracking des objets
#     results = model.track(image, persist=True)

#     # Dessiner les lignes de séparation
#     cv2.line(image, (200, 0), (200, 600), (0, 255, 0), 2)  # Gauche
#     cv2.line(image, (400, 0), (400, 600), (0, 255, 0), 2)  # Droite

#     # Parcourir les objets détectés
#     for result in results:
#         for obj in result.boxes:
#             x1, y1, x2, y2 = obj.xyxy[0]  # Coordonnées de la boîte englobante
#             largeur_objet = x2 - x1  # Largeur de l'objet

#             # Déterminer la position
#             centre_objet = (x1 + x2) / 2
#             if centre_objet < coord['gauche'][1]:
#                 position = "Gauche"
#             elif centre_objet > coord['droite'][0]:
#                 position = "Droite"
#             else:
#                 position = "Milieu"

#             # Afficher la boîte et la position
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#             cv2.putText(image, f"{position}", (int(x1), int(y1) - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#     # Afficher l'image
#     cv2.imshow("Image", image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
from ultralytics import YOLO

# Charger le modèle YOLOv8n
model = YOLO("yolov10n.pt")

# Accéder à la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break
    frame=cv2.flip(frame,1)
    # Récupérer la largeur de l'image
    frame_width = frame.shape[1]

    # Détection et suivi des objets
    results = model.track(frame, persist=True)

    # Analyser les objets détectés
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées de la boîte
            track_id = int(box.id.item()) if box.id is not None else -1  # Récupérer track_id
            
            # Calcul du centre de l'objet
            object_center = (x1 + x2) // 2

            # Déterminer la position de l'objet
            if object_center < frame_width * 0.33:
                position = "Gauche"
            elif object_center < frame_width * 0.66:
                position = "Milieu"
            else:
                position = "Droite"

            # Dessiner la boîte et afficher l'ID + Position
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id} - {position}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# import cv2
# from ultralytics import YOLO

# # Charger le modèle YOLOv8n
# model = YOLO("yolov8n.pt")

# # Accéder à la webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Erreur : Impossible d'accéder à la caméra.")
#     exit()

# while True:
#     success, frame = cap.read()
#     if not success:
#         break

#     # Redimensionner et retourner l'image pour la cohérence
#     frame = cv2.resize(frame, (600, 600))
#     frame = cv2.flip(frame, 1)

#     # Détection et suivi des objets
#     results = model.track(frame, persist=True)

#     # Dessiner les boîtes englobantes avec `track_id`
#     if results[0].boxes:
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées de la boîte
#             track_id = int(box.id.item()) if box.id is not None else -1  # Récupérer le track_id
            
#             # Dessiner la boîte et afficher l'ID
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Afficher l'image
#     cv2.imshow("Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
