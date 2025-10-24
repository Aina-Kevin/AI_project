
import cv2
import face_recognition
import json
import os
import numpy as np
from traitementjson import *
# Chemins des fichiers
file_path = "./Computer Vision/Module Face Recognition/data_file.json"
output_dir = "./Computer Vision/Module Face Recognition/data/"
os.makedirs(output_dir, exist_ok=True)

# Charger la base de données JSON
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

Identifiant = data.get("Conservation_ID")
Identifiant=Identifiant["i"]

# Fonction de reconnaissance de visage
def Reconnaissance(frame, Identifiant):
    margin = 50
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations:
        top, right, bottom, left = face_locations[0]

        # Appliquer la marge
        top = max(top - margin, 0)
        right = min(right + margin, frame.shape[1])
        bottom = min(bottom + margin, frame.shape[0])
        left = max(left - margin, 0)

        # Obtenir l'encodage directement depuis l'image d'origine
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
        if not face_encoding:
            print("Impossible d'encoder le visage")
            return
        face_encoding = face_encoding[0]

        # Charger les données JSON
        data = load_data()
        face_base = data["Face_Code"]["Coding_Face"]

        # Vérifier si la personne est connue
        for i, face in enumerate(face_base):
            if face_recognition.compare_faces([face], face_encoding)[0]:
                print("Visage Trouvé")
                id_found = i + 1
                Nom = data["All_information"][str(id_found)]["name"]
                Description = data["All_information"][str(id_found)]["description"]
                print(f"Nom : {Nom}")
                print(f"Description : {Description}")
                return

        # Ajouter une nouvelle personne
        print("Visage Non Trouvé")
        permission = input("Tu veux ajouter ? (o/n) ").strip().lower()
        if permission == "o":
            new_name = input("Nom de la personne : ")
            new_description = input("Description : ")
            add_new_person(str(int(Identifiant) + 1), new_name, new_description, face_encoding.tolist())
 # Convertir en liste avant d'enregistrer

# Capture vidéo
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    Reconnaissance(frame, Identifiant)
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





















# import cv2
# import numpy as np
# import face_recognition
# import threading
# import json
# import os
# # Charger les données une seule fois en mémoire
# file_path = "./Computer Vision/Module Face Recognition/data_file.json"
# # output_dir = "./Computer Vision/Module Face Recognition/data/"
# # os.makedirs(output_dir, exist_ok=True)

# # Charger la base de données JSON
# with open(file_path, "r", encoding="utf-8") as file:
#     data = json.load(file)

# # Charger la base de visages une seule fois
# # data = load_data()
# face_base = np.array(data["Face_Code"]["Coding_Face"])  # Optimisation NumPy

# def Reconnaissance(frame, Identifiant):
#     global face_base

#     # Convertir en RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Détecter les visages
#     face_locations = face_recognition.face_locations(rgb_frame, model="hog")

#     if len(face_locations) > 0:
#         # Encoder le premier visage détecté
#         face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]

#         # Calculer la distance avec les visages connus
#         face_distances = face_recognition.face_distance(face_base, face_encoding)
#         best_match_index = np.argmin(face_distances)

#         # Seuil de reconnaissance
#         if face_distances[best_match_index] < 0.6:
#             print("Visage Trouvé - ID:", best_match_index + 1)
#             Identifiant[0] = best_match_index + 1  # Stocker l'identifiant trouvé

# # Capture vidéo
# cap = cv2.VideoCapture(0)  # 0 = Webcam

# # Stocker l'ID reconnu
# Identifiant = [None]

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Lancer la reconnaissance en parallèle
#     thread = threading.Thread(target=Reconnaissance, args=(frame, Identifiant))
#     thread.start()

#     # Affichage
#     cv2.imshow("Reconnaissance Faciale", frame)

#     # Quitter avec 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
