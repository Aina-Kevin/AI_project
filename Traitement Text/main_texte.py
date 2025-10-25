
import cv2
import numpy as np
import easyocr  

def Lecture(image_path, widthImg, heightImg):
    # Redimensionner l'image
    img = cv2.resize(image_path, (widthImg, heightImg))

    # Convertir en niveaux de gris et flouter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equ, (3, 3), 0)
    corrected = cv2.addWeighted(equ, 1.5, blurred, -0.5, 0)

    # Détection des contours avec Canny
    median_value = np.median(corrected)
    low_thresh = int(max(0, 0.66 * median_value))
    high_thresh = int(min(255, 1.33 * median_value))
    edges = cv2.Canny(corrected, low_thresh, high_thresh)

    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fonction pour voir si deux contours sont proches
    def are_contours_connected(cnt1, cnt2, threshold=10):
        if len(cnt1) == 0 or len(cnt2) == 0:
            return False  # Éviter l'erreur si un contour est vide

        for point in cnt1:
            if len(point) < 1:
                continue  # Ignorer les points invalides

            pt = tuple(point[0])  # Convertir en tuple (x, y)
            
            if not (isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(coord, (int, float)) for coord in pt)):
                continue  # Vérifier que pt est bien un tuple de 2 nombres

            dist = cv2.pointPolygonTest(cnt2, pt, True)
            if dist >= -threshold:  # Si proche ou à l'intérieur du contour
                return True

        return False

    # Calculer la diagonale de l'image pour définir un seuil dynamique
    diagonale = np.sqrt(widthImg**2 + heightImg**2)

    # Définir le seuil des contours en fonction de la diagonale (par exemple, 10% de la diagonale)
    seuil_contour = diagonale * 0.1

    # Grouper les contours proches
    visited = set()
    groups = []

    for i, cnt1 in enumerate(contours):
        if cv2.arcLength(cnt1, True) > seuil_contour:
            if i in visited:
                continue  # Déjà dans un groupe
            group = [cnt1]
            visited.add(i)
        
            for j, cnt2 in enumerate(contours):
                if j != i and j not in visited and are_contours_connected(cnt1, cnt2):
                    if cv2.arcLength(cnt2, True) > seuil_contour:
                        group.append(cnt2)
                        visited.add(j)
            groups.append(group)

    # Trouver le plus grand groupe de contours en additionnant leurs longueurs
    max_length = 0
    biggest_contour = None

    for group in groups:
        total_length = sum(cv2.arcLength(cnt, True) for cnt in group)
        if total_length > max_length:
            max_length = total_length
            biggest_contour = np.vstack(group)  # Fusionner les contours

    # Dessiner uniquement le plus grand contour fusionné
    # Vérifier si un plus grand contour a été trouvé
    if biggest_contour is not None:
        x, y, w, h = cv2.boundingRect(biggest_contour)  # Obtenir les coordonnées et dimensions
        cropped_img = img[y:y+h, x:x+w]  # Découper l'image en fonction du contour
        cv2.imshow("Image découpée", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Utiliser EasyOCR pour lire le texte sur l'image découpée
    reader = easyocr.Reader(["fr"], gpu=False)
    results = reader.readtext(cropped_img)

    # Extraire le texte
    text_all = " ".join([text for (_, text, _) in results])
    
    print(text_all)
    return text_all

# Capture vidéo
cap = cv2.VideoCapture(0)
alpha =1.5
beta = 30
# image = cv2.convertScaleAbs(frame,alpha=alpha, beta = beta)
heightImg = 650
widthImg = 950
cap.set(cv2.CAP_PROP_FRAME_WIDTH,650)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,950)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # image = cv2.convertScaleAbs(frame,alpha=alpha, beta = beta)
    # frame = cv2.flip(frame, 1)
    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        
        Lecture(frame, widthImg, heightImg)
        break

cap.release()
cv2.destroyAllWindows()
