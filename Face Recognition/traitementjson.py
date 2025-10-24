import json
file_path = "./Computer Vision/Module Face Recognition/data_file.json"
def load_data():
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
def save_data(data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def add_new_person(new_id, new_name, new_description, face_encoding):
    data = load_data()
    
    # Ajouter les infos de la nouvelle personne
    data["All_information"][str(new_id)] = {
        "name": new_name,
        "description": new_description
    }

    # Ajouter l'encodage du visage
    data["Face_Code"]["Coding_Face"].append(face_encoding)

    # Mettre à jour l'ID pour la prochaine personne
    data["Conservation_ID"]["i"] = new_id  

    save_data(data)
    print("Nouvelle personne ajoutée avec succès.")

