from flask import Flask, request, jsonify

app = Flask(__name__)

# Exemple de "base de données" simple (liste d'activités)
activites = [
    Libelle Nopama 
]

@app.route("/recherche_activite", methods=["GET"])
def recherche_activite():
    query = request.args.get("q", "").lower()  # récupère le paramètre 'q', converti en minuscules
    if not query:
        return jsonify({"erreur": "Merci de fournir un paramètre de recherche 'q'"}), 400

    # Filtrer les activités contenant le mot-clé dans le nom ou le type
    resultats = [act for act in activites if query in act["nom"].lower() or query in act["type"].lower()]

    return jsonify({"resultats": resultats})

if __name__ == "__main__":
    app.run(debug=True)
