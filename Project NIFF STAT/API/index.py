from flask import Flask, request, jsonify
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import json
import joblib

# Télécharger les ressources nécessaires pour nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('french'))  # Utilisation des stopwords en français

# Charger le modèle et le label encoder
# model = joblib.load("Model_Final.joblib")

model_path = r"D:\Burr_2\Project NIFF STAT\API\Model_Final.joblib"
model = joblib.load(model_path)
label_path = r"D:\Burr_2\Project NIFF STAT\API\Label.joblib"
data_path = r"D:\Burr_2\Project NIFF STAT\API\\data.json"
label_encoder = joblib.load(label_path)

abbreviations = {
    "vent": "vente",
    "tts": "toutes",
    "consom": "consommables",
    "recher": "recherche",
    "instal": "installation",
    "syndic": "syndicat ou syndic",
    "sav": "service après-vente",
    "stes": "sociétés",
    "sces": "services",
    "cciales": "commerciales",
    "representat": "représentation",
    "sci": "Société Civile Immobilière",
     "elect": "électricité",
    "fret": "fret (transport de marchandises)",
    "libr": "librairie",
    "equip": "équipement",
    "transforma": "transformation",
    "promot": "promotion",
    "conditionmt": "conditionnement",
    "fabric": "fabrication",
    "transform": "transformation",
    "confect": "confection",
    "op indl": "opération industrielle",
    "comm transf": "commerce de transformation",
    "fav": "faveur (ou favorable, selon le contexte)",
    "jurd": "juridique",
    "vt": "vente",
    "serigr": "sérigraphie",
    "medic": "médical",
    "spor": "sportif",
    "fnrs": "fournisseurs",
    "frntures": "fournitures",
    "frn": "fourniture",
    "iec": "import-export commercial",
    "sr": "supermarché / surface de vente",
    "gde surf": "grande surface",
    "distrib": "distribution",
    "mses": "marchandises",
    "emplact": "emplacement",
    "ccle": "clé commerciale (ou centre commercial)",
    "transf": "transformation",
    "fabr": "fabrication",
    "commerc": "commerce",
    "detailsa": "détail / détail SA (société anonyme)",
    "pcs": "pièces",
    "info": "informatique",
    "fseur": "fournisseur",
    "tdr": "termes de référence",
    "sce": "service",
    "bpo": "business process outsourcing",
    "epi": "équipement de protection individuelle",
    "sce operat": "service opérationnel",
    "f": "fonds",
    "cpble": "comptable",
    "fin": "finance",
    "cier": "financier",
    "info": "informatique",
    "entre": "entreprise",
    "otre": "autres",
    "ss": "sous",
    "e": "et",
    "ce": "centre",
    "w": "work (travail ou lieu de travail)",
    "s ces": "services",
    "entr gen travx": "entreprise générale de travaux",
    "commerc": "commerce",
    "cred": "crédit",
    "amenagt": "aménagement",
    "fournis": "fournisseur",
    "habill": "habillement",
    "menag": "ménager",
    "trav": "travaux",
    "reha": "réhabilitation",
    "consn": "consommation",
    
    "ts": "tous",
    "det": "détachées",
    "ts typ": "tous types",
    "pces comp": "pièces de compétition",
    "frnt": "fourniture",
    "out": "outillage",
    "veh": "véhicules",
    "e r": "équipements et réparations",
    "frns": "fournisseurs",
    "frnit": "fourniture",
    "bur": "bureau",
    "pces det": "pièces détachées",
    "quinc hab": "quincaillerie et habillement",
    "mat": "matériel",
    "cons info": "consommables informatiques",
    "art frnt": "articles de fourniture",
    "sporti": "sportif",
    "pdts d_ nt": "produits d'entretien",
    "format": "formation",
    "repar": "réparation",
    "ent": "entretien",
    "rehabil": "réhabilitation",
    "bati": "bâtiment",
    "commis": "commerce",
    "transp": "transport",
    "prest": "prestations",
    "sces denr": "services de denrées",
    "alim": "alimentaires",
    "sce trait": "service de traitement",
    "loca mat": "location de matériel",
    "prod alim": "produits alimentaires",
    "collo": "colloques",
    "semin": "séminaires",
    "atel": "ateliers",
    "org fete cerem": "organisation de fêtes et cérémonies",
    "rech": "recherche",
    "telpq": "télécommunication",
    "btp": "bâtiment et travaux publics",
    "BTP": "bâtiment et travaux publics",
    "route piste": "routes et pistes",
    "ust ": "ustensiles de cuisine",
    "fournisseur fourniture": "fournisseur de fournitures",
    "gles": "générales",
    "recharge teleph": "recharge téléphonique",
    "mat tech": "matériel technique",
    "EGC": "éducation et formation continue",
    "garag": "garage",
    "org evenement": "organisation d'événements",
    "serigraphie": "sérigraphie",
    "bur etude": "bureau d'étude",
    "consultance": "consultation",
    "frais colloq": "frais de colloques",
    "seminair": "séminaires",
    "conferenc": "conférences",
    "fournisseur ses": "fournisseur de services",
    "ameublement": "ameublement",
    "maint" : "maintenance",
    "rep": "réparation",
    "commerce_negoces_services": "commerce, négoces et services",
    "importation_materiels": "importation de matériels",
    "equipement_informatique": "équipements informatiques",
    "fourniture_installation": "fourniture et installation",
    "reseautiques": "réseautiques",
    "cctv": "circuit de télévision fermé",
    "energies_renouvelables": "énergies renouvelables",
    "solaire": "solaire",
    "telecommunications": "télécommunications",
    "gsm": "téléphonie mobile",
    "fh vsat": "fourniture de services VSAT",
    "pdts" : "produit",
    "pdt" : "produit",
    "gle" : "general",
    "fsseur": "fournisseur",
    "mmb": "matériel médical",
    "hab": "habillement",
    "friperies": "friperies",
    "quinc": "quincaillerie",
    "ese": "entreprise de service et équipement",
    "gale": "générale",
    "pces": "pièce",
    "art": "article",
    "rep": "réparation",
    "instrumt": "instrument",
    "agri": "agricole",
    "march": "marchandises",
    "vet": "vétérinaire",
    "creat": "création",
    "acquisit": "acquisition",
    "exploitat": "exploitation",
    "com": "commercial",
    "det": "détachées",
    "maint": "maintenance",
    "rehab": "réhabilitation",
    "pharm": "pharmaceutiques",
    "vehic": "véhicules",
    "rev": "revente",
    "sce": "services ",
    "prest": "prestation",
    "evaluat": "évaluation",
    "mg": "magasin",
    "ouvr": "ouvrages",
    "hydr": "hydrauliques",
    "metal": "métalliques",
    "scol": "scolaire",
    "meteor": "météorologique",
    "tel": "téléphone",
    "xts": "exportations",
    "alim": "alimentation",
    "agro": "agroalimentaire / agro-industriel",
    "indus": "industriel",
    "veto": "vétérinaire",
    "elevage": "élevage",
    "derives": "dérivés",
    "xts mat": "exportation de matières",
    "xts equip": "exportation d'équipements",
    "tech": "technique",
    "mecaniq": "mécanique",
    "fab industr": "fabrication industrielle",
    "import export": "importation et exportation",
    "comm gros details": "commerce en gros et détail",
    "savons": "savons",
    "xts cosmetq hygieniques": "cosmétiques et produits hygiéniques",
    "agro alimentaire": "produits agro-alimentaires",
    "paramedicaux": "produits paramédicaux",
    "materiaux d emballage": "matériaux d'emballage",
    "bureautiques": "produits bureautiques",
    "mat prem": "matières premières",
    "multi packaging": "emballages multiples",
    "ttes" : "toutes",
    "m ses" : "marchandises",
    "quinca" : "quincallerie",
    "frniture": "fourniture",
    "frsns": "fournisseurs",
    "vte": "vente",
    "logt": "logement",
    "mob": "mobilier",
    "agce": "agence",
    "bat": "bâtiment",
    "entret": "entretien",
    "construct": "construction",
    "prestat": "prestations",
    "g": "générale",
    "ge": "générale",
    "interm": "intermédiaire",
    "prod": "produits",
    "fourn": "fourniture",
    "cons": "consommables",
    "construc": "construction",
    "pharma": "pharmaceutique",
    "teleph": "téléphonie",
    "elec": "électricité",
    "techn": "technique",
    "stte":	"société",
    "npks": "engrais NPK (azote, phosphore, potassium, soufre)", 
    "contra": "contrôle",
    "ile": "de la",
    "qualita": "qualité",
    "quantita": "quantité",
    "pa troliers": "produits pétroliers",
    "vtes": "ventes", "general constr": "construction générale", "mses générales": "marchandises générales",
    "matériel electrom": "matériel électroménager",
    "ameub": "ameublement",
    "meub": "meubles",  "deco": "décoration",
    "trait collecte": "traitement et collecte",
    "pdtsloc": "produits locaux", "div":"divers",
    "desinf": "désinfection",
    "sco": "scolaire",
    "prestat": "prestations",
    "bat": "bâtiment",
    "impri": "imprimerie",
    "conso": "consommables",
    "ext": "extérieur",
    "relat": "relations",
    "spec": "spécialisé",
    "appar": "appareils",
    "constr": "construction",
    "ameub": "ameublement",
    "adm": "administratif",
    "doc": "documents",
    "labo": "laboratoire",
    "div": "divers",
    "outi": "outillage",
    "cce": "centre de commerce et d'échange",
    "oeuv": "œuvres",
    "cstr": "construction",
    "bat": "bâtiment",
    "cst": "construction",
    "frs": "fournisseurs",
    "frt": "fret",
    "mob": "mobilier",
    "cons": "consommables",
    "mse": "marchandises et services",
    "qnc": "quincaillerie",
    "trait": "traitement",
    "gar": "garage",
    "t": "technique",
    "locat": "location",
    "voit": "voiture",
    "logi": "logistique",
    "inf": "informatique",
    "vis": "visioconférence",
    "tec": "technique",
    "rest": "restauration",
    "intr": "intra",
    "tte": "toute",
    "exp": "exportation",
    "loc": "local",
    "div": "divers",
    "plantat": "plantation",
    "produc": "production",
    "prem": "premières",
    "veg": "végétaux",
    "energ": "énergie",
    "dvlpt": "développement",
    "prestat": "prestation",
    "prestat intellectuelle": "prestation intellectuelle",
    "egc": "éducation et formation continue",
    "vn": "vente",
    "tl": "télécommunication",
    "seri": "sérigraphie",
    "tablissements": "établissements",
    "four": "fourniture",
    "m": "matériel",
    "g": "général",
    "fnrt": "fourniture",
    "prod": "produits",
    "habil": "habillement",
    "techniq": "technique",
    "inform": "informatique",
    "pneumatiq": "pneumatique",
    "teleph": "téléphonie",
    "habillt": "habillement",
    "fournitur": "fourniture",
    "electriq": "électrique",
    "constru": "construction",
    "hebergt": "hébergement",
    "intellec": "intellectuelle",
    "logt": "logement",
    "fourni": "fourniture",
    "fournit": "fourniture",
    "transit": "transit",
    "y afferents": "différents", 
    "consom": "consommables",
    "assist": "assistance",
    "ameubl": "ameublement",
    "publact": "publicité et actions",
    "immob": "immobilier",
    "exploit": "exploitation",
    "equipmt": "équipement",
    "act": "activités",
    "logs": "logistiques",
    "btb": "business to business",
    "trans": "transport",
    "app": "appareils",
    "ett": "établissement",
    "org": "organisation",
    "even": "événement",
    "spec": "spécialisé",
    "impr": "imprimerie",
    "secu": "sécurité",
    "incend": "incendie",
    "civ": "civil",
    "equi": "équipements",
    "telec": "télécommunication",
    "coll": "collectivités",
    "roul": "roulant",
    "comb": "combustible",
    "pts": "points",
    "pr": "prestations",
    "acc": "accueil",
    "restau": "restauration",
    "log": "logement",
    "p": "personnel",
    "dvpt": "développement",
    "eco": "économie",
    "attrac": "attraction",
    "fabri": "fabrication",
    "chim": "chimie",
    "const": "construction",
    "obj": "objets",
    "scieries": "scieries",
    "demen": "déménagement",
    "rngestion": "regestion",  # Peut-être une faute, à confirmer
    "rnfabrication": "renouvellement de fabrication",
    "rnrn": "renouvellement",  # probable redondance
    "rnlocation": "renouvellement de location",
    "log": "logistique",
    "med": "médical",
    "ord": "ordinaire",
    "mili": "militaire",
    "attrb": "attributs",  # ou "attributions", à confirmer selon le contexte
    "pd": "produits",
    # "ys": "services",  # peut-être mal orthographié ou "systèmes"
    "sol": "solaire",
    "matériel naval": "matériel naval",
    "mfb": "ministère des finances et du budget",
    "sg": "secrétariat général",
    "dgi": "direction générale des impôts",
    "dco": "direction du contrôle",
    "drivak": "direction de la recherche, innovation, valorisation et acquisition de connaissances",  # hypothèse à confirmer selon contexte
    "informatiq": "informatique",
    "matl": "matériel"
}
# 🔁 Remplacer les abréviations


# Charger le fichier JSON
with open(data_path, "r", encoding="utf-8") as file:
    data_code = json.load(file)
def replace_abbreviations(text):
    if not isinstance(text, str):
        return text
    for abbr, full in sorted(abbreviations.items(), key=lambda x: -len(x[0])):
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full, text)
    return text

# 🧼 Nettoyage du texte
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)                
    text = re.sub(r'[^\w\s]', ' ', text)            
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# 🧠 Tokenisation + lemmatisation
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

@app.route('/')
def home():
    return "Bienvenue sur mon API de traitement de texte avec NLTK"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'texte' not in data:
        return jsonify({'error': 'Champ "texte" manquant'}), 400

    raw_text = data['texte']

    # Étapes de traitement du texte
    text = replace_abbreviations(raw_text)
    cleaned_text = clean_text(text)

    try:
        # Prédiction
        prediction = model.predict([cleaned_text])[0]
        label = label_encoder.inverse_transform([prediction])[0]

        # Recherche du libellé correspondant dans le fichier JSON
        resultat = next((item for item in data_code["categories"] if item["Libellé NOPAMA"] == label), None)

        if not resultat:
            return jsonify({
                'texte_nettoye': cleaned_text,
                'prediction': label,
                'message': 'Libellé non trouvé dans le fichier JSON'
            }), 200

        return jsonify({
            'texte_nettoye': cleaned_text,
            'prediction': label,
            'NOPAMA': resultat["NOPAMA"],
            'NOPAMA 2': resultat["NOPAMA 2"]
        }), 200

    except Exception as e:
        return jsonify({'error': f'Erreur pendant la prédiction : {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)






