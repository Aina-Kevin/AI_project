# import pandas as pd

# # Charger le fichier Excel
# df = pd.read_excel("Final.xlsx")

# # Remplacer les NaN et "Missing" par "Autres activités"
# df['Libellé NOPAMA'] = df['Libellé NOPAMA'].fillna("Autres services aux entreprises")
# df['Libellé NOPAMA'] = df['Libellé NOPAMA'].replace("SIFIM", "Autres services aux entreprises")


# df.to_excel("Final_2.xlsx", index=False)


# import pandas as pd

# # 📂 Charger le fichier Excel
# df = pd.read_excel("brute.xlsx")

# # 🧹 Remplacer les valeurs manquantes
# df['Libellé NOPAMA'] = df['Libellé NOPAMA'].fillna("Autres activités")

# # 📊 Compter les occurrences de chaque classe
# occurrences = df['Libellé NOPAMA'].value_counts()

# # 🖨️ Afficher les résultats
# print("Nombre d'occurrences pour chaque classe :\n")
# for i, (classe, count) in enumerate(occurrences.items(), 1):
#     print(f"{i}. {classe} : {count}")



# import pandas as pd

# # 📂 Charger le fichier Excel
# df = pd.read_excel(r"C:\Users\AINA KEVIN\Desktop\Project NIFF STAT\API\brute.xlsx")


# # 📑 Afficher la liste des colonnes
# print("Liste des colonnes :")
# print(df.columns.tolist())






import pandas as pd

# 📥 Charger le fichier Excel
df = pd.read_excel("C:/Users/AINA KEVIN/Desktop/Project NIFF STAT/API/brute.xlsx")

# 📤 Nettoyer : enlever les lignes sans libellé
df = df.dropna(subset=['Libellé NOPAMA'])

# 🧠 Grouper par Libellé
grouped = df.groupby('Libellé NOPAMA')

# ✍️ Créer un fichier texte
with open("résultat.txt", "w", encoding="utf-8") as f:
    for libelle, group in grouped:
        f.write(f"📌 Libellé NOPAMA : {libelle}\n")
        
        # Liste des NOPAMA
        nopama_list = group['NOPAMA'].dropna().unique().tolist()
        f.write(f"  🔹 NOPAMA     : {nopama_list}\n")
        
        # Liste des NOPAMA 2
        nopama2_list = group['NOPAMA 2'].dropna().unique().tolist()
        f.write(f"  🔸 NOPAMA 2   : {nopama2_list}\n")
        
        f.write("-" * 60 + "\n")

print("✅ Le fichier 'résultat.txt' a été généré avec succès.")
