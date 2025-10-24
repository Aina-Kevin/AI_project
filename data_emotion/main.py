
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Générateur d'images pour augmenter la diversité de l'ensemble de données
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, 
    rotation_range=20,  # Augmente la diversité avec des rotations
    width_shift_range=0.2,  # Décalages horizontaux
    height_shift_range=0.2,  # Décalages verticaux
    shear_range=0.2,  # Distorsions angulaires
    zoom_range=0.2,  # Zoom aléatoire
    horizontal_flip=True  # Retourner les images horizontalement
)

# Chargement des images d'entraînement
train_data = datagen.flow_from_directory(
    'data_emotion/train', 
    target_size=(48, 48), 
    batch_size=16, 
    color_mode='grayscale',  # Les images sont en niveaux de gris
    class_mode='categorical', 
    subset='training'
)

# Chargement des images de validation
val_data = datagen.flow_from_directory(
    'data_emotion/train',  # Doit être le même chemin que pour l'entraînement
    target_size=(48, 48), 
    batch_size=16, 
    color_mode='grayscale', 
    class_mode='categorical', 
    subset='validation'
)

# # Création du modèle CNN
# model = Sequential()

# # Bloc convolutionnel 1
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Bloc convolutionnel 2
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Bloc convolutionnel 3
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # Aplatir et connecter aux couches denses
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(train_data.num_classes, activation='softmax'))  # Couche de sortie
model = Sequential([
    # --- 1ère couche convolutionnelle ---
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    # --- 2ème couche convolutionnelle ---
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    # --- Couches de sortie ---
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')  # 7 émotions : happy, sad, etc.
])

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Résumé du modèle
model.summary()

# Entraînement du modèle
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=6,
)

 # Sauvegarder le modèle après l'entraînement
model.save('emotion_model_one.h5')  # Sauvegarde en format HDF5
print("Le modèle a été sauvegardé sous le nom 'emotion_model.h5'.")

# Évaluation du modèle
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
