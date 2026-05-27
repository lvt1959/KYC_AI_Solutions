import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

# --- 1. CONFIGURATION ---
# Assure-toi que ce chemin pointe vers le dossier généré à l'étape précédente
DATA_DIR = 'dataset_final/'
IMG_SIZE = (224, 224)  # Taille standard pour les réseaux profonds
BATCH_SIZE = 64  # Avec une RTX 5090, tu peux largement encaisser des lots de 64 images
EPOCHS = 50  # Le nombre maximum de passages sur le dataset

print(f"TensorFlow détecte les GPUs suivants : {tf.config.list_physical_devices('GPU')}")

# --- 2. CHARGEMENT DES DONNÉES (Train & Validation Split) ---
# Keras va automatiquement lire tes dossiers, déduire les classes et créer les lots
print("Chargement du dataset d'entraînement...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,  # On garde 20% des images pour vérifier si le modèle apprend bien
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'  # On utilise le format 'categorical' car on a plusieurs classes
)

print("Chargement du dataset de validation...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# 1. On lit les classes AVANT d'optimiser
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Classes détectées : {class_names}")

# 2. ON OPTIMISE ENSUITE
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 2. SEULEMENT APRÈS, on applique l'optimisation des performances
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. AUGMENTATION DE DONNÉES INTEGREE ---
# Ces transformations s'appliqueront à la volée pendant l'entraînement, directement sur le GPU
# --- 3. AUGMENTATION DE DONNÉES AGRÉSSIVE (Mode From Scratch) ---
data_augmentation = tf.keras.Sequential([
    # 1. Déformations spatiales (Mimer un utilisateur qui cadre très mal)
    layers.RandomTranslation(height_factor=0.15, width_factor=0.15),  # Décale l'image jusqu'à 15% sur les côtés
    layers.RandomRotation(0.1),  # Rotation jusqu'à +/- 36 degrés
    layers.RandomZoom(height_factor=(-0.2, 0.2)),  # Dézoom et zoom aléatoires

    # Attention : On n'utilise surtout pas layers.RandomFlip() ici.
    # Mettre un document d'identité en miroir détruit la structure du texte et les repères visuels réels.

    # 2. Altérations lumineuses et colorimétriques (Mimer de mauvaises webcams/smartphones)
    layers.RandomContrast(factor=0.3),
    layers.RandomBrightness(factor=0.3),

    # 3. Bruit de capteur
    layers.GaussianNoise(0.2)  # On double l'intensité du bruit par rapport à ta version initiale
], name="Heavy_Data_Augmentation")
# --- 4. L'ARCHITECTURE DU CNN (Optimisée pour ta RTX 5090) ---
model = models.Sequential(name="KYC_Document_Classifier")

# Entrée et Augmentation
model.add(layers.InputLayer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(data_augmentation)

# Normalisation des pixels (de 0-255 à 0-1)
model.add(layers.Rescaling(1. / 255))

# Bloc Convolutif 1
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.BatchNormalization())  # Accélère l'apprentissage et stabilise le réseau
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Bloc Convolutif 2
model.add(layers.Conv2D(128, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Bloc Convolutif 3
model.add(layers.Conv2D(256, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Bloc Convolutif 4 (On va profond car on a le GPU pour)
model.add(layers.Conv2D(512, (3, 3), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Tête de classification moderne (GlobalAveragePooling remplace Flatten pour éviter l'overfitting)
model.add(layers.GlobalAveragePooling2D())

# Couche dense finale
model.add(layers.Dense(512))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))  # On coupe 50% des connexions pour forcer la généralisation

# Couche de sortie
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

# --- 5. COMPILATION ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 1e-4 au lieu de 1e-3
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. CALLBACKS (Les sécurités de l'entraînement) ---
# EarlyStopping : Arrête l'entraînement si la précision ne s'améliore plus après 8 époques
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# ModelCheckpoint : Sauvegarde le meilleur modèle sur le disque (format .keras)
checkpoint = callbacks.ModelCheckpoint(
    filepath='kyc_classifier_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # On divise le taux d'apprentissage par 5
    patience=3,  # S'il ne s'améliore pas pendant 3 époques
    min_lr=1e-6,  # Limite basse pour ne pas s'arrêter complètement
    verbose=1
)

# --- 7. L'ENTRAÎNEMENT ---
print("\n🚀 Lancement de l'entraînement sur le pod GPU...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

print("\n✅ Entraînement terminé ! Le meilleur modèle a été sauvegardé sous 'kyc_classifier_best.keras'.")