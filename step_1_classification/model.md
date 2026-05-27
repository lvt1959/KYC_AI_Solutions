# Classificateur de Documents KYC - Documentation du Modèle

Ce dépôt contient un réseau de neurones convolutifs (CNN) sur mesure, entraîné de zéro (from scratch) pour classifier des documents d'identité en deux catégories principales : **Cartes d'identité (`id`)** et **Passeports (`passport`)**. 

Ce modèle a été conçu pour s'intégrer dans des pipelines d'automatisation KYC (Know Your Customer) à haut débit et optimisé pour fonctionner efficacement sur du matériel de pointe (GPU).

## 1. Jeu de Données & Prétraitement

### Source des données
Le modèle a été entraîné à partir des "templates" du dataset **MIDV-2020 (Mobile Identity Document Video Dataset)**. 
* **Nature des données :** Modèles synthétiques de haute qualité représentant divers passeports et cartes d'identité internationaux.
* **Volume :** 1 000 images au total.
* **Répartition :** 80% Entraînement (800 images) / 20% Validation (200 images), partitionnées avec une graine fixe (`seed=123`) pour garantir une reproductibilité parfaite.
* **Dimensions d'entrée :** Toutes les images sont redimensionnées à `224 × 224` pixels sur 3 canaux (RGB).

### Restructuration & Équilibrage des classes
La structure brute du dataset MIDV-2020 (divisée par pays et sous-types) a été aplatie et consolidée dans une architecture de classification binaire :
```text
dataset_final/
├── id/          # 565 images (classe majoritaire à environ 56,5%)
└── passport/    # 435 images
```
Pour éviter les conflits de noms lors de la consolidation, les fichiers ont été renommés dynamiquement en utilisant le préfixe de leur dossier parent (ex: `alb_id_00.jpg`), garantissant que les identifiants uniques restent intacts juste avant l'extension du fichier.

### Pipeline d'Augmentation de Données Massive
Puisque les templates synthétiques sont géométriquement parfaits et nets, un puissant pipeline de régularisation a été intégré directement dans le graphe de calcul pour simuler les défauts de capture du monde réel (smartphones bas de gamme, mauvais éclairage, mauvais cadrage) et empêcher le sur-apprentissage (overfitting) :

```python
data_augmentation = tf.keras.Sequential([
    # Déformations spatiales (Erreurs de cadrage et de manipulation)
    layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
    layers.RandomRotation(0.1),  # Jusqu'à ±36 degrés
    layers.RandomZoom(height_factor=(-0.2, 0.2)),
    
    # Altérations photométriques (Qualité de l'éclairage et du capteur)
    layers.RandomContrast(factor=0.3),
    layers.RandomBrightness(factor=0.3),
    layers.GaussianNoise(0.2)  # Simule le bruit numérique du capteur
])
```
*Note : Les retournements horizontaux/verticaux (flips) ont été strictement exclus afin de préserver les contraintes de mise en page textuelle inhérentes aux documents d'identité officiels.*

## 2. Architecture du Modèle

Le modèle est un réseau de neurones convolutifs (CNN) profond et sur mesure, construit séquentiellement avec TensorFlow/Keras. Il s'appuie sur des techniques de régularisation agressives (`BatchNormalization`, `Dropout` et `GlobalAveragePooling2D`) pour s'entraîner de manière fiable de zéro, même avec un volume de données limité.

| Couche (Type) | Dimension de Sortie | Paramètres | Rôle Fonctionnel |
| :--- | :--- | :--- | :--- |
| **`InputLayer`** | `(None, 224, 224, 3)` | 0 | Point d'entrée pour les images RGB. |
| **`Sequential (Augmentation)`** | `(None, 224, 224, 3)` | 0 | Applique des distorsions à la volée sur le GPU. |
| **`Rescaling (1./255)`** | `(None, 224, 224, 3)` | 0 | Normalise les pixels de `[0, 255]` à `[0, 1]`. |
| **`Conv2D` + `BatchNorm` + `ReLU`** | `(None, 224, 224, 64)` | 1 984 | Bloc 1 : Extrait les contours et textures de bas niveau. |
| **`MaxPooling2D`** | `(None, 112, 112, 64)` | 0 | Réduction spatiale (facteur 2). |
| **`Conv2D` + `BatchNorm` + `ReLU`** | `(None, 112, 112, 128)`| 74 240 | Bloc 2 : Détecte les configurations de formes locales. |
| **`MaxPooling2D`** | `(None, 56, 56, 128)` | 0 | Réduction spatiale. |
| **`Conv2D` + `BatchNorm` + `ReLU`** | `(None, 56, 56, 256)` | 295 936 | Bloc 3 : Extrait les motifs structurels complexes. |
| **`MaxPooling2D`** | `(None, 28, 28, 256)` | 0 | Réduction spatiale. |
| **`Conv2D` + `BatchNorm` + `ReLU`** | `(None, 28, 28, 512)` | 1 181 696 | Bloc 4 : Représentation sémantique de haut niveau. |
| **`MaxPooling2D`** | `(None, 14, 14, 512)` | 0 | Réduction spatiale finale. |
| **`GlobalAveragePooling2D`** | `(None, 512)` | 0 | Écrase les dimensions spatiales pour éviter l'overfitting. |
| **`Dense`** | `(None, 512)` | 262 656 | Couche de raisonnement entièrement connectée. |
| **`BatchNormalization` + `ReLU`** | `(None, 512)` | 2 048 | Stabilise et accélère l'apprentissage des représentations. |
| **`Dropout (0.5)`** | `(None, 512)` | 0 | Désactive aléatoirement 50% des neurones pour forcer la généralisation. |
| **`Dense (Sortie)`** | `(None, 2)` | 1 026 | Activation Softmax mappant les probabilités pour `[id, passport]`. |

## 3. Stratégie d'Entraînement & Optimisation

* **Optimiseur :** Adam avec un taux d'apprentissage affiné à `1e-4` (réduit par rapport au standard `1e-3` pour éviter de perturber les gradients sur des templates parfaits).
* **Fonction de perte :** `categorical_crossentropy` associée au mode `label_mode='categorical'`.
* **Taille des lots (Batch Size) :** 64 (fortement parallélisé pour tirer parti des architectures de mémoire GPU modernes type RTX 5090).
* **Callbacks activés :**
  * **`ModelCheckpoint`** : Sauvegarde automatique des meilleurs poids en fonction de la précision de validation (`kyc_classifier_best.keras`).
  * **`ReduceLROnPlateau`** : Réduit le taux d'apprentissage d'un facteur `0.2` si la perte de validation stagne pendant `3` époques.
  * **`EarlyStopping`** : Surveille la `val_accuracy` avec une patience de `8` époques, restaurant automatiquement les meilleurs poids enregistrés pour éviter le sur-apprentissage.

## 4. Métriques de Performance & Historique de Convergence

### Journal des Métriques (Exécution Optimisée)

| Époque | Précision (Entraînement) | Perte (Entraînement) | Précision (Validation) | Perte (Validation) | Statut |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Époque 1** | 86.37% | 0.2992 | 56.50% | 0.6833 | Initialisation |
| **Époque 2** | 97.75% | 0.0644 | 57.00% | 0.6775 | Apprentissage des invariants spatiaux |
| **Époque 4** | 98.87% | 0.0436 | 85.50% | 0.6630 | Percée de la généralisation |
| **Époque 5** | 98.87% | 0.0338 | 98.50% | 0.6608 | Cartographie à haute confiance |
| **Époque 7** | **99.00%** | **0.0268** | **99.00%** | **0.6451** | 🏆 **Meilleur modèle sauvegardé** |
| **Époque 11**| 98.62% | 0.0411 | 84.00% | 0.5941 | Phase d'exploration |
| **Époque 15**| 98.50% | 0.0340 | 88.00% | 0.4642 | Arrêt anticipé déclenché (Patience 8) |


