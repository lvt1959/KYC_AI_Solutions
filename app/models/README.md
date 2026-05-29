# Modeles KYC

Ce dossier contient les poids des modeles utilises par l'application.

**Les fichiers `.keras` et `.pt` ne sont PAS inclus dans le repo (trop lourds).**

## Fichiers attendus

| Fichier | Taille | Source | Etape |
|---------|--------|--------|-------|
| `kyc_classifier_best.keras` | ~7 MB | Entrainement CNN (Colab) | Etape 1 |
| `best.pt` | ~5 MB | Entrainement YOLO11n (Colab) | Etape 2.1 |

## Comment les obtenir

### Etape 1 — CNN Classifier

1. Ouvrir `step_1_classification/network.py` dans Google Colab (GPU)
2. Lancer l'entrainement (~15 min sur T4)
3. Telecharger `kyc_classifier_best.keras`
4. Le placer ici

### Etape 2.1 — YOLO11n Detector

1. Ouvrir `step_2_1_detection/notebooks/training_colab.ipynb` dans Colab
2. Lancer l'entrainement (~20 min sur T4)
3. Telecharger `runs/detect/train/weights/best.pt`
4. Le renommer `best.pt` et le placer ici

### Modeles auto-telecharges (pas besoin de les mettre ici)

| Modele | Taille | Usage |
|--------|--------|-------|
| YOLOv8-Face | ~25 MB | Detection de visage (HuggingFace) |
| ArcFace buffalo_l | ~300 MB | Embeddings faciaux (insightface) |

Ces modeles sont telecharges automatiquement au premier lancement de l'etape Face Match.
