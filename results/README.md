# Results

Ce dossier stocke les resultats d'entrainement et d'evaluation. Les fichiers sont **gitignored** (generes automatiquement).

## Contenu apres un training

```
results/
├── confusion_matrix.png       # Matrice de confusion (val set)
├── PR_curve.png               # Precision-Recall curve
├── F1_curve.png               # F1 score curve
├── results.png                # Courbes loss + metriques par epoch
├── results.csv                # Metriques par epoch (parseable)
└── val_predictions/           # Exemples de predictions visuelles
```

## Comment generer

Les plots sont generes automatiquement par Ultralytics pendant le training (`plots=True` dans la config).

```bash
# Apres training, copier les resultats ici :
cp runs/train/kyc_2_1_midv2020/*.png results/
cp runs/train/kyc_2_1_midv2020/results.csv results/
```

Ou directement dans le notebook Colab (section 6 — Evaluation).

## Derniers resultats (YOLO11n, GenMRP+MIDV-2020)

| Metrique | Val | Test |
|---|---|---|
| mAP@50 | 0.989 | 0.988 |
| mAP@50-95 | 0.834 | 0.839 |
| Precision | 0.985 | 0.983 |
| Recall | 0.972 | 0.970 |
