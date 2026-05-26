# Models

Ce dossier contient les weights du modele entraine. Les fichiers `.pt`, `.onnx`, `.torchscript` sont **gitignored** (trop lourds pour git).

## Comment obtenir les weights

### Option 1 — Entrainer (recommande)

Lance le notebook Colab ou le CLI :

```bash
# Colab (recommande)
# Ouvre notebooks/training_colab.ipynb -> Runtime -> Run all

# Local
python -m src.train --data data/dataset/data.yaml --epochs 40 --model yolo11n
```

Les weights atterrissent dans `runs/train/kyc_2_1_midv2020/weights/`.
Copie `best.pt` ici :

```bash
cp runs/train/kyc_2_1_midv2020/weights/best.pt models/
```

### Option 2 — Telecharger depuis Colab

Apres avoir lance le notebook, telecharge les weights depuis Colab :
- `best.pt` (5.5 MB) — PyTorch
- `best.onnx` (10.5 MB) — CPU/edge/mobile
- `best.torchscript` (10.9 MB) — PyTorch standalone

Place-les dans ce dossier.

## Utilisation

```python
from src.inference import predict_and_crop

result = predict_and_crop("doc.jpg", model_path="models/best.pt")
```
