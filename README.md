# KYC — Étape 2.1 : Détection d'objets sur documents d'identité

Détection automatique des éléments clés (photo du porteur, MRZ, dates, champs texte) sur des documents d'identité (passeport, carte d'identité, permis de conduire) avec **YOLOv11**, entraîné sur **MIDV-2020**.

Module 2.1 d'un pipeline KYC complet :

```
[doc image] → [classification (2.0)] → [détection champs (2.1) ← ici] → [face match (2.2)] → [rapport LLM (3)]
```

---

## 🎯 Objectif

Détecter 6 classes critiques pour le KYC :

| Classe | Pourquoi |
|---|---|
| `photo` | Input direct du face-match (étape 2.2) |
| `expiry_date` | Vérifier que le doc n'est pas expiré |
| `mrz` | Vérification croisée des champs (passeports) |
| `name` | Identification du porteur |
| `birth_date` | Vérif d'identité |
| `document_number` | Identifiant unique |

---

## 🚀 Quickstart

### Option 1 — Colab (recommandé, GPU gratuit)

1. Ouvre [`notebooks/training_colab.ipynb`](notebooks/training_colab.ipynb) sur [colab.research.google.com](https://colab.research.google.com) (File → Upload notebook)
2. *Runtime → Change runtime type → T4 GPU*
3. *Runtime → Run all*

Tu obtiens en ~30 min un modèle entraîné + tous les plots d'évaluation.

### Option 2 — Local (GPU requis)

```bash
git clone https://github.com/<your-username>/kyc-doc-detection.git
cd kyc-doc-detection
pip install -r requirements.txt

# 1. Télécharger le dataset MIDV-2020 (~2 GB)
bash scripts/download_midv2020.sh

# 2. Convertir les annotations VIA → format YOLO
python -m src.data_prep --input data/raw --output data/yolo

# 3. Entraîner
python -m src.train --data data/yolo/data.yaml --epochs 50 --model yolo11s

# 4. Inférence sur une image
python -m src.inference --model runs/train/weights/best.pt --image path/to/doc.jpg
```

---

## 📂 Structure du repo

```
kyc-doc-detection/
├── README.md                     ← tu es ici
├── requirements.txt              ← deps pinned
├── .gitignore                    ← Python + ML
├── docs/
│   └── plan_technique.md         ← Justification des choix techniques
├── notebooks/
│   └── training_colab.ipynb      ← Pipeline complet end-to-end
├── src/
│   ├── config.py                 ← Classes, mapping VIA→YOLO, hyperparams
│   ├── data_prep.py              ← Conversion VIA JSON → format YOLO
│   ├── train.py                  ← Entraînement CLI
│   └── inference.py              ← predict_and_crop() + validation KYC
├── scripts/
│   └── download_midv2020.sh      ← Téléchargement dataset
└── tests/
    └── test_conversion.py        ← Tests unitaires conversion VIA→YOLO
```

---

## 📊 Modèle & dataset

**Modèle :** [YOLOv11s](https://docs.ultralytics.com/models/yolo11/) (Ultralytics)
- 9.4 M params · mAP COCO 47.0
- Transfer learning depuis poids pré-entraînés COCO
- Fine-tuné 50 epochs sur MIDV-2020 photo (1000 images)

**Dataset :** [MIDV-2020](https://l3i-share.univ-lr.fr/MIDV2020/) split `photo`
- 1000 images de documents d'identité photographiés au smartphone
- Annotations field-level au format VIA v2 JSON
- ⚠️ MIDV-500 (référencé dans l'énoncé) ne fournit que les coins du document — voir [`docs/plan_technique.md`](docs/plan_technique.md) §1.1 pour le pourquoi du choix MIDV-2020.

---

## 📈 Résultats obtenus

Entraînement : **48 epochs** sur T4 (Colab free), early stopping (patience=15), durée totale **1h41**.

### Métriques globales (val set)

| Métrique | Cible | Résultat | Statut |
|---|---|---|---|
| mAP@50 | > 0.85 | **0.995** | ✅ |
| mAP@50-95 | > 0.65 | **0.911** | ✅ |
| Precision | > 0.90 | **0.995** | ✅ |
| Recall | > 0.85 | **0.990** | ✅ |

### Métriques par classe

| Classe | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **all** | 0.996 | 0.991 | 0.995 | 0.912 |
| photo ⭐ | 0.998 | 1.000 | 0.995 | **0.995** |
| mrz | 0.996 | 1.000 | 0.995 | 0.697 |
| name | 0.991 | 0.996 | 0.995 | 0.919 |
| birth_date | 1.000 | 0.989 | 0.995 | 0.969 |
| expiry_date | 1.000 | 0.999 | 0.995 | 0.960 |
| document_number | 0.994 | 0.963 | 0.995 | 0.933 |

> La classe `photo` atteint un mAP@50-95 de **0.995** — quasi-parfait pour l'input du face-match (étape 2.2).

Les courbes (PR, F1, confusion matrix) sont générées automatiquement par Ultralytics dans `runs/train/`.

---

## 🔌 Interface pour l'aval (étape 2.2, étape 3)

La fonction `predict_and_crop()` produit un dict structuré directement consommable par les étapes suivantes :

```python
from src.inference import predict_and_crop

result = predict_and_crop("doc.jpg", model_path="runs/train/weights/best.pt")
# {
#   "detections": {
#     "photo": {"bbox": [x1,y1,x2,y2], "conf": 0.97, "crop_path": "crops/photo.png"},
#     "expiry_date": {"bbox": [...], "conf": 0.92, "crop_path": "crops/expiry.png"},
#   },
#   "missing_fields": ["birth_date"],
#   "is_valid_kyc_input": True
# }
```

Le `crop_path` de la photo est l'input direct du face-match (étape 2.2).
Le dict complet est l'input direct du LLM de rapport (étape 3).

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/
```

---

## 📚 Documentation

- **Plan technique complet** : [`docs/plan_technique.md`](docs/plan_technique.md)
  - Pourquoi MIDV-2020 et pas MIDV-500
  - Pourquoi YOLOv11 et pas RF-DETR
  - Stratégie d'entraînement détaillée

---

## 📝 Licence

MIT — voir [LICENSE](LICENSE).

---

## 🙋 Auteur

Mathis · Module KYC · 2026
