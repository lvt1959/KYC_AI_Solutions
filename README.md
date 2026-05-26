# KYC -- Etape 2.1 : Detection de champs sur documents d'identite

Detection automatique des elements cles (photo du porteur, MRZ, dates, champs texte) sur des documents d'identite (passeport, carte d'identite, permis de conduire) avec **YOLO11n**, entraine sur **GenMRP+MIDV-2020**.

Module 2.1 d'un pipeline KYC complet :

```
[doc image] -> [classification (2.0)] -> [detection champs (2.1) <- ici] -> [face match (2.2)] -> [rapport LLM (3)]
```

---

## Objectif

Detecter **33 classes field-level** sur les documents d'identite. Les classes critiques pour le KYC :

| Classe | Pourquoi |
|---|---|
| `face_image` | Input direct du face-match (etape 2.2) |
| `date_of_expiry` | Verifier que le doc n'est pas expire |
| `MRZ_line_1/2` | Verification croisee des champs (passeports) |
| `primary_identifier` | Nom du porteur |
| `date_of_birth` | Verification d'identite |
| `document_number` | Identifiant unique |

Et 27 autres classes (captions, signature, nationalite, sexe, autorite d'emission...).

---

## Quickstart

### Option 1 -- Colab (recommande, GPU gratuit)

1. Ouvre [`notebooks/training_colab.ipynb`](notebooks/training_colab.ipynb) sur [colab.research.google.com](https://colab.research.google.com) (File -> Upload notebook)
2. *Runtime -> Change runtime type -> T4 GPU*
3. Configure ta cle API Roboflow (gratuite) :
   - Cree un compte sur https://app.roboflow.com
   - Va sur https://app.roboflow.com/settings/api
   - Dans Colab : icone cle dans la barre de gauche -> "Add new secret" -> nom = `ROBOFLOW_API_KEY` -> colle la cle
4. *Runtime -> Run all*

Tu obtiens en ~25 min un modele entraine + tous les plots d'evaluation.

### Option 2 -- Local (GPU requis)

```bash
git clone https://github.com/lvt1959/KYC_AI_Solutions.git
cd KYC_AI_Solutions
pip install -r requirements.txt

# 1. Telecharger le dataset via Roboflow
export ROBOFLOW_API_KEY=your_key_here
bash scripts/download_dataset.sh

# 2. Entrainer
python -m src.train --data data/dataset/data.yaml --epochs 40 --model yolo11n

# 3. Inference sur une image
python -m src.inference --model runs/train/kyc_2_1_midv2020/weights/best.pt --image path/to/doc.jpg
```

---

## Structure du repo

```
KYC_AI_Solutions/
├── README.md                     <- tu es ici
├── requirements.txt              <- deps pinned
├── .gitignore                    <- Python + ML
│
├── notebooks/                    <- Colab notebooks
│   └── training_colab.ipynb      <- Pipeline complet end-to-end
│
├── src/                          <- Code source (importable)
│   ├── config.py                 <- 33 classes, aliases, hyperparams
│   ├── data_prep.py              <- Download Roboflow + legacy VIA->YOLO
│   ├── train.py                  <- Entrainement CLI
│   └── inference.py              <- predict_and_crop() + validate_for_kyc()
│
├── models/                       <- Weights entrainees (.gitignored)
│   └── README.md                 <- Comment obtenir/generer les weights
│
├── data/                         <- Dataset telecharge (.gitignored)
│   └── .gitkeep
│
├── results/                      <- Metriques, plots, exports (.gitignored)
│   └── README.md                 <- Derniers resultats documentes
│
├── scripts/                      <- Scripts utilitaires
│   └── download_dataset.sh       <- Telechargement dataset (Roboflow ou legacy)
│
├── docs/                         <- Documentation technique
│   └── plan_technique.md         <- Justification des choix techniques
│
└── tests/                        <- Tests unitaires
    └── test_pipeline.py          <- Data prep + validation KYC (31 tests)
```

---

## Modele & dataset

**Modele :** [YOLO11n](https://docs.ultralytics.com/models/yolo11/) (Ultralytics)
- 2.6M params, inference 3-5ms sur T4
- Transfer learning depuis poids pre-entraines COCO
- Fine-tune 40 epochs sur GenMRP+MIDV-2020

**Dataset :** [GenMRP+MIDV-2020](https://universe.roboflow.com/maastricht-university/genmrp-midv-2020/dataset/1) (Maastricht University, via Roboflow)
- 4720 images (MIDV-2020 + passeports synthetiques GenMRP)
- Split : 4480 train / 160 val / 80 test
- 33 classes field-level, annotations au format YOLO
- Licence CC BY 4.0

---

## Resultats obtenus

Entrainement sur T4 (Colab free), 40 epochs, ~25 min.

### Metriques globales

| Metrique | Cible | Val | Test | Statut |
|---|---|---|---|---|
| mAP@50 | > 0.85 | **0.989** | **0.988** | OK |
| mAP@50-95 | > 0.65 | **0.834** | **0.839** | OK |
| Precision | > 0.90 | **0.985** | **0.983** | OK |
| Recall | > 0.85 | **0.972** | **0.970** | OK |

### Classes KYC-critiques

| Classe | Val mAP@50 | Test mAP@50 | Val mAP@50-95 |
|---|---|---|---|
| `face_image` | **0.995** | **0.995** | **0.994** |
| `date_of_expiry` | **0.995** | **0.995** | **0.939** |
| `document_number` | **0.995** | **0.994** | **0.886** |
| `date_of_birth` | **0.995** | **0.995** | **0.922** |
| `MRZ_line_1` | **0.989** | **0.987** | **0.495** |

> La classe `face_image` atteint un mAP@50-95 de **0.994** -- quasi-parfait pour l'input du face-match (etape 2.2).

Les courbes (PR, F1, confusion matrix) sont generees automatiquement dans `runs/`.

---

## Interface pour l'aval (etape 2.2, etape 3)

### predict_and_crop()

```python
from src.inference import predict_and_crop

result = predict_and_crop("doc.jpg", model_path="runs/train/kyc_2_1_midv2020/weights/best.pt")
# {
#   "image_path": "doc.jpg",
#   "detections": {
#     "face_image": {"bbox": [x1,y1,x2,y2], "conf": 0.95, "crop_path": "crops/doc/face_image.jpg"},
#     "date_of_expiry": {"bbox": [...], "conf": 0.88, "crop_path": "crops/doc/date_of_expiry.jpg"},
#     ...
#   },
#   "kyc_critical": {
#     "photo": {"bbox": ..., "conf": 0.95, "crop_path": ...},
#     "date_of_expiry": {"bbox": ..., "conf": 0.88, "crop_path": ...}
#   },
#   "missing_fields": []
# }
```

Le `crop_path` de la photo est l'input direct du face-match (etape 2.2).
Le dict complet est l'input direct du LLM de rapport (etape 3).
Crops en **JPG** (quality 95).

### validate_for_kyc()

```python
from src.inference import validate_for_kyc

verdict = validate_for_kyc(result)
# {"status": "OK", "reasons": []}
# {"status": "REVIEW", "reasons": ["Date d'expiration non detectee..."]}
# {"status": "REJECTED", "reasons": ["CRITIQUE : photo non detectee..."]}
```

Regles metier :
- Photo absente -> **REJECTED** (face-match impossible)
- Photo conf < 0.60 -> **REVIEW**
- Date d'expiration absente -> **REVIEW** (rapport incomplet)
- Confiance moyenne < 0.65 -> **REVIEW** (image probablement floue)

### Export

| Format | Taille | Usage |
|---|---|---|
| `best.pt` | 5.5 MB | PyTorch |
| `best.onnx` | 10.5 MB | CPU/edge/mobile |
| `best.torchscript` | 10.9 MB | PyTorch standalone |

---

## Tests

```bash
pip install pytest pillow pyyaml
pytest tests/ -v
```

---

## Documentation

- **Plan technique complet** : [`docs/plan_technique.md`](docs/plan_technique.md)
  - Pourquoi GenMRP+MIDV-2020 et pas MIDV-2020 officiel
  - Pourquoi YOLO11n et pas YOLO11s
  - Strategie d'entrainement detaillee

---

## Licence

MIT -- voir [LICENSE](LICENSE).

---

## Auteur

Mathis - Module KYC - 2026
