# Etape 2.1 — Detection de champs sur documents d'identite

**Responsable :** Mathis
**Modele :** YOLO11n (Ultralytics, 2.6M params)
**Dataset :** GenMRP+MIDV-2020 (4720 images, 33 classes, via Roboflow)

## Ce que fait cette etape

Detecte **33 champs field-level** sur un document d'identite : photo du porteur, dates, MRZ, signature, numero, etc.

```
[image doc] --> [YOLO11n] --> { detections, kyc_critical, missing_fields }
```

## Sortie pour les etapes suivantes

```python
from src.inference import predict_and_crop, validate_for_kyc

result = predict_and_crop("doc.jpg", model_path="models/best.pt")
# result["kyc_critical"]["photo"]          --> input etape 2.2 (face-match)
# result["kyc_critical"]["date_of_expiry"] --> input etape 3 (rapport)
# result["detections"]                     --> dict complet pour etape 3

verdict = validate_for_kyc(result)
# {"status": "OK" | "REVIEW" | "REJECTED", "reasons": [...]}
```

Crops en **JPG** (quality 95) dans `crops/<image_stem>/`.

## Quickstart

### Colab (recommande)

1. Ouvre `notebooks/training_colab.ipynb` sur Colab
2. Runtime -> T4 GPU
3. Configure ta cle Roboflow (gratuite) en secret Colab
4. Runtime -> Run all (~25 min)

### Local

```bash
cd step_2_1_detection
pip install -r requirements.txt

export ROBOFLOW_API_KEY=your_key
bash scripts/download_dataset.sh

python -m src.train --data data/dataset/data.yaml --epochs 40 --model yolo11n
python -m src.inference --model models/best.pt --image path/to/doc.jpg
```

## Resultats

| Metrique | Val | Test |
|---|---|---|
| mAP@50 | **0.989** | **0.988** |
| mAP@50-95 | **0.834** | **0.839** |
| Precision | **0.985** | **0.983** |
| Recall | **0.972** | **0.970** |

Classes critiques : `face_image` mAP@50 = 0.995, `date_of_expiry` mAP@50 = 0.995.

## Tests

```bash
cd step_2_1_detection
pytest tests/ -v
# 31 tests passing
```

## Structure

```
step_2_1_detection/
├── notebooks/training_colab.ipynb   # Pipeline complet Colab
├── src/                             # Code source
│   ├── config.py                    # 33 classes, hyperparams
│   ├── data_prep.py                 # Download Roboflow
│   ├── train.py                     # CLI training
│   └── inference.py                 # predict_and_crop + validate_for_kyc
├── models/                          # Weights (.gitignored)
├── data/                            # Dataset (.gitignored)
├── results/                         # Plots et metriques (.gitignored)
├── scripts/download_dataset.sh      # Download dataset
├── docs/plan_technique.md           # Justifications techniques
└── tests/test_pipeline.py           # 31 tests unitaires
```

## Documentation detaillee

Voir [`docs/plan_technique.md`](docs/plan_technique.md) pour les justifications techniques (choix du dataset, du modele, strategie d'entrainement).
