# KYC AI Solutions

Pipeline KYC (Know Your Customer) complet : de l'image du document d'identite au rapport de verification.

## Pipeline

```
                    step_2_1                step_2_2              step_3
                  ┌───────────┐          ┌───────────┐        ┌─────────┐
  [image doc] --> │ Detection │ -photo-> │   Face    │ -----> │ Rapport │ --> APPROVED
                  │  champs   │          │  Matching │        │   LLM   │    / REJECTED
                  │ (YOLO11n) │ -fields> │           │        │         │    / REVIEW
                  └───────────┘          └───────────┘        └─────────┘
                   33 classes             selfie vs doc         synthese
                   face, dates,           match/no match        decision
                   MRZ, signature                               finale
```

## Etapes

| Etape | Dossier | Responsable | Statut | Description |
|---|---|---|---|---|
| **2.1** | [`step_2_1_detection/`](step_2_1_detection/) | Mathis | Done | Detection de 33 champs (YOLO11n, mAP@50=0.989) |
| **2.2** | [`step_2_2_face_match/`](step_2_2_face_match/) | (a assigner) | A faire | Comparaison photo doc vs selfie |
| **3** | [`step_3_rapport/`](step_3_rapport/) | (a assigner) | A faire | Rapport KYC structure via LLM |

## Comment les etapes se connectent

```python
# Etape 2.1 : detecter les champs
from step_2_1_detection.src.inference import predict_and_crop, validate_for_kyc

result = predict_and_crop("doc.jpg", model_path="step_2_1_detection/models/best.pt")
photo_crop = result["kyc_critical"]["photo"]["crop_path"]       # --> step 2.2
expiry_crop = result["kyc_critical"]["date_of_expiry"]["crop_path"]  # --> step 3
all_detections = result["detections"]                           # --> step 3

# Etape 2.2 : face matching
# face_result = compare_faces(photo_crop, "selfie.jpg")        # --> step 3

# Etape 3 : rapport final
# report = generate_report(all_detections, face_result)
```

## Quickstart par etape

Chaque etape a son propre README, ses propres requirements, et fonctionne de maniere independante.

```bash
# Clone
git clone https://github.com/lvt1959/KYC_AI_Solutions.git
cd KYC_AI_Solutions

# Etape 2.1 — Detection
cd step_2_1_detection
pip install -r requirements.txt
# Voir step_2_1_detection/README.md pour la suite

# Etape 2.2 — Face match (quand disponible)
cd ../step_2_2_face_match
# Voir step_2_2_face_match/README.md

# Etape 3 — Rapport (quand disponible)
cd ../step_3_rapport
# Voir step_3_rapport/README.md
```

## Structure du repo

```
KYC_AI_Solutions/
├── README.md                          <- tu es ici
├── LICENSE
│
├── step_2_1_detection/                <- Mathis — DONE
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/training_colab.ipynb
│   ├── src/                           # config, data_prep, train, inference
│   ├── models/                        # Weights (.gitignored)
│   ├── data/                          # Dataset (.gitignored)
│   ├── results/                       # Metriques et plots (.gitignored)
│   ├── scripts/                       # download_dataset.sh
│   ├── docs/                          # plan_technique.md
│   └── tests/                         # 31 tests (pytest)
│
├── step_2_2_face_match/               <- A faire
│   └── README.md                      # Specs + inputs attendus
│
├── step_3_rapport/                    <- A faire
│   └── README.md                      # Specs + inputs attendus
│
└── .github/workflows/test.yml         # CI
```

## Licence

MIT — voir [LICENSE](LICENSE).

## Equipe

Module KYC — 2026
