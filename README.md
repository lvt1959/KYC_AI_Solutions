# KYC AI Solutions

Pipeline KYC (Know Your Customer) complet : de l'image du document d'identite au rapport de verification.

## Pipeline

```
  step_1            step_2_1                step_2_2              step_3
┌───────────┐     ┌───────────┐          ┌───────────┐        ┌─────────┐
│ Classif.  │     │ Detection │ -photo-> │   Face    │ -----> │ Rapport │ --> APPROVED
│   CNN     │ --> │  champs   │          │  Matching │        │   LLM   │    / REJECTED
│ id/passp. │     │ (YOLO11n) │ -fields> │           │        │         │    / REVIEW
└───────────┘     └───────────┘          └───────────┘        └─────────┘
 2 classes         33 classes             selfie vs doc         synthese
 224x224           face, dates,           ArcFace 512-dim       decision
                   MRZ, signature         cosine distance       finale
```

## Application Web

L'application Streamlit integre les 4 etapes dans une interface unifiee.

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Voir [`app/models/README.md`](app/models/README.md) pour obtenir les poids des modeles.

## Etapes

| Etape | Dossier | Description | Modele |
|---|---|---|---|
| **1** | [`step_1_classification/`](step_1_classification/) | Classification binaire (CNI / Passeport) | CNN Keras (4 blocs conv, 224x224) |
| **2.1** | [`step_2_1_detection/`](step_2_1_detection/) | Detection de 33 champs (mAP@50=0.989) | YOLO11n (GenMRP+MIDV-2020) |
| **2.2** | [`step_2_2_face_match/`](step_2_2_face_match/) | Comparaison photo doc vs selfie | YOLOv8-Face + ArcFace buffalo_l |
| **3** | [`step_3_rapport/`](step_3_rapport/) | Rapport KYC legal via LLM | OpenRouter (configurable) |
| **App** | [`app/`](app/) | Application web Streamlit | Integre les 4 etapes |

## Comment les etapes se connectent

```python
# Etape 1 : classifier le document
from app.pipeline.classifier import classify_document, load_classifier
model = load_classifier("app/models/kyc_classifier_best.keras")
cls = classify_document(img_bgr, model)  # -> {type_document, confiance}

# Etape 2.1 : detecter les champs
from app.pipeline.detector import predict_and_crop, validate_for_kyc
model = load_detector("app/models/best.pt")
result = predict_and_crop(img_bgr, model)
photo_crop = result["kyc_critical"]["photo"]["crop_path"]       # --> step 2.2
quality = validate_for_kyc(result)                              # --> OK/REVIEW/REJECTED

# Etape 2.2 : face matching
from app.pipeline.face_match import run_face_match, load_face_models
yolo_face, arcface = load_face_models()  # auto-download
match = run_face_match(photo_id_bgr, selfie_bgr, yolo_face, arcface)

# Etape 3 : rapport final
from app.pipeline.report import generer_rapport
rapport = generer_rapport(..., api_key="sk-or-...")
```

## Quickstart par etape

Chaque etape a son propre README, ses propres requirements, et fonctionne de maniere independante.

```bash
# Clone
git clone https://github.com/lvt1959/KYC_AI_Solutions.git
cd KYC_AI_Solutions

# Application web (tout-en-un)
cd app && pip install -r requirements.txt && streamlit run app.py

# Ou par etape :
cd step_1_classification    # CNN classification
cd step_2_1_detection       # YOLO detection (pip install -r requirements.txt)
cd step_2_2_face_match      # Face matching
cd step_3_rapport           # Rapport LLM
```

## Structure du repo

```
KYC_AI_Solutions/
├── README.md
├── LICENSE
│
├── app/                               <- Application web Streamlit
│   ├── app.py                         # Interface principale
│   ├── requirements.txt               # Toutes les dependances
│   ├── models/                        # Poids des modeles (.gitignored)
│   │   └── README.md                  # Instructions pour obtenir les poids
│   └── pipeline/                      # Modules Python reutilisables
│       ├── classifier.py              # Etape 1 — CNN
│       ├── detector.py                # Etape 2.1 — YOLO
│       ├── face_match.py              # Etape 2.2 — ArcFace
│       └── report.py                  # Etape 3 — LLM
│
├── step_1_classification/             <- CNN (id/passport)
│   ├── model.md                       # Documentation du modele
│   └── network.py                     # Architecture + entrainement
│
├── step_2_1_detection/                <- YOLO11n (33 classes)
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
├── step_2_2_face_match/               <- YOLOv8-Face + ArcFace
│   ├── README.md
│   ├── requirements.txt
│   └── notebooks/face_match_final.ipynb
│
├── step_3_rapport/                    <- Rapport LLM (OpenRouter)
│   ├── README.md
│   └── rapport_kyc.ipynb
│
└── .github/workflows/test.yml         # CI
```

## Licence

MIT — voir [LICENSE](LICENSE).

## Equipe

Module KYC — 2026
