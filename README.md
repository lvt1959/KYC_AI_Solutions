# KYC AI Solutions

Pipeline KYC (Know Your Customer) complet : de l'image du document d'identite au rapport de verification.

## Pipeline

```
  Upload          Cadrage auto      Classification     Detection champs
┌─────────┐     ┌───────────┐     ┌───────────┐     ┌───────────────┐
│  Photo   │ --> │   YOLO    │ --> │    CNN     │ --> │   YOLO 33cl   │
│ document │     │ crop+rot  │     │ id/passp.  │     │ + InsightFace │
└─────────┘     └───────────┘     └───────────┘     └───────┬───────┘
                                                            │
                   ┌────────────────────────────────────────┘
                   │
              OCR expiration        Face Match           Rapport
            ┌───────────┐       ┌───────────┐       ┌─────────┐
            │  docTR +   │      │InsightFace│       │ Rapport │ --> APPROUVE
            │ MRZ parse  │ ---> │  SCRFD +  │ ----> │   LLM   │    / REJETE
            │ VALIDE/EXP │      │  ArcFace  │       │         │    / REVUE
            └───────────┘       └───────────┘       └─────────┘
```

## Application Web

L'application Streamlit integre toutes les etapes dans une interface unifiee.

```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Voir [`app/models/README.md`](app/models/README.md) pour obtenir les poids des modeles.

## Etapes detaillees

| Etape | Module | Description | Modele | Auto-DL |
|---|---|---|---|---|
| **0** | `detector.py` | Cadrage auto + rotation du document | YOLO11n (classe "document" + union champs) | Non |
| **1** | `classifier.py` | Classification binaire (CNI / Passeport) | CNN Keras (4 blocs conv, 224x224) | Non |
| **2.1** | `detector.py` | Detection de 33 champs + fallback visage | YOLO11n + InsightFace SCRFD | Partiel |
| **2.2** | `expiry_check.py` | OCR date expiration + verification | docTR (DBNet+CRNN) + parsing MRZ | Oui |
| **2.3** | `face_match.py` | Comparaison selfie vs document | InsightFace (SCRFD + ArcFace buffalo_l) | Oui |
| **3** | `report.py` | Rapport KYC legal | OpenRouter LLM (owl-alpha) | - |

## Decisions techniques

### Face Match : InsightFace seul (pas YOLO-Face)

InsightFace (`buffalo_l`) integre detection (SCRFD) + alignement + embedding (ArcFace) en un seul appel.
YOLO-Face ne fait que la detection sans alignement, ce qui donne des embeddings moins precis.

- **Fallback det_thresh** : si SCRFD ne detecte pas le visage a 0.25 (photos d'identite imprimees, hologrammes),
  le seuil est temporairement baisse a 0.05 puis restaure.
- **Seuil de decision** : distance cosine < 0.60 = MATCH.

### OCR : docTR (pas Tesseract ni EasyOCR)

Tesseract et EasyOCR echouent sur les photos reelles de documents (hologrammes, reflets, angles).
docTR (DBNet + CRNN) est entraine specifiquement sur des documents et produit des resultats exploitables.

**3 strategies d'extraction de la date d'expiration :**

1. **MRZ** (passeports) — format OACI standardise, 190+ pays. La date est toujours au meme endroit : 6 chiffres YYMMDD apres le champ sexe. Le plus fiable.
2. **Mots-cles** (CNI, titres de sejour) — cherche "VALABLE JUSQU'AU", "VALID UNTIL", "EXPIRY" puis la date a cote.
3. **Fallback** — prend la date la plus tardive du document (naissance=passe, emission=passe, expiration=futur).

### Cadrage automatique : double strategie

Le YOLO 33-classes est entraine sur des templates synthetiques (MIDV-2020). Sur des documents reels :

- **Strategy 1** : classe `document` detectee et assez grande (>20% de l'image) → crop avec marge 12%
- **Strategy 2** : pas de `document` mais d'autres champs detectes → bounding box englobant + marge 40%
- **Auto-rotation** : si le crop est en portrait (hauteur > largeur x 1.2), rotation 90 (docs = paysage)
- **Filtre** : la detection "document" de 126x20px = le mot "document" sur la carte, pas le document entier

### Detection de visage : fallback InsightFace

Le YOLO 33-classes rate les visages sur les vrais documents (entraine sur des placeholders synthetiques).
InsightFace SCRFD prend le relais automatiquement quand `face_image` n'est pas detecte.

## API Python

```python
# Cadrage auto
from app.pipeline.detector import crop_document, load_detector
model = load_detector("app/models/best.pt")
crop = crop_document(img_bgr, model)  # -> {found, cropped, confidence}

# Classification
from app.pipeline.classifier import classify_document, load_classifier
clf = load_classifier("app/models/kyc_classifier_best.keras")
cls = classify_document(crop["cropped"], clf)  # -> {type_document, confiance}

# Detection champs
from app.pipeline.detector import predict_and_crop, validate_for_kyc
result = predict_and_crop(crop["cropped"], model, face_model=insightface_app)
quality = validate_for_kyc(result)  # -> {status: OK/REVIEW/REJECTED}

# OCR expiration
from app.pipeline.expiry_check import lire_date_expiration, verifier_expiration
ocr = lire_date_expiration(crop["cropped"])  # -> {date_texte, date_iso}
verif = verifier_expiration(ocr["date_iso"])  # -> {statut: VALIDE/EXPIRE}

# Face match
from app.pipeline.face_match import run_face_match, load_face_model
arcface = load_face_model()  # auto-download ~260MB
match = run_face_match(doc_bgr, selfie_bgr, arcface)  # -> {result, crop_selfie, crop_id}

# Rapport
from app.pipeline.report import generer_rapport
rapport = generer_rapport(..., api_key="sk-or-...")
```

## Structure du repo

```
KYC_AI_Solutions/
├── README.md
├── LICENSE
│
├── app/                               <- Application web Streamlit
│   ├── app.py                         # Interface 4 onglets
│   ├── requirements.txt
│   ├── models/                        # Poids (.gitignored)
│   │   └── README.md
│   └── pipeline/
│       ├── classifier.py              # Etape 1 — CNN
│       ├── detector.py                # Cadrage auto + YOLO 33cl + fallback visage
│       ├── expiry_check.py            # OCR docTR + MRZ + verification expiration
│       ├── face_match.py              # InsightFace (SCRFD + ArcFace)
│       └── report.py                  # Rapport LLM (OpenRouter)
│
├── step_1_classification/             <- CNN (id/passport)
├── step_2_1_detection/                <- YOLO11n (33 classes)
├── step_2_2_face_match/               <- Notebook face match
├── step_3_rapport/                    <- Notebook rapport LLM
└── .github/workflows/test.yml         # CI
```

## Licence

MIT — voir [LICENSE](LICENSE).

## Equipe

Module KYC — 2026
