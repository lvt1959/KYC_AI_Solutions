# KYC -- Etape 2.1 : Detection de champs sur documents d'identite

**Auteur :** Mathis
**Modele retenu :** YOLO11n (Ultralytics, 2.6M params)
**Dataset retenu :** GenMRP+MIDV-2020 (Maastricht University, via Roboflow Universe)
**Livrable associe :** `notebooks/training_colab.ipynb`

---

## 1. Contexte et reformulation du besoin

L'enonce demande de **detecter des elements cles** (photo du porteur, date d'expiration, champs texte structures) sur un document d'identite, a partir du dataset reference par le repo `fcakyon/midv500`.

### 1.1 Le piege du dataset

Le repo `fcakyon/midv500` est un outil de telechargement/conversion qui couvre **MIDV-500** et **MIDV-2019**. Or :

| Dataset | Contenu annote |
|---|---|
| **MIDV-500** | Quadrilatere des **4 coins du document** uniquement |
| **MIDV-2019** | Idem MIDV-500, sur images degradees (low-light, distorsions) |
| **MIDV-2020** | **Champs au niveau pixel** : photo, MRZ, nom, date de naissance, date d'expiration, numero de document, signature, etc. |

Pour repondre a la consigne (detecter photo + dates), **MIDV-500 brut ne suffit pas**.

### 1.2 Decision : GenMRP+MIDV-2020

La source officielle MIDV-2020 (`l3i-share.univ-lr.fr`) est **124 GB sur sFTP** + demande d'acces par formulaire Google. Pas exploitable en Colab dans un timing realiste.

Le mix **GenMRP+MIDV-2020** (Maastricht University, disponible sur Roboflow Universe) est un **superset** :
- MIDV-2020 + passeports synthetiques (GenMRP)
- **4720 images** avec augmentations 8x
- Annotations **field-level** deja au format YOLO
- Telechargement en 30s via API Roboflow
- Licence CC BY 4.0

**Caveat honnete :** ce n'est pas MIDV-2020 pur, c'est un superset incluant des donnees synthetiques. C'est defendable car cela ameliore la robustesse en prod (plus de variabilite).

---

## 2. Choix du modele : YOLO11n

### 2.1 Pourquoi YOLO11 ?

| Critere | YOLO11 | RF-DETR | RT-DETR |
|---|---|---|---|
| Simplicite fine-tuning | 5/5 (1 ligne) | 3/5 | 2/5 |
| Docs / tutos | 5/5 | 3/5 | 2/5 |
| Perfs sur petits datasets | 4/5 | 4/5 | 4/5 |
| Vitesse inference (CPU) | 5/5 | 3/5 | 4/5 |
| Maturite ecosysteme | 5/5 | 2/5 | 3/5 |

### 2.2 Pourquoi nano (n) et pas small (s) ?

| Variante | Params | mAP COCO | VRAM (train) |
|---|---|---|---|
| **`yolo11n`** | **2.6M** | **39.5** | **~4 GB** |
| `yolo11s` | 9.4M | 47.0 | ~6 GB |
| `yolo11m` | 20.1M | 51.5 | ~10 GB |

YOLO11n suffit pour ~33 classes faciles (photo, dates, signature... objets bien delimites sur documents). Inference 3x plus rapide que YOLO11s, critique pour KYC mobile temps-reel.

Si la perf n'est pas au rendez-vous : `model = YOLO("yolo11s.pt")` + relancer.

---

## 3. Les 33 classes

Le dataset GenMRP+MIDV-2020 expose 33 classes field-level :

| ID | Classe | KYC-critique ? |
|---|---|---|
| 0 | `MRZ_line_1` | |
| 1 | `MRZ_line_2` | |
| 2 | `date_of_birth` | |
| 3 | `date_of_birth_caption` | |
| 4 | `date_of_expiry` | **oui** |
| 5 | `date_of_expiry_caption` | |
| 6 | `date_of_issue` | |
| 7 | `date_of_issue_caption` | |
| 8 | `document` | |
| 9 | `document_code` | |
| 10 | `document_code_caption` | |
| 11 | `document_number` | |
| 12 | `document_number_caption` | |
| 13 | `face_image` | **oui** |
| 14-32 | issue_authority, nationality, identifiers, sex, signature + captions | |

Classes KYC-critiques :
- **`face_image`** (id=13) : input direct du face-match (etape 2.2)
- **`date_of_expiry`** (id=4) : detection doc expire (etape 3 rapport)

---

## 4. Strategie d'entrainement

### 4.1 Hyperparametres

```yaml
model: yolo11n.pt           # transfer learning depuis COCO
epochs: 40                  # early stop active via patience
batch: 32                   # T4 (16GB) -> 32 OK pour yolo11n + imgsz=640
imgsz: 640                  # standard YOLO
optimizer: AdamW            # plus stable que SGD sur petits datasets
lr0: 0.001
lrf: 0.01                   # cosine schedule final lr = lr0 * lrf
weight_decay: 0.0005
warmup_epochs: 3
patience: 12                # early stopping
```

### 4.2 Augmentations doc-friendly

- `fliplr=0.0` : pas de miroir horizontal (le texte ne survit pas)
- `flipud=0.0` : pas de retournement vertical
- `degrees=10` : rotations moderees (docs souvent en biais en KYC mobile)
- `mosaic=1.0` avec `close_mosaic=10` : mosaic active sauf en fin pour stabiliser
- `mixup=0.0` : pas de melange d'images

---

## 5. Resultats obtenus

Entrainement sur T4 (Colab free).

### Metriques globales

| Metrique | Cible | Val | Test |
|---|---|---|---|
| mAP@50 | > 0.85 | **0.989** | **0.988** |
| mAP@50-95 | > 0.65 | **0.834** | **0.839** |
| Precision | > 0.90 | **0.985** | **0.983** |
| Recall | > 0.85 | **0.972** | **0.970** |

### Classes KYC-critiques

| Classe | Val mAP@50 | Test mAP@50 |
|---|---|---|
| `face_image` | **0.995** | **0.995** |
| `date_of_expiry` | **0.995** | **0.995** |

---

## 6. Integration aval (interface avec 2.2 et 3)

### 6.1 predict_and_crop()

```python
result = predict_and_crop("doc.jpg", model_obj=model)
# {
#   "image_path": "doc.jpg",
#   "detections": {
#     "face_image": {"bbox": [x1,y1,x2,y2], "conf": 0.95, "crop_path": "crops/face_image.jpg"},
#     "date_of_expiry": {"bbox": [...], "conf": 0.88, "crop_path": "crops/date_of_expiry.jpg"},
#     ...
#   },
#   "kyc_critical": {
#     "photo": {"bbox": ..., "conf": 0.95, "crop_path": ...},
#     "date_of_expiry": {"bbox": ..., "conf": 0.88, "crop_path": ...}
#   },
#   "missing_fields": []
# }
```

Les crops sont en **JPG** (quality 95) pour un bon compromis taille/qualite.

### 6.2 validate_for_kyc()

```python
verdict = validate_for_kyc(result)
# {"status": "OK", "reasons": []}
# {"status": "REVIEW", "reasons": ["Date d'expiration non detectee..."]}
# {"status": "REJECTED", "reasons": ["CRITIQUE : photo du porteur non detectee..."]}
```

Regles :
- Photo absente -> REJECTED (face-match impossible)
- Photo conf < 0.60 -> REVIEW
- Date d'expiration absente -> REVIEW (rapport incomplet)
- Confiance moyenne < 0.65 -> REVIEW (image probablement floue)

### 6.3 Export

| Format | Taille | Usage |
|---|---|---|
| `best.pt` | 5.5 MB | PyTorch (entrainement, fine-tune) |
| `best.onnx` | 10.5 MB | Deploiement CPU/edge/mobile |
| `best.torchscript` | 10.9 MB | PyTorch standalone |

---

## 7. Pour aller plus loin

- **TTA (Test-Time Augmentation)** : `model.val(augment=True)` -> +1-2 pts mAP
- **OCR sur les crops** : Tesseract ou PaddleOCR pour extraire le texte des dates
- **Pipeline 2-stages** : YOLO upstream (MIDV-500) pour localiser+rectifier le doc avant detection field-level
- **Active learning** : logger les detections `conf < 0.5` en prod pour reannotation humaine

---

## Sources

- [GenMRP+MIDV-2020 (Roboflow Universe)](https://universe.roboflow.com/maastricht-university/genmrp-midv-2020/dataset/1)
- [MIDV-2020 dataset (L3i, Universite de La Rochelle)](https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
- [MIDV-2020 paper (arXiv)](https://arxiv.org/abs/2107.00396)
- [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
- [fcakyon/midv500 (GitHub)](https://github.com/fcakyon/midv500)
