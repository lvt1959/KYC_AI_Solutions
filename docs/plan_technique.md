# KYC — Étape 2.1 : Détection d'objets sur documents d'identité

**Auteur :** Mathis
**Modèle retenu :** YOLOv11 (Ultralytics)
**Dataset retenu :** MIDV-2020 (extension officielle de MIDV-500, avec annotations field-level)
**Livrable associé :** `kyc_etape_2_1_yolov11_midv2020.ipynb`

---

## 1. Contexte et reformulation du besoin

L'énoncé demande de **détecter des éléments clés** (photo du porteur, date d'expiration, champs texte structurés) sur un document d'identité, à partir du dataset référencé par le repo `fcakyon/midv500`.

### 1.1 Le piège du dataset

Le repo `fcakyon/midv500` est un outil de téléchargement/conversion qui couvre **MIDV-500** et **MIDV-2019**. Or :

| Dataset | Contenu annoté |
|---|---|
| **MIDV-500** | Quadrilatère des **4 coins du document** uniquement |
| **MIDV-2019** | Idem MIDV-500, sur images dégradées (low-light, distorsions) |
| **MIDV-2020** | **Champs au niveau pixel** : photo, MRZ, nom, date de naissance, date d'expiration, numéro de document, signature, etc. |

→ Pour répondre à la consigne (détecter photo + dates), **MIDV-500 brut ne suffit pas**. Il faut MIDV-2020, qui est l'évolution officielle (mêmes auteurs, l3i-share + Smart Engines).

### 1.2 Décision

J'utilise **MIDV-2020** et je le justifie dans le rapport : c'est la version étendue de la même famille MIDV, et c'est le seul des trois qui contient les annotations field-level requises par l'énoncé. En soutenance : « MIDV-500 = corners only, MIDV-2020 = fields ; même série, vraie tâche. »

> **Option de repli si le prof exige strictement MIDV-500 :** détecter le document entier sur MIDV-500, puis pré-annoter manuellement 200-300 images avec Roboflow Annotate pour les champs (~3h de boulot). Cette option est documentée en fin de notebook.

---

## 2. Choix du modèle : YOLOv11

### 2.1 Pourquoi pas RF-DETR / RT-DETR ?

| Critère | YOLOv11 | RF-DETR | RT-DETR |
|---|---|---|---|
| Simplicité fine-tuning | ★★★★★ (1 ligne) | ★★★ | ★★ |
| Docs / tutos | ★★★★★ | ★★★ | ★★ |
| Perfs sur petits datasets | ★★★★ | ★★★★ | ★★★★ |
| Vitesse inférence (CPU) | ★★★★★ | ★★★ | ★★★★ |
| Défendable en soutenance | ★★★★★ | ★★★★ | ★★★ |
| Maturité écosystème | ★★★★★ | ★★ | ★★★ |

YOLOv11 sorti fin 2024 par Ultralytics — état de l'art sur COCO pour la catégorie temps-réel, **45 % moins de paramètres que YOLOv8m** à mAP équivalente. Le format d'annotation YOLO (1 .txt par image avec `class x_center y_center w h` normalisés) est trivial à générer depuis n'importe quelle source.

### 2.2 Variante recommandée

| Variante | Params | mAP COCO | VRAM (train) | Cas d'usage |
|---|---|---|---|---|
| `yolo11n` | 2.6M | 39.5 | ~4 GB | Prototype rapide |
| **`yolo11s`** | **9.4M** | **47.0** | **~6 GB** | **Recommandée — Colab free** |
| `yolo11m` | 20.1M | 51.5 | ~10 GB | Si tu as Colab Pro / A100 |
| `yolo11l` | 25.3M | 53.4 | ~16 GB | Production |

Je pars sur **YOLOv11s** : sweet spot perf/VRAM, tourne en moins de 30 min sur T4 (Colab free) pour 50 epochs sur ~1000 images.

---

## 3. Architecture du pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                        ÉTAPE 2.1                            │
│                                                             │
│   ID image → [YOLOv11s] → bbox(photo) + bbox(fields)       │
│                              │                              │
│                              ▼                              │
│            crops/photo.png  + JSON {champ: bbox}            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Étape 2.2 (face match)    │
                │   crops/photo.png + selfie  │
                └─────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │   Étape 3 (LLM rapport)     │
                │   inputs = crops + champs   │
                └─────────────────────────────┘
```

**Sortie clé pour l'aval :** un dict JSON par image avec, pour chaque classe détectée, la bbox + le crop sauvegardé. Ça donne directement à manger à 2.2 et à l'étape 3.

---

## 4. Préparation des données

### 4.1 Téléchargement

MIDV-2020 expose 3 modalités :

| Modalité | Volume | Annotations | Utilité |
|---|---|---|---|
| **photo** | 1000 images | VIA JSON, fields | ⭐ La plus utile pour notre tâche |
| scans | 2000 images | VIA JSON, fields | Bonus pour robustesse |
| clips video | 1000 clips | VIA JSON, frames | Moins prioritaire |

→ On part sur **`photo`** (1000 images de docs photographiés au smartphone, conditions réalistes pour du KYC mobile).

Sources :
- `ftp://smartengines.com/midv-2020`
- `https://l3i-share.univ-lr.fr/MIDV2020/`

### 4.2 Classes à détecter

Je propose un schéma **6 classes** (équilibre couverture / volume d'annotation) :

| ID | Classe | Justification |
|---|---|---|
| 0 | `photo` | ⭐ Obligatoire — input direct pour étape 2.2 |
| 1 | `mrz` | Vérif croisée des champs |
| 2 | `name` | Identification |
| 3 | `birth_date` | Vérif d'identité |
| 4 | `expiry_date` | ⭐ Détection KYC critique (doc expiré) |
| 5 | `document_number` | Identification unique |

> Tu peux pruner à 2 classes (`photo`, `expiry_date`) si tu veux coller au minimum vital de l'énoncé. Mais 6 classes ne coûtent rien de plus en compute et c'est plus défendable.

### 4.3 Conversion VIA → YOLO

VIA (VGG Image Annotator) stocke les annotations en JSON avec des polygones. YOLO veut des bboxes normalisées. Le notebook contient le script de conversion (≈40 lignes) qui :

1. Parse le JSON VIA
2. Pour chaque région : récupère le polygone → calcule la bounding box englobante
3. Normalise par (W, H) de l'image
4. Mappe le label VIA vers l'ID de classe (table de mapping configurable)
5. Écrit `image.txt` avec une ligne par bbox : `class_id cx cy w h`

### 4.4 Split

Standard 70/20/10 : `train` / `val` / `test`. Stratification par **type de document** (passport / id_card / driving_license) pour garantir que les 3 splits couvrent toutes les catégories.

---

## 5. Stratégie d'entraînement

### 5.1 Hyperparamètres de départ

```yaml
model: yolo11s.pt           # transfer learning depuis COCO
epochs: 50                  # early stop activé via patience
batch: 16                   # T4 (16GB) → 16 OK pour imgsz=640
imgsz: 640                  # standard YOLO
optimizer: AdamW            # plus stable que SGD sur petits datasets
lr0: 0.001
lrf: 0.01                   # cosine schedule final lr = lr0 * lrf
weight_decay: 0.0005
warmup_epochs: 3
patience: 15                # early stopping
```

### 5.2 Augmentations (Ultralytics auto)

YOLOv11 inclut nativement :
- **Mosaic** (4 images concaténées) → désactivé sur les 10 dernières epochs (`close_mosaic=10`)
- **Mixup** (alpha=0.0 par défaut, à augmenter à 0.1 si overfitting)
- **HSV, flip, rotation, scale, translate** → laissés par défaut
- **Auto-augment** ❌ → désactivé : on ne veut pas que les chiffres de la date soient flippés (risque de leak)

⚠️ **Attention spécifique aux documents d'identité :** désactiver le **flip horizontal** (`fliplr=0.0`). Sinon la MRZ et les dates deviennent illisibles → modèle apprend des features inversés.

### 5.3 Transfer learning

On part des poids **COCO pré-entraînés** (`yolo11s.pt`). Les 1000 images de MIDV-2020 photo sont insuffisantes pour un train from scratch.

Backbone : on **ne gèle rien** au début (les features bas niveau de COCO sont utiles : edges, textures). Si tu vois de l'overfitting (val mAP qui décroche), tu peux freezer les 10 premières couches : `freeze=10`.

---

## 6. Évaluation

### 6.1 Métriques

| Métrique | Cible | Lecture |
|---|---|---|
| **mAP@50** | > 0.85 | Détections "OK pour humain" |
| **mAP@50-95** | > 0.65 | Détections précises (utilisable en aval) |
| **mAP par classe** | photo > 0.95 | La classe photo doit être quasi parfaite |
| **Precision** | > 0.90 | Peu de faux positifs (KYC = critique) |
| **Recall** | > 0.85 | Peu de faux négatifs (sinon doc bloqué à tort) |

### 6.2 Analyse qualitative obligatoire

Le notebook produit :
- **Matrice de confusion** (val set)
- **20 exemples visualisés** (10 succès + 10 échecs) → utiles pour la soutenance
- **Distribution des IoU** par classe
- **Failure cases** triés par confiance → identifie les patterns d'erreur

---

## 7. Intégration aval (interface avec 2.2 et 3)

### 7.1 Format de sortie

Le notebook fournit une fonction `predict_and_crop(image_path)` qui retourne :

```json
{
  "image_path": "...",
  "detections": {
    "photo": {"bbox": [x1, y1, x2, y2], "conf": 0.97, "crop_path": "crops/photo.png"},
    "expiry_date": {"bbox": [...], "conf": 0.92, "crop_path": "crops/expiry.png"},
    "mrz": {"bbox": [...], "conf": 0.95, "crop_path": "crops/mrz.png"}
  },
  "missing_fields": ["birth_date"],
  "is_valid_kyc_input": true
}
```

→ La personne qui code l'étape 2.2 reçoit `crops/photo.png` direct.
→ L'étape 3 (LLM) reçoit le JSON entier comme input.

### 7.2 Règles de validation embarquées

Le notebook inclut une fonction `validate_detection_quality(result)` qui flag :
- `photo` non détectée → ❌ rejet immédiat
- Score moyen des bboxes < 0.7 → ⚠️ doc à revérifier
- Plus de 2 fields manquants sur 6 → ⚠️ photo de doc floue/coupée

---

## 8. Compute & coûts

### 8.1 Recommandation : Colab free

- **GPU :** Tesla T4 (16 GB VRAM)
- **Disque :** /content (~70 GB libre)
- **Limite :** 12h max session, déco après 90 min d'inactivité
- **Coût :** 0 €

### 8.2 Estimation

| Étape | Durée Colab T4 |
|---|---|
| Download MIDV-2020 photo (~2 GB) | 3-5 min |
| Conversion VIA → YOLO | < 1 min |
| Training 50 epochs YOLOv11s | 25-40 min |
| Eval + visualisations | 5 min |
| **Total session** | **~50 min** |

Marge confortable sous la limite Colab free.

---

## 9. Roadmap & extensions (bonus pour la soutenance)

### Niveau 1 — MVP (livré dans le notebook)
- ✅ Détection 6 classes sur MIDV-2020 photo
- ✅ Eval mAP + matrice de confusion
- ✅ Function `predict_and_crop()` pour l'aval

### Niveau 2 — Améliorations défendables
- Ajout des scans MIDV-2020 (2000 images en plus → +30 % de données)
- Test-time augmentation (TTA) → +1-2 pts mAP gratuits via `val(augment=True)`
- Export ONNX pour déploiement mobile : `model.export(format='onnx')`

### Niveau 3 — Production-ready
- Pipeline 2 étapes : YOLOv11 doc-corners (MIDV-500) → rectification homographique → YOLOv11 fields (MIDV-2020 sur images rectifiées). Boost mAP attendu : +5-8 pts car le modèle field-level voit toujours le doc dans la même orientation.
- Fine-tune sur dataset privé avec docs FR (CNI, passeport, permis FR) — MIDV est mock, donc en prod il faut compléter.
- Active learning : flagger les détections low-confidence pour annotation humaine.

---

## 10. Pour la soutenance — 3 points à marteler

1. **« J'ai choisi MIDV-2020 plutôt que MIDV-500 brut parce que MIDV-500 n'annote que les coins du document, pas les champs internes. MIDV-2020 est l'évolution officielle de la même famille avec les annotations field-level requises. »** → montre que tu as lu les datasets, pas juste pris le premier lien.

2. **« YOLOv11 vs RF-DETR : j'ai privilégié YOLOv11 pour la maturité de l'écosystème et la simplicité du fine-tuning ; RF-DETR donnerait probablement +2-3 pts mAP mais avec un coût de complexité bien supérieur pour un gain marginal sur ce dataset. »** → montre que tu as comparé.

3. **« J'ai désactivé le flip horizontal en augmentation parce que les dates et la MRZ ne sont pas symétriques — sinon le modèle apprend des features inversés inutiles. »** → un petit détail technique qui montre que tu as réfléchi au domaine.

---

## Sources

- [fcakyon/midv500 (GitHub)](https://github.com/fcakyon/midv500)
- [MIDV-2020 dataset (L3i, Université de La Rochelle)](https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
- [MIDV-2020 paper (arXiv)](https://arxiv.org/abs/2107.00396)
- [Ultralytics YOLOv11 docs](https://docs.ultralytics.com/models/yolo11/)
- [VIA annotator v2 format](https://www.robots.ox.ac.uk/~vgg/software/via/)
