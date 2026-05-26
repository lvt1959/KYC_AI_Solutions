# Etape 2.2 — Face Matching

**Responsable :** (a assigner)
**Statut :** A faire

## Objectif

Comparer la photo extraite du document d'identite (etape 2.1) avec un selfie du porteur pour verifier l'identite.

```
[crop photo doc (etape 2.1)] + [selfie] --> [face match] --> { match: bool, score: float }
```

## Input attendu (depuis etape 2.1)

```python
# Le crop de la photo est disponible via :
result = predict_and_crop("doc.jpg", model_path="...")
photo_crop = result["kyc_critical"]["photo"]["crop_path"]
# --> "crops/doc/face_image.jpg"
```

## Output attendu (vers etape 3)

```python
{
    "match": True,           # True si la personne correspond
    "confidence": 0.94,      # Score de similarite
    "photo_crop": "...",     # Chemin du crop utilise
    "selfie_path": "...",    # Chemin du selfie
}
```

## Pistes techniques

- **face_recognition** (dlib) — simple, performant, CPU-friendly
- **InsightFace** — SOTA, plus precis, GPU recommande
- **DeepFace** — wrapper multi-backend (VGG-Face, ArcFace, Facenet)

## Pour demarrer

```bash
cd step_2_2_face_match
pip install -r requirements.txt  # (a creer)
```
