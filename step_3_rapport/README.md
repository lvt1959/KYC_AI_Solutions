# Etape 3 — Rapport KYC (LLM)

**Responsable :** (a assigner)
**Statut :** A faire

## Objectif

Generer un rapport KYC structure a partir des resultats des etapes 2.1 (detection) et 2.2 (face-match). Un LLM synthetise toutes les informations pour produire un verdict final.

```
[detections (2.1)] + [face match (2.2)] --> [LLM] --> rapport KYC structure
```

## Inputs attendus

### Depuis etape 2.1 (detection)

```python
from step_2_1_detection.src.inference import predict_and_crop, validate_for_kyc

result = predict_and_crop("doc.jpg", model_path="...")
# result["detections"]     --> tous les champs detectes avec bbox + conf + crop
# result["kyc_critical"]   --> photo + date_of_expiry
# result["missing_fields"] --> champs attendus mais non trouves

verdict_2_1 = validate_for_kyc(result)
# {"status": "OK" | "REVIEW" | "REJECTED", "reasons": [...]}
```

### Depuis etape 2.2 (face-match)

```python
face_result = {
    "match": True,
    "confidence": 0.94,
    "photo_crop": "crops/doc/face_image.jpg",
    "selfie_path": "selfie.jpg",
}
```

## Output attendu

```python
{
    "decision": "APPROVED" | "MANUAL_REVIEW" | "REJECTED",
    "confidence": 0.91,
    "summary": "Document valide. Photo correspond au selfie. Date d'expiration 2028.",
    "checks": {
        "document_detected": True,
        "photo_match": True,
        "document_expired": False,
        "all_fields_present": True,
    },
    "flags": [],
}
```

## Pistes techniques

- **Claude API** (Anthropic) — vision + texte, ideal pour analyser les crops
- **GPT-4o** (OpenAI) — alternative multimodale
- **Prompt engineering** : passer les crops + metadonnees en contexte, demander un JSON structure

## Pour demarrer

```bash
cd step_3_rapport
pip install -r requirements.txt  # (a creer)
```
