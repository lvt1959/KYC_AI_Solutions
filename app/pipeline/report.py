"""Step 3 — KYC report generation via OpenRouter LLM.

Uses OpenAI-compatible client pointed at openrouter.ai.
Model: openrouter/auto (or configurable).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone


def determiner_statut(est_match: bool, photo_detectee: bool,
                      date_expiration: str, confiance_doc: float) -> str:
    """Determine the global KYC status from all step results.

    Returns: APPROUVE, REJETE, or REVUE MANUELLE.
    """
    if not est_match:
        return "REJETE"
    if not photo_detectee:
        return "REJETE"
    if date_expiration in ("Non detectee", "", None):
        return "REVUE MANUELLE"
    if confiance_doc < 0.70:
        return "REVUE MANUELLE"
    return "APPROUVE"


def generer_rapport(
    type_document: str,
    confiance_doc: float,
    photo_detectee: bool,
    date_expiration: str,
    est_match: bool,
    score_similarite: float,
    confiance_match: str,
    distance: float,
    api_key: str,
    model: str = "openrouter/auto",
) -> str:
    """Generate a legal KYC report via OpenRouter.

    Returns the report as a Markdown string.
    """
    from openai import OpenAI

    statut = determiner_statut(est_match, photo_detectee, date_expiration, confiance_doc)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    system_prompt = """
Tu es un expert en conformite KYC (Know Your Customer).
Tu rediges des rapports de verification d'identite formels, factuels et juridiquement precis.
Ton langage est professionnel, objectif et oriente vers la decision finale.
Tu utilises le format Markdown avec les sections suivantes uniquement :

# RAPPORT DE VERIFICATION KYC

## 1. Resume Executif
## 2. Analyse du Document d'Identite
## 3. Verification Biometrique
## 4. Conclusion Legale

Regles strictes :
- Factuel uniquement, ne pas inventer d'informations
- Convertir les scores decimaux en pourcentages
- Utiliser uniquement les emojis : OK, ATTENTION, ERREUR pour les statuts
- Langage juridique et professionnel
"""

    user_prompt = f"""
Voici les resultats de la verification KYC a rapporter :

ETAPE 1 - Classification du document :
- Type de document : {type_document}
- Confiance de classification : {confiance_doc:.0%}

ETAPE 2.1 - Detection des elements cles :
- Photo sur le document : {'Detectee' if photo_detectee else 'Non detectee'}
- Date d'expiration : {date_expiration}

ETAPE 2.2 - Verification biometrique (Face Match) :
- Resultat : {'MATCH - Meme personne' if est_match else 'NO MATCH - Personnes differentes'}
- Score de similarite : {score_similarite}%
- Niveau de confiance : {confiance_match}
- Distance cosine : {distance} (seuil KYC : 0.30)

STATUT GLOBAL DETERMINE : {statut}
"""

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/lvt1959/KYC_AI_Solutions",
            "X-Title": "KYC AI Solutions",
        },
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return completion.choices[0].message.content


def build_result_json(
    statut: str,
    type_document: str,
    confiance_doc: float,
    photo_detectee: bool,
    date_expiration: str,
    est_match: bool,
    score_similarite: float,
    rapport_markdown: str,
) -> dict:
    """Build the final structured JSON result."""
    return {
        "statut": statut,
        "type_document": type_document,
        "confiance_doc": confiance_doc,
        "photo_detectee": photo_detectee,
        "date_expiration": date_expiration,
        "face_match": est_match,
        "score_similarite": score_similarite,
        "rapport_markdown": rapport_markdown,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
