"""Verification de la date d'expiration via docTR.

docTR (Document Text Recognition) est un OCR deep learning specialise
pour les documents : detection de texte (DBNet) + reconnaissance (CRNN).
Il gere les photos reelles de documents (angles, hologrammes, reflets)
bien mieux que Tesseract ou EasyOCR.

Requires: pip install python-doctr[torch]
"""
from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np

# Cache le predictor docTR (lourd a charger)
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from doctr.models import ocr_predictor
        _predictor = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
    return _predictor


def lire_date_expiration(doc_image_bgr: np.ndarray) -> dict:
    """Lit la date d'expiration depuis l'image du document via docTR.

    Parameters
    ----------
    doc_image_bgr : image BGR du document (numpy array)

    Returns
    -------
    dict avec:
      - date_texte: str — la date lue
      - date_iso: str | None — format YYYY-MM-DD
      - ocr_complet: str — tout le texte OCR
      - success: bool
      - error: str | None
    """
    import cv2
    from doctr.io import DocumentFile

    predictor = _get_predictor()

    # Upscale agressif pour les petits crops (dates sur passeports FR = ~20px de haut)
    # docTR a besoin d'au moins ~500px pour lire correctement
    h, w = doc_image_bgr.shape[:2]
    img_to_ocr = doc_image_bgr
    is_crop = max(h, w) < 300  # Petit crop = probablement juste la zone date

    if max(h, w) < 500:
        scale = 500 / max(h, w)
        img_to_ocr = cv2.resize(doc_image_bgr, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
    elif max(h, w) < 1500:
        scale = 1500 / max(h, w)
        img_to_ocr = cv2.resize(doc_image_bgr, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)

    # Sauvegarder en temp (docTR prend un chemin fichier)
    import tempfile
    from pathlib import Path
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img_to_ocr)

    try:
        doc = DocumentFile.from_images(tmp.name)
        result = predictor(doc)
    except Exception as e:
        Path(tmp.name).unlink(missing_ok=True)
        return {"date_texte": None, "date_iso": None, "ocr_complet": "",
                "success": False, "error": str(e)}

    Path(tmp.name).unlink(missing_ok=True)

    # Extraire toutes les lignes avec leur texte et position
    lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                text = " ".join(w.value for w in line.words)
                conf = min(w.confidence for w in line.words) if line.words else 0
                # Position Y moyenne (normalise 0-1)
                y_center = (line.geometry[0][1] + line.geometry[1][1]) / 2
                lines.append({
                    "text": text.strip(),
                    "conf": conf,
                    "y": y_center,
                })

    if not lines:
        return {"date_texte": None, "date_iso": None, "ocr_complet": "",
                "success": False, "error": "Aucun texte detecte"}

    ocr_complet = "\n".join(f"[{l['conf']:.0%}] {l['text']}" for l in lines)

    # Trouver la date d'expiration — 3 strategies :
    # 1. MRZ (passeports) — le plus fiable, format standardise mondial
    # 2. Mots-cles (VALID UNTIL, VALABLE...) — pour les CNI, titres de sejour
    # 3. Date la plus tardive — fallback universel
    date_texte, date_iso = None, None

    if is_crop:
        # Petit crop = juste la zone date, prendre la premiere date
        all_dates = []
        for line in lines:
            all_dates.extend(extraire_dates(line["text"]))
        if all_dates:
            date_texte, date_iso = all_dates[0]
    else:
        # Document entier : essayer MRZ d'abord (passeports)
        date_texte, date_iso = extraire_date_mrz(lines)

        # Si pas de MRZ, chercher par mots-cles puis fallback
        if not date_iso:
            date_texte, date_iso = trouver_date_expiration(lines)

    return {
        "date_texte": date_texte,
        "date_iso": date_iso,
        "ocr_complet": ocr_complet,
        "success": date_iso is not None,
        "error": None if date_iso else "Date d'expiration non trouvee",
    }


def trouver_date_expiration(lines: list[dict]) -> tuple[str | None, str | None]:
    """Trouve la date d'expiration dans les lignes OCR.

    Strategie double :
    1. Par mot-cle : cherche VALABLE/VALID UNTIL/EXPIR puis la date juste apres
    2. Fallback : prend la date la plus tardive (= toujours l'expiration)
    """
    keywords = [
        "valable", "valid until", "expir", "validite", "expire",
    ]

    all_dates = []

    for i, line in enumerate(lines):
        # Extraire les dates de cette ligne
        dates_in_line = extraire_dates(line["text"])
        all_dates.extend(dates_in_line)

        # Verifier si cette ligne contient un mot-cle d'expiration
        text_lower = line["text"].lower()
        for kw in keywords:
            if kw in text_lower:
                # La date peut etre sur la meme ligne
                if dates_in_line:
                    return dates_in_line[0]
                # Ou sur la ligne suivante
                if i + 1 < len(lines):
                    next_dates = extraire_dates(lines[i + 1]["text"])
                    if next_dates:
                        return next_dates[0]

    # Fallback : la date la plus tardive = expiration
    if all_dates:
        all_dates.sort(key=lambda x: x[1])
        return all_dates[-1]

    return (None, None)


def extraire_date_mrz(lines: list[dict]) -> tuple[str | None, str | None]:
    """Extrait la date d'expiration depuis la MRZ (Machine Readable Zone).

    Format MRZ TD3 (passeports), ligne 2 :
    PPPPPPPPPCNNNDDMMDDFSYYMMDDCXXXXXXXXXXXXXX

    La date d'expiration est en position 21-27 (YYMMDD) de la ligne 2.
    La MRZ se reconnait par les caracteres '<' et le format alphanumerique.
    """
    for line in lines:
        text = line["text"].replace(" ", "").replace(".", "")

        # Detecter une ligne MRZ : contient des '<' et fait ~44 chars
        if "<" in text and len(text) > 30:
            # Nettoyer : garder seulement alphanumerique et <
            clean = re.sub(r"[^A-Z0-9<]", "", text.upper())

            if len(clean) >= 28:
                # Chercher le pattern : 3 lettres (nationalite) + 6 chiffres (DOB) + 1 chiffre + 1 lettre (sexe) + 6 chiffres (expiry)
                match = re.search(r"[A-Z]{3}(\d{6})\d[MF<](\d{6})", clean)
                if match:
                    expiry_raw = match.group(2)  # YYMMDD
                    yy = int(expiry_raw[0:2])
                    mm = int(expiry_raw[2:4])
                    dd = int(expiry_raw[4:6])

                    # Convention MRZ : YY < 30 = 20XX, YY >= 30 = 19XX
                    year = 2000 + yy if yy < 50 else 1900 + yy

                    try:
                        dt = datetime(year, mm, dd)
                        texte_date = f"{dd:02d}/{mm:02d}/{year}"
                        return (f"{texte_date} (MRZ)", dt.strftime("%Y-%m-%d"))
                    except ValueError:
                        pass

    return (None, None)


def extraire_dates(texte: str) -> list[tuple[str, str]]:
    """Extrait toutes les dates d'une ligne de texte.

    Returns: liste de (texte_original, date_iso)
    """
    results = []

    # DD/MM/YYYY ou DD-MM-YYYY ou DD.MM.YYYY
    for match in re.finditer(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", texte):
        d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            dt = datetime(y, m, d)
            results.append((match.group(0), dt.strftime("%Y-%m-%d")))
        except ValueError:
            pass

    # DD MM YYYY (espaces)
    for match in re.finditer(r"(\d{1,2})\s+(\d{1,2})\s+(\d{4})", texte):
        d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            dt = datetime(y, m, d)
            results.append((match.group(0), dt.strftime("%Y-%m-%d")))
        except ValueError:
            pass

    return results


def verifier_expiration(date_iso: str | None) -> dict:
    """Verifie si le document est expire."""
    aujourdhui = date.today()
    result = {
        "est_expire": None,
        "date_expiration": None,
        "date_aujourdhui": aujourdhui.isoformat(),
        "jours_restants": None,
        "statut": "INCONNU",
    }

    if not date_iso:
        return result

    try:
        date_exp = datetime.strptime(date_iso, "%Y-%m-%d").date()
        result["date_expiration"] = date_exp.isoformat()
        delta = (date_exp - aujourdhui).days
        result["jours_restants"] = delta
        result["est_expire"] = delta < 0
        result["statut"] = "EXPIRE" if delta < 0 else "VALIDE"
    except ValueError:
        pass

    return result
