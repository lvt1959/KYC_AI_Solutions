"""
KYC AI Solutions — Application Web
===================================
Pipeline complet de verification d'identite :
  1. Classification du document (CNN)
  2. Detection des champs (YOLO11n, 33 classes)
  3. Verification biometrique (Face Match)
  4. Generation du rapport legal (LLM)

Usage:
    streamlit run app.py
"""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ──
st.set_page_config(
    page_title="KYC AI Solutions",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 10px; padding: 12px 16px;
        border-left: 4px solid #1D9E75;
    }
    .status-approved { color: #1D9E75; font-weight: bold; font-size: 1.3em; }
    .status-rejected { color: #D85A30; font-weight: bold; font-size: 1.3em; }
    .status-review   { color: #E6A817; font-weight: bold; font-size: 1.3em; }
    .step-header {
        background: linear-gradient(90deg, #185FA5 0%, #1D9E75 100%);
        color: white; padding: 10px 20px; border-radius: 8px; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Session State Initialization
# =====================================================================
DEFAULTS = {
    "step": 1,
    # Step 1 results
    "doc_image": None,
    "doc_bgr": None,
    "classification": None,
    # Step 2.1 results
    "detection": None,
    "validation": None,
    # Step 2.2 results
    "selfie_image": None,
    "face_match": None,
    # Step 3 results
    "rapport": None,
    "result_json": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =====================================================================
# Sidebar — Configuration
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/identification-documents.png", width=64)
    st.title("KYC AI Solutions")
    st.caption("Pipeline de verification d'identite")

    st.divider()
    st.subheader("Modeles")

    classifier_path = st.text_input(
        "CNN Classifier (.keras)",
        value="models/kyc_classifier_best.keras",
        help="Chemin vers le modele TensorFlow de classification (etape 1).",
    )
    detector_path = st.text_input(
        "YOLO Detector (.pt)",
        value="models/best.pt",
        help="Chemin vers les poids YOLO11n de detection des champs (etape 2.1).",
    )

    st.divider()
    st.subheader("OpenRouter (Etape 4)")
    openrouter_key = st.text_input(
        "Cle API OpenRouter",
        type="password",
        help="Necessaire pour generer le rapport KYC legal.",
    )
    llm_model = st.text_input(
        "Modele LLM",
        value="openrouter/owl-alpha",
        help="Ex: openrouter/owl-alpha, openrouter/auto, anthropic/claude-3.5-sonnet, etc.",
    )

    st.divider()

    # Reset button
    if st.button("Nouvelle verification", type="primary", use_container_width=True):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val
        st.rerun()

    # Progress tracker
    st.divider()
    st.subheader("Progression")
    steps = [
        ("1. Classification", st.session_state.classification is not None),
        ("2. Detection champs", st.session_state.detection is not None),
        ("3. Face Match", st.session_state.face_match is not None),
        ("4. Rapport KYC", st.session_state.rapport is not None),
    ]
    for name, done in steps:
        icon = "✅" if done else "⬜"
        st.write(f"{icon}  {name}")


# =====================================================================
# Helpers
# =====================================================================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR numpy to RGB for display in Streamlit."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def file_exists(path: str) -> bool:
    """Check if a model file exists (absolute or relative to app dir)."""
    p = Path(path)
    if p.is_absolute():
        return p.exists()
    # Check relative to app directory
    app_dir = Path(__file__).parent
    return (app_dir / p).exists() or p.exists()


def resolve_path(path: str) -> Path:
    """Resolve a model path (absolute or relative to app dir)."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    app_dir = Path(__file__).parent
    if (app_dir / p).exists():
        return app_dir / p
    return p


@st.cache_resource
def cached_load_classifier(path: str):
    from pipeline.classifier import load_classifier
    return load_classifier(resolve_path(path))


@st.cache_resource
def cached_load_detector(path: str):
    from pipeline.detector import load_detector
    return load_detector(resolve_path(path))


@st.cache_resource
def cached_load_face_models():
    from pipeline.face_match import load_face_models
    return load_face_models()


# =====================================================================
# Title
# =====================================================================
st.markdown("""
<div class="step-header">
    <h1 style="margin:0; font-size:1.6em;">🪪 KYC AI Solutions — Verification d'Identite</h1>
    <p style="margin:4px 0 0; opacity:0.9;">Pipeline automatise : Classification → Detection → Face Match → Rapport</p>
</div>
""", unsafe_allow_html=True)

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣  Classification",
    "2️⃣  Detection des champs",
    "3️⃣  Face Match",
    "4️⃣  Rapport KYC",
])


# =====================================================================
# TAB 1 — Document Classification
# =====================================================================
with tab1:
    st.header("Etape 1 — Classification du document")
    st.info("Uploadez votre document d'identite (CNI ou Passeport). Le modele CNN va determiner automatiquement le type de document.")

    uploaded_doc = st.file_uploader(
        "Document d'identite",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="upload_doc",
        help="Formats supportes : JPG, PNG, WebP, BMP",
    )

    if uploaded_doc is not None:
        pil_img = Image.open(uploaded_doc)
        st.session_state.doc_image = pil_img
        st.session_state.doc_bgr = pil_to_bgr(pil_img)

        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(pil_img, caption="Document uploade", use_container_width=True)

        with col_res:
            if not file_exists(classifier_path):
                st.warning(f"⚠️ Modele introuvable : `{classifier_path}`")
                st.markdown("""
**Comment obtenir le modele :**
1. Ouvrir `step_1_classification/network.py` dans Google Colab
2. Entrainer le CNN sur le dataset MIDV-2020
3. Telecharger `kyc_classifier_best.keras`
4. Le placer dans `app/models/`
                """)
                # Allow manual classification
                st.divider()
                st.subheader("Classification manuelle")
                manual_type = st.selectbox(
                    "Type de document",
                    ["Carte Nationale d'Identite", "Passeport"],
                )
                manual_conf = st.slider("Confiance estimee", 0.0, 1.0, 0.95, 0.01)
                if st.button("Valider la classification manuelle", type="primary"):
                    st.session_state.classification = {
                        "type_document": manual_type,
                        "confiance": manual_conf,
                        "classe_brute": "id" if "Identite" in manual_type else "passport",
                        "mode": "manuel",
                    }
                    st.rerun()
            else:
                if st.button("Lancer la classification", type="primary", use_container_width=True):
                    with st.spinner("Classification en cours..."):
                        from pipeline.classifier import classify_document
                        model = cached_load_classifier(classifier_path)
                        result = classify_document(st.session_state.doc_bgr, model)
                        result["mode"] = "auto"
                        st.session_state.classification = result
                        st.rerun()

    # Display results
    if st.session_state.classification:
        cls = st.session_state.classification
        st.divider()
        st.subheader("Resultat")

        c1, c2, c3 = st.columns(3)
        c1.metric("Type de document", cls["type_document"])
        c2.metric("Confiance", f"{cls['confiance']:.0%}")
        c3.metric("Mode", "Automatique" if cls.get("mode") == "auto" else "Manuel")

        if cls["confiance"] >= 0.90:
            st.success(f"✅ Document identifie avec haute confiance : **{cls['type_document']}**")
        elif cls["confiance"] >= 0.70:
            st.warning(f"⚠️ Confiance moyenne — verifier manuellement : **{cls['type_document']}**")
        else:
            st.error(f"❌ Confiance faible — document potentiellement non conforme")

        st.info("➡️ Passez a l'onglet **Detection des champs** pour continuer.")


# =====================================================================
# TAB 2 — Field Detection
# =====================================================================
with tab2:
    st.header("Etape 2.1 — Detection des champs du document")

    if st.session_state.doc_bgr is None:
        st.warning("⚠️ Veuillez d'abord uploader un document dans l'etape 1.")
    elif st.session_state.classification is None:
        st.warning("⚠️ Veuillez d'abord classifier le document dans l'etape 1.")
    else:
        st.info("Le modele YOLO11n va detecter les 33 champs du document (photo, date d'expiration, MRZ, nom, etc.).")

        if not file_exists(detector_path):
            st.warning(f"⚠️ Modele introuvable : `{detector_path}`")
            st.markdown("""
**Comment obtenir le modele :**
1. Ouvrir `step_2_1_detection/notebooks/training_colab.ipynb` dans Google Colab
2. Entrainer YOLO11n sur le dataset GenMRP+MIDV-2020
3. Telecharger `best.pt` depuis `runs/detect/train/weights/`
4. Le placer dans `app/models/`
            """)

            # Manual fallback
            st.divider()
            st.subheader("Mode manuel")
            manual_photo = st.checkbox("Photo detectee sur le document", value=True)
            manual_expiry = st.text_input("Date d'expiration (YYYY-MM-DD)", value="2030-01-01")

            if st.button("Valider la detection manuelle", type="primary"):
                st.session_state.detection = {
                    "detections": {},
                    "kyc_critical": {
                        "photo": {"conf": 0.95, "crop_path": None} if manual_photo else None,
                        "date_of_expiry": {"conf": 0.90, "crop_path": None} if manual_expiry else None,
                    },
                    "missing_fields": [],
                    "annotated_image": None,
                    "mode": "manuel",
                }
                st.session_state.validation = {
                    "status": "OK" if manual_photo else "REJECTED",
                    "reasons": [] if manual_photo else ["Photo non detectee"],
                }
                st.session_state._manual_expiry = manual_expiry
                st.rerun()
        else:
            if st.button("Lancer la detection", type="primary", use_container_width=True):
                with st.spinner("Detection des champs en cours..."):
                    from pipeline.detector import predict_and_crop, validate_for_kyc
                    model = cached_load_detector(detector_path)
                    result = predict_and_crop(
                        st.session_state.doc_bgr,
                        model,
                        image_name="document",
                    )
                    result["mode"] = "auto"
                    st.session_state.detection = result
                    st.session_state.validation = validate_for_kyc(result)
                    st.rerun()

        # Display results
        if st.session_state.detection is not None:
            det = st.session_state.detection
            val = st.session_state.validation
            st.divider()

            # Annotated image
            if det.get("annotated_image") is not None:
                st.subheader("Document annote")
                st.image(bgr_to_rgb(det["annotated_image"]), use_container_width=True)

            # Validation status
            st.subheader("Validation KYC")
            status = val["status"]
            if status == "OK":
                st.markdown('<p class="status-approved">✅ STATUT : OK — Document conforme</p>', unsafe_allow_html=True)
            elif status == "REVIEW":
                st.markdown('<p class="status-review">⚠️ STATUT : REVUE MANUELLE</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-rejected">❌ STATUT : REJETE</p>', unsafe_allow_html=True)

            if val.get("reasons"):
                for r in val["reasons"]:
                    st.write(f"  - {r}")

            # Detection details
            detections = det.get("detections", {})
            if detections:
                st.subheader(f"Champs detectes ({len(detections)})")
                det_data = []
                for cls_name, info in sorted(detections.items(), key=lambda x: -x[1]["conf"]):
                    det_data.append({
                        "Champ": cls_name,
                        "Confiance": f"{info['conf']:.0%}",
                        "Critique": "🔑" if cls_name in ("face_image", "date_of_expiry") else "",
                    })
                st.dataframe(det_data, use_container_width=True, hide_index=True)

            # Photo crop preview
            crit = det.get("kyc_critical", {})
            if crit.get("photo") and crit["photo"].get("crop_path"):
                crop_path = Path(crit["photo"]["crop_path"])
                if crop_path.exists():
                    st.subheader("Photo extraite du document")
                    crop_img = cv2.imread(str(crop_path))
                    st.image(bgr_to_rgb(crop_img), width=200)

            if status != "REJECTED":
                st.info("➡️ Passez a l'onglet **Face Match** pour la verification biometrique.")


# =====================================================================
# TAB 3 — Face Match
# =====================================================================
with tab3:
    st.header("Etape 2.2 — Verification biometrique (Face Match)")

    if st.session_state.detection is None:
        st.warning("⚠️ Veuillez d'abord completer la detection des champs (etape 2).")
    elif st.session_state.validation and st.session_state.validation["status"] == "REJECTED":
        st.error("❌ Le document a ete rejete a l'etape 2. La verification biometrique n'est pas possible.")
    else:
        st.info("Uploadez un selfie pour comparer avec la photo du document. Le systeme utilise ArcFace (512 dimensions) pour la comparaison.")

        uploaded_selfie = st.file_uploader(
            "Selfie de l'utilisateur",
            type=["jpg", "jpeg", "png", "webp", "bmp", "heic"],
            key="upload_selfie",
            help="Photo de face, bien eclairee, regard vers la camera.",
        )

        if uploaded_selfie is not None:
            col_selfie, col_id = st.columns(2)

            with col_selfie:
                selfie_pil = Image.open(uploaded_selfie)
                st.image(selfie_pil, caption="Selfie uploade", use_container_width=True)

            with col_id:
                # Show the ID photo crop if available
                crit = st.session_state.detection.get("kyc_critical", {})
                if crit.get("photo") and crit["photo"].get("crop_path"):
                    crop_path = Path(crit["photo"]["crop_path"])
                    if crop_path.exists():
                        id_crop = cv2.imread(str(crop_path))
                        st.image(bgr_to_rgb(id_crop), caption="Photo ID (extraite)", use_container_width=True)
                else:
                    st.image(st.session_state.doc_image, caption="Document original", use_container_width=True)

            if st.button("Lancer le Face Match", type="primary", use_container_width=True):
                with st.spinner("Chargement des modeles (premiere fois: ~30s)..."):
                    yolo_face, arcface = cached_load_face_models()

                with st.spinner("Verification biometrique en cours..."):
                    from pipeline.face_match import run_face_match, pretraiter_image

                    # Get the ID photo
                    if crit.get("photo") and crit["photo"].get("crop_path"):
                        crop_path = Path(crit["photo"]["crop_path"])
                        if crop_path.exists():
                            photo_id_bgr = cv2.imread(str(crop_path))
                        else:
                            photo_id_bgr = st.session_state.doc_bgr
                    else:
                        photo_id_bgr = st.session_state.doc_bgr

                    # Save selfie to temp for processing
                    selfie_bgr = pil_to_bgr(selfie_pil)

                    try:
                        match_result = run_face_match(
                            photo_id_bgr=photo_id_bgr,
                            selfie_input=selfie_bgr,
                            yolo_model=yolo_face,
                            arcface_app=arcface,
                        )
                        st.session_state.face_match = match_result
                        st.session_state.selfie_image = selfie_pil
                        st.rerun()
                    except ValueError as e:
                        st.error(f"❌ Erreur : {e}")

        # Display results
        if st.session_state.face_match is not None:
            fm = st.session_state.face_match
            res = fm["result"]
            st.divider()

            # Cropped faces side by side
            st.subheader("Visages compares")
            col_a, col_b = st.columns(2)
            with col_a:
                if fm.get("crop_selfie") is not None:
                    st.image(bgr_to_rgb(fm["crop_selfie"]), caption="Visage selfie", use_container_width=True)
            with col_b:
                if fm.get("crop_id") is not None:
                    st.image(bgr_to_rgb(fm["crop_id"]), caption="Visage document", use_container_width=True)

            # Metrics
            st.subheader("Resultat")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Decision", "MATCH ✅" if res["est_match"] else "NO MATCH ❌")
            c2.metric("Score", f"{res['score']}%")
            c3.metric("Distance cosine", f"{res['distance']}")
            c4.metric("Confiance", res["confiance"])

            if res["est_match"]:
                st.success(f"✅ **MEME PERSONNE** — Score de similarite : {res['score']}% (confiance {res['confiance']})")
            else:
                st.error(f"❌ **PERSONNES DIFFERENTES** — Score de similarite : {res['score']}% (distance {res['distance']} > seuil {res['seuil']})")

            st.caption(f"Seuil KYC strict = {res['seuil']} | Distance cosine ∈ [0, 2] | Plus la distance est faible, plus les visages sont similaires.")

            st.info("➡️ Passez a l'onglet **Rapport KYC** pour generer le rapport legal.")


# =====================================================================
# TAB 4 — KYC Report
# =====================================================================
with tab4:
    st.header("Etape 3 — Generation du rapport KYC")

    if st.session_state.face_match is None:
        st.warning("⚠️ Veuillez d'abord completer le Face Match (etape 3).")
    else:
        cls = st.session_state.classification
        det = st.session_state.detection
        fm = st.session_state.face_match["result"]
        crit = det.get("kyc_critical", {})

        # Gather all data
        type_document = cls["type_document"]
        confiance_doc = cls["confiance"]
        photo_detectee = crit.get("photo") is not None
        # Try to read expiry from detection or manual input
        date_expiration = getattr(st.session_state, "_manual_expiry", "Non detectee")
        if crit.get("date_of_expiry"):
            date_expiration = date_expiration if date_expiration != "Non detectee" else "Detectee (valeur non extraite)"

        est_match = fm["est_match"]
        score_similarite = fm["score"]
        confiance_match = fm["confiance"]
        distance = fm["distance"]

        # Summary before generation
        st.subheader("Resume des resultats")

        from pipeline.report import determiner_statut
        statut = determiner_statut(est_match, photo_detectee, date_expiration, confiance_doc)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Document**")
            st.write(f"- Type : {type_document}")
            st.write(f"- Confiance : {confiance_doc:.0%}")
            st.write(f"- Photo : {'Detectee' if photo_detectee else 'Non detectee'}")
            st.write(f"- Expiration : {date_expiration}")

        with col2:
            st.markdown("**Biometrie**")
            st.write(f"- Face Match : {'MATCH' if est_match else 'NO MATCH'}")
            st.write(f"- Score : {score_similarite}%")
            st.write(f"- Confiance : {confiance_match}")
            st.write(f"- Distance : {distance}")

        # Global status
        if statut == "APPROUVE":
            st.markdown('<p class="status-approved">✅ STATUT GLOBAL : APPROUVE</p>', unsafe_allow_html=True)
        elif statut == "REJETE":
            st.markdown('<p class="status-rejected">❌ STATUT GLOBAL : REJETE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-review">⚠️ STATUT GLOBAL : REVUE MANUELLE</p>', unsafe_allow_html=True)

        st.divider()

        # Generate report
        if not openrouter_key:
            st.warning("⚠️ Cle API OpenRouter requise (sidebar) pour generer le rapport.")
            st.markdown("""
**Pour obtenir une cle :**
1. Creer un compte sur [openrouter.ai](https://openrouter.ai)
2. Recuperer votre cle API
3. La coller dans le champ de la sidebar
            """)
        else:
            if st.button("Generer le rapport KYC", type="primary", use_container_width=True):
                with st.spinner("Generation du rapport via LLM..."):
                    from pipeline.report import generer_rapport, build_result_json

                    try:
                        rapport = generer_rapport(
                            type_document=type_document,
                            confiance_doc=confiance_doc,
                            photo_detectee=photo_detectee,
                            date_expiration=date_expiration,
                            est_match=est_match,
                            score_similarite=score_similarite,
                            confiance_match=confiance_match,
                            distance=distance,
                            api_key=openrouter_key,
                            model=llm_model,
                        )
                        st.session_state.rapport = rapport
                        st.session_state.result_json = build_result_json(
                            statut=statut,
                            type_document=type_document,
                            confiance_doc=confiance_doc,
                            photo_detectee=photo_detectee,
                            date_expiration=date_expiration,
                            est_match=est_match,
                            score_similarite=score_similarite,
                            rapport_markdown=rapport,
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la generation : {e}")

        # Display report
        if st.session_state.rapport:
            st.divider()
            st.subheader("Rapport genere")
            st.markdown(st.session_state.rapport)

            # Downloads
            st.divider()
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="Telecharger le rapport (.md)",
                    data=st.session_state.rapport,
                    file_name=f"rapport_kyc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_dl2:
                if st.session_state.result_json:
                    st.download_button(
                        label="Telecharger les resultats (.json)",
                        data=json.dumps(st.session_state.result_json, ensure_ascii=False, indent=2),
                        file_name=f"rapport_kyc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )


# =====================================================================
# Footer
# =====================================================================
st.divider()
st.caption("KYC AI Solutions — Equipe lvt1959 | CNN + YOLO11n + ArcFace + LLM | 2026")
