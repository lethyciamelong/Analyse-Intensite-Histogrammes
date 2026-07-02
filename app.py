import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from skimage.feature import local_binary_pattern, hog
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
import time

# =============================================================================
# DESIGN SYSTEM : ACADEMIC PREMIUM v5.1
# =============================================================================
st.set_page_config(
    page_title="Vision Par Ordinateur Lab | Invariance & Robustesse",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;700&family=Montserrat:wght@300;400;700;900&display=swap');
    
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stMarkdown code { font-family: 'Source Code Pro', monospace; }
    
    .main { background: #0b0e14; }
    .stApp { background: radial-gradient(circle at 50% 50%, #10141d 0%, #0b0e14 100%); }
    
    .brand-header {
        background: linear-gradient(90deg, #00d4ff11 0%, #00ff7f08 100%);
        padding: 2rem;
        border-radius: 20px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 25px;
    }

    .sci-unit {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 30px;
        position: relative;
    }
    .unit-label {
        position: absolute;
        top: -12px;
        left: 20px;
        background: #00d4ff;
        color: #000;
        padding: 2px 15px;
        border-radius: 10px;
        font-size: 0.65rem;
        font-weight: 900;
        letter-spacing: 1px;
    }

    .score-box {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .score-title { color: #888; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
    .score-val { font-size: 2.2rem; font-weight: 900; font-family: 'Source Code Pro', monospace; }
    
    .status-ok { color: #00ff7f; }
    .status-bad { color: #ff4b4b; }

    .calc-note {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SCIENTIFIC PROCESSING KERNEL
# =============================================================================

def convert_to_gray(img_rgb):
    """Projection de Luminosité Invariante: Y = 0.299R + 0.587G + 0.114B"""
    return np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def apply_scientific_stress(img, scenario, intensity, mode="8-bit"):
    img_f = img.astype(np.float32)
    
    def process(chan, s, v):
        if s == "🌑 Assombrissement (Low Light)":
            factor = 1.0 - (v / 100.0)
            return np.clip(factor * chan - (v / 2), 0, 255)
        if s == "☀️ Surexposition (Burnout)":
            factor = 1.0 + (v / 100.0)
            return np.clip(factor * chan + (v / 2), 0, 255)
        if s == "🌫️ Faible Contraste (Brouillard)":
            factor = 1.0 - (v / 100.0)
            return np.clip(128 + factor * (chan - 128), 0, 255)
        return chan

    if mode == "Mode Couleur (RGB)":
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = process(hsv[:, :, 2], scenario, intensity)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return process(img_f, scenario, intensity).astype(np.uint8)

def harmonize(img, method, mode):
    if method == "Aucun": return img
    if mode == "Mode Couleur (RGB)":
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        if method == "Min-Max": v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
        elif method == "GHE": v = cv2.equalizeHist(v)
        elif method == "CLAHE": v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(v)
        hsv[:, :, 2] = v
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        if method == "Min-Max": return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        elif method == "GHE": return cv2.equalizeHist(img)
        elif method == "CLAHE": v = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img); return v
    return img

def extract_signature(img, desc_type, mode):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if mode == "Mode Couleur (RGB)" else img
    
    viz = None
    if desc_type == "HOG":
        feats, viz = hog(gray, orientations=9, pixels_per_cell=(10,10), cells_per_block=(2,2), visualize=True)
        viz = exposure.rescale_intensity(viz, in_range=(0, 10))
    elif desc_type == "LBP":
        viz = local_binary_pattern(gray, P=8, R=2, method='uniform')
        feats, _ = np.histogram(viz.ravel(), bins=np.arange(0, 11), range=(0, 10))
        viz = (viz / (viz.max() if viz.max() > 0 else 1) * 255).astype(np.uint8)
    else: # Histogramme
        feats = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        viz = np.zeros((100, 256), dtype=np.uint8)
        max_f = feats.max() if feats.max() > 0 else 1
        for i, v in enumerate(feats): cv2.line(viz, (i, 100), (i, 100-int(v/max_f*100)), 255, 1)

    norm = np.linalg.norm(feats)
    return (feats / norm if norm > 0 else feats), viz

# =============================================================================
# MAIN INTERFACE
# =============================================================================

def main():
    st.markdown("""
        <div class="brand-header">
            <h1 style="margin:0; color:white; font-weight:900; font-size:2.8rem; letter-spacing:-2px;">Lab Vision Par Ordinateur</h1>
            <p style="margin:0; color:#00d4ff; font-weight:600; letter-spacing:2px; font-size:0.8rem;">Technique d’égalisation et de normalisation pour la robustesse des descripteurs globaux</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 💠 Paramètres d'Entrée")
        work_mode = st.radio("Domaine de l'Image :", ["Mode Niveaux de Gris (8-bits)", "Mode Couleur (RGB)"], index=0)
        up_file = st.file_uploader("Injecter une matrice source", type=['jpg','png','jpeg'])
        
        st.markdown("---")
        st.markdown("### ⚠️ Modèle de Stress")
        scenario = st.selectbox("Scénario d'Éclairage :", [
            "🌑 Assombrissement (Low Light)", 
            "☀️ Surexposition (Burnout)", 
            "🌫️ Faible Contraste (Brouillard)"
        ])
        intensity = st.slider("Magnitude du Stress (%) :", 0, 100, 80)
        
        st.markdown("---")
        st.markdown("### 🧪 Moteur d'Analyse")
        equalizer = st.selectbox("Algorithme de Correction :", ["CLAHE", "GHE", "Min-Max"])
        descriptor = st.selectbox("Descripteur Global :", ["HOG", "LBP", "Histogramme Intensité"])

    if not up_file:
        st.info("💡 **Système en attente** : veuillez charger une image pour initialiser le pipeline.")
        return

    # Image Processing
    raw_bytes = np.frombuffer(up_file.read(), np.uint8)
    img_orig_raw = cv2.cvtColor(cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img_orig_raw.shape[:2]
    img_orig_raw = cv2.resize(img_orig_raw, (int(w*(400/max(h,w))), int(h*(400/max(h,w)))))

    img_ref = convert_to_gray(img_orig_raw) if work_mode == "Mode Niveaux de Gris (8-bits)" else img_orig_raw
    img_deg = apply_scientific_stress(img_ref, scenario, intensity, work_mode)
    
    # Harmonisation des deux flux pour une comparaison équitable
    img_deg_t = harmonize(img_deg, equalizer, work_mode)
    img_ref_t = harmonize(img_ref, equalizer, work_mode)

    tab1, tab2 = st.tabs(["⚙️ PIPELINE EXPÉRIMENTAL", "📊 RÉCAPITULATIF & RÉSULTATS"])

    with tab1:
        # STEP 1
        st.markdown('<div class="sci-unit"><div class="unit-label">ÉTAPE 1 : SIMULATION DU STRESS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.image(img_ref, caption="SCÈNE DE RÉFÉRENCE", use_container_width=True)
        with c2: st.image(img_deg, caption=f"SCÈNE ALTÉRÉE ({scenario})", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # STEP 2
        st.markdown('<div class="sci-unit"><div class="unit-label">ÉTAPE 2 : HARMONISATION ET SIGNATURES</div>', unsafe_allow_html=True)
        st.markdown(f"**Méthodologie :** Nous appliquons **{equalizer}** sur l'image dégradée pour restaurer sa distribution, et sur la référence pour garantir un alignement spectral parfait.")
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.image(img_deg_t, caption=f"SCÈNE DÉGRADÉE (CORRIGÉE VIA {equalizer})", use_container_width=True)
            v_deg_t, viz_deg_t = extract_signature(img_deg_t, descriptor, work_mode)
            st.markdown(f"**Signature Géométrique ({descriptor})**")
            st.image(viz_deg_t, use_container_width=True, clamp=True)
            
        with c_b:
            st.image(img_ref_t, caption="SCÈNE DE RÉFÉRENCE (HARMONISÉE)", use_container_width=True)
            v_ref_t, viz_ref_t = extract_signature(img_ref_t, descriptor, work_mode)
            st.markdown(f"**Signature Géométrique ({descriptor})**")
            st.image(viz_ref_t, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # STEP 3
        st.markdown('<div class="sci-unit"><div class="unit-label">ÉTAPE 3 : MESURES DE ROBUSTESSE</div>', unsafe_allow_html=True)
        v_raw_dark, _ = extract_signature(img_deg, descriptor, work_mode)
        sim_native = np.dot(v_ref_t, v_raw_dark) # Comparaison flux A (Brut) vs Référence harmonisée
        sim_harmonic = np.dot(v_ref_t, v_deg_t) # Comparaison flux B (Corrigé) vs Référence harmonisée
        
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""<div class="score-box"><div class="score-title">Robustesse Native (Flux A)</div><div class="score-val status-bad">{sim_native*100:.2f}%</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="score-box"><div class="score-title">Robustesse Corrigée (Flux B)</div><div class="score-val status-ok">{sim_harmonic*100:.2f}%</div></div>""", unsafe_allow_html=True)
        
        st.markdown("""<p class="calc-note">Source des statistiques : calcul de la Similarité Cosinus entre les vecteurs de caractéristiques normalisés L2. Un score de 100% indique une invariance totale.</p>""", unsafe_allow_html=True)
        
        delta = (sim_harmonic - sim_native) * 100
        if delta > 0:
            st.success(f"**Validation du Théorème :** L'harmonisation génère un gain d'invariance de **+{delta:.2f}%**. Les signatures sont désormais quasi-identiques malgré le stress lumineux.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("📑 Rapport de Benchmarking")
        st.markdown("#### 1. Dimensions des Vecteurs")
        desc_list = ["Histogramme Intensité", "HOG", "LBP"]
        dims = []
        for d in desc_list:
            vt, _ = extract_signature(img_ref, d, work_mode)
            dims.append({"Descripteur": d, "Dimensions": len(vt)})
        st.table(pd.DataFrame(dims))
        
        st.markdown("---")
        st.markdown(f"#### 2. Performance Comparative (Scenario: {scenario})")
        methods = ["Aucun", "Min-Max", "GHE", "CLAHE"]
        full_bench = []
        for d in desc_list:
            row = {"Descripteur": d}
            for m in methods:
                r_t = harmonize(img_ref, m, work_mode)
                d_t = harmonize(img_deg, m, work_mode)
                vr, _ = extract_signature(r_t, d, work_mode)
                vd, _ = extract_signature(d_t, d, work_mode)
                row[m] = round(np.dot(vr, vd) * 100, 2)
            full_bench.append(row)
        
        st.dataframe(pd.DataFrame(full_bench).set_index("Descripteur").style.format("{:.2f}%").highlight_max(axis=1, color="rgba(0, 212, 255, 0.2)"), use_container_width=True)

if __name__ == "__main__":
    main()
