"""
RetinaScope — Diabetic Retinopathy Detection
Streamlit Backend App
Model: MobileNetV2 fine-tuned on APTOS 2019 (dr_model.keras)
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import time
import base64

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaScope · DR Detection",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CLASS_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
CLASS_FULL   = [
    "No Diabetic Retinopathy",
    "Mild Non-Proliferative DR",
    "Moderate Non-Proliferative DR",
    "Severe Non-Proliferative DR",
    "Proliferative Diabetic Retinopathy",
]
GRADE_COLORS = ["#00e5a0", "#f5c518", "#ff7c2a", "#ff5533", "#ff3d5a"]
CLINICAL_NOTES = [
    "No signs of diabetic retinopathy detected. Retinal vasculature and optic disc appear within normal limits. Continue routine annual screening.",
    "Microaneurysms present. No vision-threatening features at this stage; closer monitoring is warranted to detect early progression.",
    "Dot/blot hemorrhages, hard exudates, or cotton wool spots present. Risk of progression to vision-threatening disease is elevated.",
    "Extensive intraretinal hemorrhages or venous beading present. High risk of progression to proliferative DR within 1 year.",
    "Neovascularisation detected on disc or elsewhere. Vitreous or pre-retinal haemorrhage may be present. Immediate intervention critical.",
]
RECOMMENDATIONS = [
    ["📅 Schedule next eye exam in 12 months", "🩸 Maintain HbA1c below 7%", "✅ Continue current diabetes management plan"],
    ["📅 Follow-up examination in 6–9 months", "💊 Optimise blood glucose & blood pressure", "🔬 Consider fluorescein angiography if uncertain"],
    ["⚡ Urgent referral to a retinal specialist", "📸 Fundus fluorescein angiography recommended", "💉 Evaluate for anti-VEGF therapy"],
    ["🚨 Immediate retinal specialist referral required", "💉 Pan-retinal photocoagulation evaluation indicated", "🏥 Monthly follow-up with dilated exam"],
    ["🚨 Emergency ophthalmology consultation required", "💉 Intravitreal anti-VEGF injection and/or PRP laser", "🔪 Vitrectomy may be required in advanced cases"],
]
INPUT_SIZE = (224, 224)
MODEL_PATH = "dr_model.keras"


# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@300;400;500;600&display=swap');

/* ── GLOBAL ── */
html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif !important;
    background-color: #050a0f !important;
    color: #e8f4f8 !important;
}

.stApp {
    background: #050a0f;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── HEADER BRAND ── */
.rs-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 24px 0 20px;
    border-bottom: 1px solid rgba(0,180,220,0.12);
    margin-bottom: 40px;
}
.rs-logo-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #e8f4f8;
    letter-spacing: -0.02em;
}
.rs-logo-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #00c8f0;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.rs-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.25);
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #00e5a0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.rs-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00e5a0;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.5; }
}

/* ── CARDS ── */
.rs-card {
    background: #0b1520;
    border: 1px solid rgba(0,180,220,0.12);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.rs-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00c8f0, transparent);
    opacity: 0.5;
}
.rs-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #3d6070;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── RESULT CARD ── */
.rs-result-card {
    background: #0b1520;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid rgba(0,180,220,0.12);
    margin-bottom: 20px;
}
.rs-grade-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
    border: 1px solid currentColor;
}
.rs-diagnosis-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}
.rs-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #7a9bb0;
}
.rs-clinical {
    font-size: 0.875rem;
    color: #7a9bb0;
    line-height: 1.65;
    padding: 14px 16px;
    background: #0f1e2e;
    border-radius: 0 8px 8px 8px;
    margin-top: 14px;
    border-left: 3px solid currentColor;
}

/* ── PROBABILITY BAR ── */
.rs-prob-row {
    display: grid;
    grid-template-columns: 130px 1fr 52px;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.rs-prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #7a9bb0;
}
.rs-prob-track {
    height: 7px;
    background: #0f1e2e;
    border-radius: 4px;
    overflow: hidden;
}
.rs-prob-fill {
    height: 100%;
    border-radius: 4px;
}
.rs-prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #7a9bb0;
    text-align: right;
}

/* ── REC ITEM ── */
.rs-rec {
    background: #0f1e2e;
    border: 1px solid rgba(0,180,220,0.1);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #7a9bb0;
    margin-bottom: 8px;
}

/* ── METRIC ROW ── */
.rs-metric {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: #0f1e2e;
    border-radius: 8px;
    margin-bottom: 8px;
}
.rs-metric-k { font-size: 0.8rem; color: #7a9bb0; }
.rs-metric-v {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #e8f4f8;
}

/* ── DISCLAIMER ── */
.rs-disclaimer {
    background: rgba(245,197,24,0.04);
    border: 1px solid rgba(245,197,24,0.18);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.78rem;
    color: #7a9bb0;
    line-height: 1.6;
    margin-top: 24px;
}
.rs-disclaimer strong { color: #f5c518; }

/* ── UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #0b1520 !important;
    border: 1px dashed rgba(0,180,220,0.3) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label {
    color: #7a9bb0 !important;
    font-family: 'Instrument Sans', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, rgba(0,200,240,0.15), rgba(0,150,200,0.1)) !important;
    border: 1px solid #00c8f0 !important;
    color: #00c8f0 !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 24px rgba(0,200,240,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── DIVIDERS ── */
hr {
    border-color: rgba(0,180,220,0.1) !important;
    margin: 24px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADER ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained Keras model. Handles cross-version TF compatibility."""
    def focal_loss_fn(gamma=2., alpha=0.25):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * tf.pow(1 - y_pred, gamma)
            return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
        return loss
    custom_objects = {"focal_loss": focal_loss_fn, "loss": focal_loss_fn()}
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, None
    except Exception as e1:
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH, compile=False, custom_objects=custom_objects
            )
            return model, None
        except Exception as e2:
            return None, f"Could not load model.\nError 1: {e1}\nError 2: {e2}"


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize, normalise, and expand dims for model input."""
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, INPUT_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)


def run_inference(model, img_array: np.ndarray):
    """Run model prediction and return probabilities."""
    preds = model.predict(img_array, verbose=0)
    return preds[0]  # shape (5,)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rs-header">
  <div>
    <div class="rs-logo-name">👁 RetinaScope</div>
    <div class="rs-logo-sub">Diabetic Retinopathy Detection System</div>
  </div>
  <div class="rs-status">
    <span class="rs-dot"></span>
    Model Active
  </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
with st.spinner("Loading MobileNetV2 model…"):
    model, model_err = load_model()

if model_err:
    st.error(f"⚠️ Could not load model: `{model_err}`\n\nEnsure `dr_model.keras` is in the same directory as `app.py`.")
    st.stop()

# ── LAYOUT ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="margin-bottom:8px">
      <span style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#00c8f0;letter-spacing:0.2em;text-transform:uppercase;">
        ─── AI-Powered Ophthalmology
      </span>
    </div>
    <div style="font-family:'DM Serif Display',serif;font-size:2.6rem;line-height:1.05;letter-spacing:-0.03em;margin-bottom:16px">
      Detect <em style="color:#00c8f0">Retinal</em><br>Pathology<br>Instantly
    </div>
    <div style="font-size:0.92rem;color:#7a9bb0;line-height:1.7;margin-bottom:28px;max-width:400px">
      Upload a fundus photograph to receive an automated severity assessment 
      for Diabetic Retinopathy. Powered by MobileNetV2 trained on the APTOS 2019 
      dataset with focal loss optimisation.
    </div>
    """, unsafe_allow_html=True)

    # Severity legend
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;color:#3d6070;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:10px">Severity Scale</div>', unsafe_allow_html=True)
    legend_cols = st.columns(5)
    for i, (label, color) in enumerate(zip(CLASS_LABELS, GRADE_COLORS)):
        with legend_cols[i]:
            st.markdown(f'<div style="display:flex;align-items:center;gap:6px;padding:5px 8px;background:#0b1520;border:1px solid rgba(0,180,220,0.12);border-radius:6px;font-family:\'DM Mono\',monospace;font-size:0.58rem;color:#7a9bb0"><div style="width:7px;height:7px;border-radius:50%;background:{color};flex-shrink:0"></div>{label}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model info card
    st.markdown("""
    <div class="rs-card">
      <div class="rs-card-title">Model Information</div>
    """, unsafe_allow_html=True)
    for k, v in [("Architecture", "MobileNetV2"), ("Dataset", "APTOS 2019"), ("Input Size", "224 × 224 px"),
                 ("Loss Function", "Focal Loss (γ=2, α=0.25)"), ("Optimizer", "Adam (lr=1e-4)"), ("Classes", "5 (Grade 0–4)")]:
        st.markdown(f'<div class="rs-metric"><span class="rs-metric-k">{k}</span><span class="rs-metric-v">{v}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="rs-card">', unsafe_allow_html=True)
    st.markdown('<div class="rs-card-title">Image Input</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a retinal fundus image",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        img_array = preprocess_image(pil_img)

        st.markdown("<br>", unsafe_allow_html=True)
        img_col, meta_col = st.columns([1, 1])

        with img_col:
            st.image(pil_img, use_container_width=True, caption="Fundus Input")

        with meta_col:
            file_kb = uploaded.size / 1024
            st.markdown(f"""
            <div class="rs-metric"><span class="rs-metric-k">Filename</span><span class="rs-metric-v">{uploaded.name[:18]}…</span></div>
            <div class="rs-metric"><span class="rs-metric-k">Size</span><span class="rs-metric-v">{file_kb:.1f} KB</span></div>
            <div class="rs-metric"><span class="rs-metric-k">Format</span><span class="rs-metric-v">{uploaded.type.split("/")[1].upper()}</span></div>
            <div class="rs-metric"><span class="rs-metric-k">Dimensions</span><span class="rs-metric-v">{pil_img.width} × {pil_img.height}</span></div>
            <div class="rs-metric"><span class="rs-metric-k">Status</span><span class="rs-metric-v" style="color:#00e5a0">✓ Ready</span></div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyse = st.button("🔬  Analyse Fundus Image", use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#3d6070">
          <div style="font-size:2.5rem;margin-bottom:12px">👁</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.1em;color:#3d6070">
            AWAITING FUNDUS IMAGE
          </div>
        </div>
        """, unsafe_allow_html=True)
        analyse = False

    st.markdown("</div>", unsafe_allow_html=True)

# ── INFERENCE & RESULTS ───────────────────────────────────────────────────────
if uploaded and analyse:
    st.markdown("---")
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#3d6070;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:24px">── Diagnostic Report</div>', unsafe_allow_html=True)

    with st.spinner("Analysing retinal image…"):
        t0 = time.time()
        probs = run_inference(model, img_array)
        elapsed = time.time() - t0

    grade_idx = int(np.argmax(probs))
    conf = float(probs[grade_idx])
    color = GRADE_COLORS[grade_idx]

    res_col, side_col = st.columns([3, 2], gap="large")

    with res_col:
        # ── MAIN DIAGNOSIS ──
        st.markdown(f"""
        <div class="rs-result-card" style="border-color:{color}22">
          <div class="rs-grade-badge" style="color:{color};background:{color}18;border-color:{color}55">
            Grade {grade_idx} · {CLASS_LABELS[grade_idx]}
          </div>
          <div class="rs-diagnosis-name" style="color:{color}">{CLASS_FULL[grade_idx]}</div>
          <div class="rs-conf">Confidence: <span style="color:#e8f4f8;font-weight:600">{conf*100:.1f}%</span></div>
          <div class="rs-clinical" style="border-color:{color}">{CLINICAL_NOTES[grade_idx]}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── PROBABILITY BARS ──
        st.markdown('<div class="rs-card">', unsafe_allow_html=True)
        st.markdown('<div class="rs-card-title">Class Probability Distribution</div>', unsafe_allow_html=True)
        for i, (label, p) in enumerate(zip(CLASS_LABELS, probs)):
            bar_w = int(p * 100)
            c = GRADE_COLORS[i]
            st.markdown(f"""
            <div class="rs-prob-row">
              <span class="rs-prob-label">{label}</span>
              <div class="rs-prob-track">
                <div class="rs-prob-fill" style="width:{bar_w}%;background:{c}"></div>
              </div>
              <span class="rs-prob-pct">{p*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with side_col:
        # ── RECOMMENDATIONS ──
        st.markdown('<div class="rs-card">', unsafe_allow_html=True)
        st.markdown('<div class="rs-card-title">Clinical Recommendations</div>', unsafe_allow_html=True)
        for rec in RECOMMENDATIONS[grade_idx]:
            st.markdown(f'<div class="rs-rec">{rec}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── ANALYSIS METRICS ──
        st.markdown('<div class="rs-card">', unsafe_allow_html=True)
        st.markdown('<div class="rs-card-title">Analysis Metrics</div>', unsafe_allow_html=True)
        for k, v in [("Inference Time", f"{elapsed*1000:.0f} ms"),
                     ("Input Resolution", "224 × 224"),
                     ("Output Classes", "5"),
                     ("Top-1 Grade", f"Grade {grade_idx}"),
                     ("Top-1 Conf.", f"{conf*100:.1f}%")]:
            st.markdown(f'<div class="rs-metric"><span class="rs-metric-k">{k}</span><span class="rs-metric-v">{v}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── DISCLAIMER ──
    st.markdown("""
    <div class="rs-disclaimer">
      <strong>⚠️ Clinical Disclaimer:</strong> This tool is for research and screening assistance only.
      Results must not replace professional ophthalmological evaluation. All findings should be reviewed
      and confirmed by a qualified medical practitioner before any clinical decisions are made.
    </div>
    """, unsafe_allow_html=True)
