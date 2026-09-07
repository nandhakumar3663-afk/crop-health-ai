import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Crop Health AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #f3faf5;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ================= HERO ================= */

.hero {
    background: linear-gradient(135deg, #087f3d, #16a34a);
    padding: 42px 30px;
    border-radius: 24px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(8, 127, 61, 0.18);
}

.hero h1 {
    font-size: 44px;
    font-weight: 800;
    margin: 0;
    color: white;
}

.hero p {
    font-size: 18px;
    margin-top: 12px;
    color: #ecfdf5;
}


/* ================= SECTION ================= */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #166534;
    margin-top: 28px;
    margin-bottom: 18px;
}


/* ================= CARD ================= */

.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #d7eedf;
    box-shadow: 0 6px 20px rgba(0, 80, 30, 0.07);
    margin-bottom: 20px;
}

.card-text {
    color: #475569;
    font-size: 16px;
    line-height: 1.7;
    margin: 0;
}


/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #22a653;
    border-radius: 18px;
    padding: 14px;
}


/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #15803d, #22a652);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #116530, #168c44);
    color: white;
}


/* ================= IMAGE ================= */

div[data-testid="stImage"] {
    border-radius: 18px;
    overflow: hidden;
}


/* ================= PREVIEW ================= */

.preview-box {
    background: white;
    min-height: 320px;
    border-radius: 18px;
    border: 1px solid #d7eedf;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0, 80, 30, 0.07);
    padding: 20px;
}

.preview-icon {
    font-size: 60px;
}

.preview-title {
    font-size: 23px;
    font-weight: 800;
    color: #166534;
}

.preview-text {
    color: #64748b;
    font-size: 14px;
    margin-top: 8px;
}


/* ================= RESULT ================= */

.prediction-card {
    background: linear-gradient(135deg, #e9f9ee, #ffffff);
    border-left: 6px solid #16a34a;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0, 80, 30, 0.07);
}

.prediction-label {
    color: #15803d;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 1px;
}

.prediction-name {
    color: #123d25;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
    line-height: 1.3;
}


/* ================= STATUS ================= */

.healthy {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    padding: 15px 18px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 800;
}

.diseased {
    background: #fff1f2;
    color: #be123c;
    border: 1px solid #fecdd3;
    padding: 15px 18px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 800;
}


/* ================= CONFIDENCE ================= */

.confidence-box {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #d7eedf;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0, 80, 30, 0.07);
}

.confidence-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 700;
}

.confidence-number {
    color: #15803d;
    font-size: 38px;
    font-weight: 800;
    margin-top: 5px;
}


/* ================= TOP PREDICTION ================= */

.top-prediction {
    background: white;
    padding: 15px 18px;
    border-radius: 12px;
    border: 1px solid #d7eedf;
    margin-top: 10px;
    box-shadow: 0 3px 12px rgba(0, 80, 30, 0.04);
}

.top-name {
    color: #166534;
    font-size: 16px;
    font-weight: 700;
}

.top-score {
    color: #15803d;
    font-size: 16px;
    font-weight: 800;
    float: right;
}


/* ================= MODEL INFO ================= */

.info-card {
    background: white;
    min-height: 155px;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #d7eedf;
    text-align: center;
    box-shadow: 0 5px 18px rgba(0, 80, 30, 0.06);
}

.info-icon {
    font-size: 34px;
}

.info-title {
    color: #166534;
    font-size: 17px;
    font-weight: 800;
    margin-top: 8px;
}

.info-text {
    color: #64748b;
    font-size: 13px;
    margin-top: 6px;
    line-height: 1.5;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 45px;
    padding-top: 22px;
    border-top: 1px solid #d7eedf;
}

.footer-title {
    color: #166534;
    font-weight: 800;
    font-size: 17px;
    margin-bottom: 6px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_PATH = "scripts/models/plantvillage_efficientnet_b0_fast_best.pth"

IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    class_names = checkpoint["class_names"]

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, class_names


model, class_names = load_model()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero"><h1>🌱 Crop Health AI</h1><p>AI-powered plant disease detection using EfficientNet-B0</p></div>
""", unsafe_allow_html=True)


# =========================================================
# INTRO
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Analyze Your Crop</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card"><p class="card-text">Upload a clear image of a plant leaf and our AI model will analyze it to identify the most likely disease or determine whether the leaf is healthy.</p></div>
""", unsafe_allow_html=True)


# =========================================================
# UPLOAD + PREVIEW
# =========================================================

upload_col, preview_col = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# UPLOAD
# =========================================================

with upload_col:

    st.markdown(
        '<div class="section-title">📤 Upload Leaf Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        label_visibility="collapsed"
    )

    if uploaded_file:

        st.success(
            "✅ Image uploaded successfully!"
        )

    st.write("")

    analyze = st.button(
        "🔬 Analyze Crop"
    )


# =========================================================
# PREVIEW
# =========================================================

with preview_col:

    st.markdown(
        '<div class="section-title">🖼️ Image Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

    else:

        st.markdown("""
<div class="preview-box"><div><div class="preview-icon">🌿</div><div class="preview-title">No image selected</div><div class="preview-text">Upload a leaf image to begin analysis</div></div></div>
""", unsafe_allow_html=True)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file and analyze:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 AI Analysis Result</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "🧠 AI is analyzing the leaf..."
    ):

        image_tensor = transform(
            image
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = F.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = class_names[
            predicted.item()
        ]

        confidence_value = (
            confidence.item() * 100
        )

        top_probs, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )


    # =====================================================
    # HEALTH STATUS
    # =====================================================

    is_healthy = (
        "healthy"
        in predicted_class.lower()
    )


    # =====================================================
    # RESULT
    # =====================================================

    result_col, confidence_col = st.columns(
        [2, 1],
        gap="large"
    )


    # =====================================================
    # PREDICTION CARD
    # =====================================================

    with result_col:

        st.markdown(f"""
<div class="prediction-card"><div class="prediction-label">AI PREDICTION</div><div class="prediction-name">🌿 {predicted_class}</div></div>
""", unsafe_allow_html=True)

        st.write("")

        if is_healthy:

            st.markdown("""
<div class="healthy">✅ Disease Status: HEALTHY</div>
""", unsafe_allow_html=True)

        else:

            st.markdown("""
<div class="diseased">🚨 Disease Status: DISEASED</div>
""", unsafe_allow_html=True)


    # =====================================================
    # CONFIDENCE CARD
    # =====================================================

    with confidence_col:

        st.markdown(f"""
<div class="confidence-box"><div class="confidence-label">MODEL CONFIDENCE</div><div class="confidence-number">{confidence_value:.2f}%</div></div>
""", unsafe_allow_html=True)

        st.write("")

        st.progress(
            min(confidence_value / 100, 1.0)
        )


    # =====================================================
    # TOP 3
    # =====================================================

    st.markdown(
        '<div class="section-title">🏆 Top Predictions</div>',
        unsafe_allow_html=True
    )

    for rank in range(3):

        probability = (
            top_probs[0][rank].item()
            * 100
        )

        class_name = class_names[
            top_indices[0][rank].item()
        ]

        st.markdown(f"""
<div class="top-prediction"><span class="top-name">{rank + 1}. {class_name}</span><span class="top-score">{probability:.2f}%</span></div>
""", unsafe_allow_html=True)

        st.progress(
            min(probability / 100, 1.0)
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)

info1, info2, info3, info4 = st.columns(
    4,
    gap="medium"
)


with info1:

    st.markdown("""
<div class="info-card"><div class="info-icon">🧠</div><div class="info-title">EfficientNet-B0</div><div class="info-text">Deep learning image classification architecture</div></div>
""", unsafe_allow_html=True)


with info2:

    st.markdown("""
<div class="info-card"><div class="info-icon">🌱</div><div class="info-title">PlantVillage</div><div class="info-text">Plant leaf disease image dataset</div></div>
""", unsafe_allow_html=True)


with info3:

    st.markdown("""
<div class="info-card"><div class="info-icon">🔬</div><div class="info-title">38 Classes</div><div class="info-text">Disease and healthy plant categories</div></div>
""", unsafe_allow_html=True)


with info4:

    st.markdown("""
<div class="info-card"><div class="info-icon">⚡</div><div class="info-title">Fast Analysis</div><div class="info-text">AI-based image prediction in seconds</div></div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer"><div class="footer-title">🌱 Crop Health AI</div>Intelligent plant disease detection powered by deep learning<br><br>EfficientNet-B0 • PlantVillage • 38 Classes</div>
""", unsafe_allow_html=True)