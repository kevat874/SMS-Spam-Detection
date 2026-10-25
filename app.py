"""
SMS Spam Detection – Streamlit Web App
Loads the pre-trained pipeline and classifies new messages.
"""

import streamlit as st
from pathlib import Path
import sys

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import predict_message, load_model

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
)

st.title("📱 SMS Spam Detector")
st.markdown(
    """
    Enter an SMS message below and the model will classify it as **Ham** (legitimate)
    or **Spam**.  
    *This is a classical ML model (TF-IDF + LinearSVC / Logistic Regression) trained on the UCI SMS Spam Collection.*
    """
)

# -------------------------------------------------
# Load model once
# -------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
    st.success("Model loaded successfully.", icon="✅")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# -------------------------------------------------
# Input
# -------------------------------------------------
message = st.text_area(
    "SMS Message",
    height=120,
    placeholder="Type or paste an SMS message here…",
    help="The model works best on English SMS-style text.",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.rerun()

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if predict_btn:
    if not message or not message.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            label, score = predict_message(message)

            if label == "spam":
                st.error(f"**Prediction: SPAM**", icon="🚨")
            else:
                st.success(f"**Prediction: HAM** (legitimate)", icon="✅")

            st.caption(
                f"Decision score: `{score:+.3f}`  \n"
                "(Positive values lean toward spam, negative toward ham. "
                "This is **not** a calibrated probability.)"
            )

            with st.expander("See the raw message"):
                st.code(message, language=None)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# Footer / disclaimer
# -------------------------------------------------
st.divider()
st.markdown(
    """
    **Disclaimer**  
    This model was trained on a public dataset collected around 2011.  
    It may miss modern phishing / scam patterns and can produce false positives.  
    Do **not** rely on it as the sole filter for important messages.
    """
)

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:0.85em">
    Built with Scikit-learn + Streamlit · UCI SMS Spam Collection
    </div>
    """,
    unsafe_allow_html=True,
)
