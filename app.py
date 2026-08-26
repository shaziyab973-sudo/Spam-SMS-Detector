import os
import sys

# Disable TensorFlow GPU CUDA search to guarantee instant startup on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(__file__))

from src.inference import SpamClassifier

# Backend Locked Decision Threshold
THRESHOLD = 0.90

# About Menu Text
ABOUT_TEXT = """
Short Message Service (SMS) remains a critical vector for phishing, financial fraud, and unauthorized advertising.

### 🛡️ Why Use Spam SMS Detector?
* **Instant Scam Protection:** Shields you from financial fraud, fake prize notifications, and dangerous phishing links right before you click or reply.
* **Smart Context Recognition:** Understands the natural intent of the message—not just standalone words—giving you accurate safety assessments even against clever text tricks.
* **Zero False Alarms:** Tuned with a precision safety threshold so your genuine bank alerts, appointment reminders, and personal texts are never flagged by mistake.
* **Private & Lightweight:** Runs fast on-device analysis to give you instant peace of mind without compromising your messaging experience.

---

### ✨ Smarter Than Traditional Filters
* **Understands Full Intent:** Unlike standard filters that only look for specific blocked keywords, our deep learning engine reads the full context and tone of the message.
* **Adapts to New Tactics:** Effectively catches modern scam variations, altered spellings, and deceptive urgency cues that bypass basic spam blockers.
* **Built for Trust & Reliability:** Designed from the ground up to keep your inbox clean, your confidential data secure, and your day-to-day communications uninterrupted.
"""

# Page Configuration
st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_classifier() -> SpamClassifier:
    """Loads and caches SpamClassifier instance."""
    return SpamClassifier(models_dir="models")


def main():
    # Initialize Session State Theme Mode safely inside main()
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"

    is_dark = st.session_state.get("theme_mode", "dark") == "dark"

    # Custom Theme CSS (Dynamic Light / Dark Mode with Adaptive Text Colors)
    if is_dark:
        THEME_CSS = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            .stApp {
                background-color: #0f172a;
                color: #f8fafc;
            }
            p, h1, h2, h3, h4, h5, h6, label, span, div, .stMarkdown {
                color: #f8fafc;
            }
            /* Presets & Secondary Buttons in Dark Mode */
            .stButton > button {
                background-color: #1e293b !important;
                color: #f8fafc !important;
                border: 1px solid #334155 !important;
                border-radius: 8px !important;
            }
            .stButton > button p, .stButton > button span, .stButton > button div {
                color: #f8fafc !important;
            }
            .stButton > button:hover {
                background-color: #334155 !important;
                border-color: #475569 !important;
            }
            /* Primary Button (Analyze SMS) in Dark Mode */
            .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
                background-color: #3b82f6 !important;
                color: #ffffff !important;
                border: none !important;
            }
            .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span,
            .stButton > button[data-testid="baseButton-primary"] p, .stButton > button[data-testid="baseButton-primary"] span {
                color: #ffffff !important;
            }
            /* Text Area Input in Dark Mode */
            [data-testid="stTextArea"] textarea {
                background-color: #1e293b !important;
                color: #f8fafc !important;
                border: 1px solid #334155 !important;
            }
            /* Expander Headers in Dark Mode */
            [data-testid="stExpander"] details, [data-testid="stExpander"] summary {
                background-color: #1e293b !important;
                color: #f8fafc !important;
                border: 1px solid #334155 !important;
                border-radius: 8px !important;
            }
            [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
                color: #f8fafc !important;
            }
            .header-box {
                background: linear-gradient(135deg, #1e1e2f 0%, #0f172a 100%);
                padding: 20px 24px;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 20px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            }
            .header-title {
                color: #ffffff !important;
                font-weight: 800;
                font-size: 2.2rem;
                margin: 0;
            }
            .prediction-card {
                background: #1e293b;
                padding: 24px;
                border-radius: 14px;
                border: 1px solid #334155;
                color: #f8fafc !important;
                margin-top: 16px;
            }
            .token-tag-normal {
                background-color: #334155;
                color: #cbd5e1 !important;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.9rem;
                display: inline-block;
                margin: 2px;
            }
            .token-tag-suspicious {
                background-color: #7f1d1d;
                color: #fca5a5 !important;
                border: 1px solid #ef4444;
                padding: 2px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.9rem;
                display: inline-block;
                margin: 2px;
            }
            .status-badge-spam {
                background-color: #ef4444;
                color: #ffffff !important;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 700;
                font-size: 1.1rem;
                display: inline-block;
            }
            .status-badge-ham {
                background-color: #10b981;
                color: #ffffff !important;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 700;
                font-size: 1.1rem;
                display: inline-block;
            }
            .risk-high {
                background: #fee2e2;
                color: #991b1b !important;
                border: 1px solid #f87171;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 700;
            }
            .risk-medium {
                background: #fef3c7;
                color: #92400e !important;
                border: 1px solid #fbbf24;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 700;
            }
            .risk-low {
                background: #d1fae5;
                color: #065f46 !important;
                border: 1px solid #34d399;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 700;
            }
            .custom-footer {
                margin-top: 60px;
                padding: 24px;
                border-top: 1px solid #1e293b;
                text-align: center;
                color: #94a3b8 !important;
            }
            .footer-heading { font-weight: 700; font-size: 1.1rem; color: #f1f5f9 !important; margin-bottom: 6px; }
            .footer-subtext { font-size: 0.95rem; color: #94a3b8 !important; margin-bottom: 8px; }
            .footer-copyright { font-size: 0.85rem; color: #64748b !important; margin-top: 6px; }
            #MainMenu, header, [data-testid="stHeader"], footer, .stDeployButton { display: none !important; }
        </style>
        """
    else:
        THEME_CSS = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            .stApp {
                background-color: #f8fafc;
                color: #0f172a;
            }
            p, h1, h2, h3, h4, h5, h6, label, span, div, .stMarkdown {
                color: #0f172a !important;
            }
            /* Presets & Secondary Buttons in Light Mode */
            .stButton > button {
                background-color: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 8px !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
            }
            .stButton > button p, .stButton > button span, .stButton > button div {
                color: #0f172a !important;
            }
            .stButton > button:hover {
                background-color: #f1f5f9 !important;
                border-color: #94a3b8 !important;
            }
            /* Primary Button (Analyze SMS) in Light Mode */
            .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
                background-color: #2563eb !important;
                color: #ffffff !important;
                border: none !important;
            }
            .stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span,
            .stButton > button[data-testid="baseButton-primary"] p, .stButton > button[data-testid="baseButton-primary"] span {
                color: #ffffff !important;
            }
            /* Text Area Input in Light Mode */
            [data-testid="stTextArea"] textarea {
                background-color: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
            }
            /* Expander Headers in Light Mode */
            [data-testid="stExpander"] details, [data-testid="stExpander"] summary {
                background-color: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 8px !important;
            }
            [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
                color: #0f172a !important;
            }
            .header-box {
                background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
                padding: 20px 24px;
                border-radius: 16px;
                border: 1px solid #cbd5e1;
                margin-bottom: 20px;
                box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
            }
            .header-title {
                color: #0f172a !important;
                font-weight: 800;
                font-size: 2.2rem;
                margin: 0;
            }
            .prediction-card {
                background: #ffffff;
                padding: 24px;
                border-radius: 14px;
                border: 1px solid #cbd5e1;
                color: #0f172a !important;
                margin-top: 16px;
            }
            .token-tag-normal {
                background-color: #e2e8f0;
                color: #1e293b !important;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.9rem;
                display: inline-block;
                margin: 2px;
            }
            .token-tag-suspicious {
                background-color: #fee2e2;
                color: #991b1b !important;
                border: 1px solid #ef4444;
                padding: 2px 8px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 0.9rem;
                display: inline-block;
                margin: 2px;
            }
            .status-badge-spam {
                background-color: #ef4444;
                color: #ffffff !important;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 700;
                font-size: 1.1rem;
                display: inline-block;
            }
            .status-badge-ham {
                background-color: #10b981;
                color: #ffffff !important;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 700;
                font-size: 1.1rem;
                display: inline-block;
            }
            .risk-high {
                background: #fee2e2;
                color: #991b1b !important;
                border: 1px solid #f87171;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 700;
            }
            .risk-medium {
                background: #fef3c7;
                color: #92400e !important;
                border: 1px solid #fbbf24;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 700;
            }
            .risk-low {
                background: #d1fae5;
                color: #065f46 !important;
                border: 1px solid #34d399;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 700;
            }
            .custom-footer {
                margin-top: 60px;
                padding: 24px;
                border-top: 1px solid #cbd5e1;
                text-align: center;
                color: #475569 !important;
            }
            .footer-heading { font-weight: 700; font-size: 1.1rem; color: #0f172a !important; margin-bottom: 6px; }
            .footer-subtext { font-size: 0.95rem; color: #475569 !important; margin-bottom: 8px; }
            .footer-copyright { font-size: 0.85rem; color: #64748b !important; margin-top: 6px; }
            #MainMenu, header, [data-testid="stHeader"], footer, .stDeployButton { display: none !important; }
        </style>
        """

    st.markdown(THEME_CSS, unsafe_allow_html=True)

    classifier = load_classifier()

    # Top Header Controls Bar (Title Left, Settings & About Top-Right)
    head_col1, head_col2, head_col3 = st.columns([5, 2.5, 2.5])

    with head_col1:
        st.markdown('<div class="header-box"><div class="header-title">🛡️ Spam SMS Detector</div></div>', unsafe_allow_html=True)

    with head_col2:
        with st.expander("⚙️ Settings"):
            st.subheader("⚙️ Settings")
            st.markdown("Customize your application interface appearance:")
            theme_choice = st.radio(
                "Select Appearance:",
                options=["🌙 Dark Mode", "☀️ Light Mode"],
                index=0 if st.session_state["theme_mode"] == "dark" else 1,
                key="theme_radio_selector"
            )
            new_mode = "light" if "Light" in theme_choice else "dark"
            if new_mode != st.session_state["theme_mode"]:
                st.session_state["theme_mode"] = new_mode
                if hasattr(st, "rerun"):
                    st.rerun()
                elif hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()

    with head_col3:
        with st.expander("ℹ️ About"):
            st.markdown("### 🛡️ About Spam SMS Detector")
            st.markdown(ABOUT_TEXT)

    # Status Banner if Model is missing
    if not classifier.is_loaded:
        st.warning("⚠️ Trained model artifacts not found! Run training pipeline using `python run_pipeline.py` or training script.")
        if st.button("🚀 Train Model Now", type="primary"):
            with st.spinner("Training Bi-LSTM model on SMS dataset..."):
                from run_pipeline import run_full_pipeline
                run_full_pipeline()
                st.cache_resource.clear()
                if hasattr(st, "rerun"):
                    st.rerun()
                elif hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()

    # Single SMS Classifier Section
    st.subheader("Analyze Text Message")

    # Quick Test Preset Buttons
    st.markdown("**Quick Test Sample Presets:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    preset_text = ""
    if col_p1.button("🚨 Cash Prize Spam"):
        preset_text = "WINNER! You have won a $1000 Walmart gift card! Call 1-800-555-0199 now to claim!"
    if col_p2.button("🚨 Urgent Bank Spam"):
        preset_text = "URGENT! Your bank account has been compromised. Verify your details at http://bit.ly/fake-bank immediately."
    if col_p3.button("💬 Meeting Legit"):
        preset_text = "Hey, are we still meeting for lunch at 12:30 today?"
    if col_p4.button("💬 Homework Legit"):
        preset_text = "Can you please send me the project report when you get a chance?"

    # Text Area Input
    user_sms = st.text_area(
        "Enter SMS text message below:",
        value=preset_text,
        height=130,
        placeholder="Type or paste SMS message here..."
    )

    col_act1, col_act2 = st.columns([1, 4])
    analyze_click = col_act1.button("🔍 Analyze SMS", type="primary", use_container_width=True)

    if analyze_click or (preset_text and user_sms == preset_text):
        if not user_sms.strip():
            st.error("Please enter a valid text message before clicking Analyze.")
        else:
            with st.spinner("Processing message through Bi-LSTM network..."):
                result = classifier.predict(user_sms, threshold=THRESHOLD)

            if result.get("error"):
                st.error(f"Validation / Inference Error: {result['error']}")
            else:
                is_spam = result["label"] == "Spam"
                raw_prob = result["raw_probability"]
                conf = result["confidence"]
                risk = result["risk_level"]

                st.markdown("---")
                
                # Result Display Card
                res_col1, res_col2, res_col3 = st.columns([2, 2, 2])

                with res_col1:
                    st.markdown("**Classification Decision**")
                    if is_spam:
                        st.markdown('<div class="status-badge-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-badge-ham">✅ LEGIT / HAM</div>', unsafe_allow_html=True)

                with res_col2:
                    st.markdown("**Risk Level**")
                    if risk == "High":
                        st.markdown('<span class="risk-high">HIGH RISK (🚨)</span>', unsafe_allow_html=True)
                    elif risk == "Medium":
                        st.markdown('<span class="risk-medium">MEDIUM RISK (⚠️)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="risk-low">LOW RISK (🟢)</span>', unsafe_allow_html=True)

                with res_col3:
                    st.markdown("**Model Confidence**")
                    st.markdown(f"### `{conf}`")

                # Confidence Progress Gauge
                st.markdown("**Spam Probability Score**")
                st.progress(float(raw_prob))
                st.caption(f"Raw Sigmoid Output: `{raw_prob:.4f}` | Decision Threshold: `{THRESHOLD}`")

                # Token Breakdown / Highlighted Words
                st.markdown("#### 🔬 Token Analysis & Trigger Words")
                tokens = result.get("token_highlights", [])
                if tokens:
                    html_tokens = []
                    suspicious_count = 0
                    for t in tokens:
                        if t["is_suspicious"]:
                            html_tokens.append(f'<span class="token-tag-suspicious">⚠️ {t["token"]}</span>')
                            suspicious_count += 1
                        else:
                            html_tokens.append(f'<span class="token-tag-normal">{t["token"]}</span>')

                    st.markdown(" ".join(html_tokens), unsafe_allow_html=True)
                    if suspicious_count > 0:
                        st.caption(f"Found {suspicious_count} potential spam trigger keyword(s).")
                    else:
                        st.caption("No high-risk spam keywords detected.")

    # Custom Footer Section
    st.markdown("""
    <div class="custom-footer">
        <div class="footer-heading">Thank you for using Spam SMS Detector!</div>
        <div class="footer-subtext">An AI-powered solution for detecting spam messages using Deep Learning & NLP.</div>
        <div class="footer-copyright">© 2026 Spam SMS Detector</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
