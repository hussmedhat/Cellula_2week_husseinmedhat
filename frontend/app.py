import streamlit as st
import requests
from PIL import Image
import io

# ============================================================
# CONFIG
# ============================================================
API_URL = "http://localhost:8000/predict"
VIOLATION_THRESHOLD = 0.35  # single source of truth for "is this flagged"

LABEL_ORDER = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_DISPLAY = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate",
}
LABEL_ICON = {
    "toxic": "☣️",
    "severe_toxic": "🔥",
    "obscene": "🚫",
    "threat": "⚠️",
    "insult": "💢",
    "identity_hate": "✋",
}

st.set_page_config(page_title="Content Safety Guardrail", page_icon="🛡️", layout="centered")

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ---------- Animated gradient background ---------- */
    .stApp {
        background: #08090d;
        overflow-x: hidden;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        background:
            radial-gradient(circle at 12% 8%, rgba(99,102,241,0.16), transparent 38%),
            radial-gradient(circle at 88% 15%, rgba(236,72,153,0.13), transparent 42%),
            radial-gradient(circle at 25% 92%, rgba(56,189,248,0.10), transparent 40%),
            radial-gradient(circle at 92% 85%, rgba(168,85,247,0.12), transparent 40%);
        animation: drift 18s ease-in-out infinite alternate;
        pointer-events: none;
    }
    @keyframes drift {
        0%   { transform: translate(0px, 0px) scale(1); }
        100% { transform: translate(-20px, 15px) scale(1.05); }
    }

    .block-container { position: relative; z-index: 1; padding-top: 2rem; }

    /* ---------- Hero ---------- */
    .hero { text-align: center; padding: 1.25rem 0 0.75rem 0; animation: fadeSlideDown 0.7s ease; }
    .hero-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(196,181,253,0.9);
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(167,139,250,0.3);
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: linear-gradient(100deg, #a5b4fc 0%, #d8b4fe 35%, #f9a8d4 65%, #93c5fd 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 6s linear infinite;
        margin-bottom: 0.4rem;
    }
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-sub { color: rgba(255,255,255,0.5); font-size: 1rem; font-weight: 400; }

    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Glass panel ---------- */
    .glass-panel {
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 22px;
        padding: 1.6rem;
        margin: 1.1rem 0;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        animation: fadeSlideUp 0.6s ease 0.1s both;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.02);
        border: 1.5px dashed rgba(167,139,250,0.35);
        border-radius: 16px;
        transition: all 0.25s ease;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(167,139,250,0.75);
        background: rgba(167,139,250,0.04);
    }

    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        color: #e5e7eb !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(167,139,250,0.65) !important;
        box-shadow: 0 0 0 4px rgba(167,139,250,0.12) !important;
    }

    .stButton > button {
        background: linear-gradient(100deg, #6366f1, #a855f7, #ec4899);
        background-size: 200% auto;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.01em;
        padding: 0.75rem 1rem;
        transition: all 0.25s ease;
        box-shadow: 0 6px 24px rgba(139,92,246,0.4);
    }
    .stButton > button:hover:not(:disabled) {
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(139,92,246,0.55);
    }
    .stButton > button:active:not(:disabled) { transform: translateY(0); }
    .stButton > button:disabled { opacity: 0.3; box-shadow: none; }

    /* ---------- Verdict card ---------- */
    .verdict-card {
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin: 1.1rem 0 1.4rem 0;
        border: 1px solid rgba(255,255,255,0.09);
        position: relative;
        overflow: hidden;
        animation: fadeSlideUp 0.5s ease;
    }
    .verdict-safe {
        background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(16,185,129,0.03));
        border-color: rgba(52,211,153,0.35);
    }
    .verdict-flagged {
        background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.04));
        border-color: rgba(248,113,113,0.4);
        animation: fadeSlideUp 0.5s ease, pulseGlow 2.4s ease-in-out infinite;
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 rgba(239,68,68,0.0); }
        50% { box-shadow: 0 0 32px rgba(239,68,68,0.22); }
    }
    .verdict-icon { font-size: 1.8rem; margin-bottom: 0.3rem; display: block; }
    .verdict-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.3rem; letter-spacing: -0.01em; }
    .verdict-sub { font-size: 0.94rem; opacity: 0.75; }

    .caption-box {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #a78bfa;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 0.9rem 0 1.2rem 0;
        font-style: italic;
        font-size: 0.95rem;
        line-height: 1.6;
        color: rgba(255,255,255,0.82);
        animation: fadeSlideUp 0.5s ease 0.05s both;
    }

    /* ---------- Score bars ---------- */
    .score-row {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin: 0.7rem 0;
        animation: fadeSlideUp 0.4s ease both;
    }
    .score-icon {
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 9px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .score-label { width: 118px; font-size: 0.87rem; font-weight: 600; color: rgba(255,255,255,0.85); }
    .score-track {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        height: 13px;
        overflow: hidden;
        position: relative;
    }
    .score-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
        position: relative;
    }
    .score-fill::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
        animation: sheen 2.4s ease-in-out infinite;
    }
    @keyframes sheen {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    .score-pct {
        width: 56px;
        text-align: right;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        opacity: 0.9;
    }

    .pill {
        display: inline-block;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0.15rem 0.35rem 0.15rem 0;
        background: rgba(239,68,68,0.18);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.4);
    }

    .signal-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.75);
        margin-top: 0.3rem;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.07);
        padding: 0.4rem 1rem;
        color: rgba(255,255,255,0.6);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(167,139,250,0.15) !important;
        border-color: rgba(167,139,250,0.4) !important;
        color: #e9d5ff !important;
    }

    /* ---------- Misc ---------- */
    #MainMenu, footer { visibility: hidden; }
    .app-footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.78rem;
        padding: 2rem 0 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def bar_gradient(score: float) -> str:
    if score >= VIOLATION_THRESHOLD:
        return "linear-gradient(90deg, #ef4444, #f87171)"
    if score >= VIOLATION_THRESHOLD * 0.5:
        return "linear-gradient(90deg, #f59e0b, #fbbf24)"
    return "linear-gradient(90deg, #22c55e, #4ade80)"


def render_score_bar(label_key: str, score: float, delay: float = 0.0):
    pct = max(0.0, min(100.0, score * 100))
    gradient = bar_gradient(score)
    st.markdown(
        f"""
        <div class="score-row" style="animation-delay:{delay:.2f}s;">
            <div class="score-icon">{LABEL_ICON.get(label_key, "")}</div>
            <div class="score-label">{LABEL_DISPLAY[label_key]}</div>
            <div class="score-track">
                <div class="score-fill" style="width:{pct:.1f}%; background:{gradient};"></div>
            </div>
            <div class="score-pct">{pct:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def call_backend(image_bytes, filename, content_type, caption_text: str):
    """image_bytes may be None (text-only request). caption_text may be
    empty string (image-only request). At least one must be non-empty --
    enforced by the UI before this is ever called."""
    data = {"caption": caption_text} if caption_text else {}

    if image_bytes is not None:
        files = {"file": (filename, image_bytes, content_type)}
        return requests.post(API_URL, files=files, data=data, timeout=60)
    else:
        return requests.post(API_URL, data=data, timeout=60)


# ============================================================
# UI
# ============================================================
st.markdown(
    """
    <div class="hero">
        <span class="hero-badge">✨ Multimodal Toxicity Detection</span>
        <div class="hero-title">Content Safety Guardrail</div>
        <div class="hero-sub">Upload an image, write a caption, or both — get an instant, fused toxicity read.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image (optional)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width='stretch')

user_caption = st.text_area(
    "Or enter a caption / comment to classify (optional)",
    height=90,
    placeholder="Type a comment or caption here...",
    label_visibility="collapsed",
)

has_input = uploaded_file is not None or bool(user_caption.strip())

analyze_clicked = st.button("🔍  Classify", type="primary", width='stretch', disabled=not has_input)
if not has_input:
    st.caption("Upload an image, enter a caption, or both, to enable classification.")

st.markdown('</div>', unsafe_allow_html=True)

if analyze_clicked:
    with st.spinner("Analyzing content..."):
        try:
            img_byte_arr = None
            filename = None
            content_type = None

            if uploaded_file is not None:
                img_buf = io.BytesIO()
                image.save(img_buf, format=image.format if image.format else "PNG")
                img_byte_arr = img_buf.getvalue()
                filename = uploaded_file.name
                content_type = uploaded_file.type

            response = call_backend(img_byte_arr, filename, content_type, user_caption.strip())

        except requests.exceptions.ConnectionError:
            st.error("❌ Can't reach the backend. Is the FastAPI server running on `localhost:8000`?")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The model may be overloaded — try again.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error contacting the server: {e}")
            st.stop()

    if response.status_code != 200:
        st.error(f"Server returned an error ({response.status_code}): {response.text}")
        st.stop()

    data = response.json()
    status = data.get("status", "success")
    description = data.get("description", "")
    fused_scores = data.get("toxicity_scores", {}) or {}
    raw_clip_scores = data.get("raw_clip_scores", {}) or {}
    raw_bert_scores = data.get("raw_bert_scores")  # may be None on partial_failure

    st.divider()

    # --------------------------------------------------
    # Caption / analyzed text
    # --------------------------------------------------
    st.markdown(f'<div class="caption-box">📄 {description}</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # Partial failure banner — caption generation failed upstream
    # --------------------------------------------------
    if status == "partial_failure":
        st.warning(
            "⚠️ Caption generation failed for this image — the verdict below is based on "
            "**visual analysis only** (CLIP), without text-based classification. Treat it as "
            "lower-confidence than a normal result."
        )

    # --------------------------------------------------
    # SINGLE SOURCE OF TRUTH: one flagging decision, everything else displays it
    # --------------------------------------------------
    triggered_categories = [
        cat for cat in LABEL_ORDER
        if fused_scores.get(cat, 0.0) >= VIOLATION_THRESHOLD
    ]
    is_flagged = bool(triggered_categories)

    if is_flagged:
        pills = "".join(f'<span class="pill">{LABEL_ICON.get(c, "")} {LABEL_DISPLAY[c]}</span>' for c in triggered_categories)
        st.markdown(
            f"""
            <div class="verdict-card verdict-flagged">
                <span class="verdict-icon">🚨</span>
                <div class="verdict-title">Flagged for review</div>
                <div class="verdict-sub">This content crossed the safety threshold in the following categories:</div>
                <div style="margin-top:0.7rem;">{pills}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="verdict-card verdict-safe">
                <span class="verdict-icon">✅</span>
                <div class="verdict-title">Approved for distribution</div>
                <div class="verdict-sub">No category crossed the safety threshold.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Supporting evidence line — informational only, does not make its own verdict
    has_clip_signal = raw_clip_scores and any(v is not None for v in raw_clip_scores.values())
    if has_clip_signal:
        top_clip_label = max(raw_clip_scores, key=lambda k: raw_clip_scores[k] or 0.0)
        top_clip_score = raw_clip_scores[top_clip_label]
        st.markdown(
            f"""
            <div class="signal-chip">
                👁️ Strongest visual signal:
                <b>{LABEL_DISPLAY.get(top_clip_label, top_clip_label)}</b>
                ({top_clip_score:.2f} raw CLIP confidence)
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # Score breakdown
    # --------------------------------------------------
    with st.expander("📊 Full probability breakdown", expanded=is_flagged):
        st.markdown("**Fused multi-modal scores**")
        for i, label in enumerate(LABEL_ORDER):
            render_score_bar(label, fused_scores.get(label, 0.0), delay=i * 0.05)

        st.markdown("&nbsp;", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🧠 Raw Text Model (BERT)", "👁️ Raw Visual Model (CLIP)"])
        with tab1:
            if raw_bert_scores:
                st.json(raw_bert_scores)
            else:
                st.info("Not available — caption generation failed for this request.")
        with tab2:
            if has_clip_signal:
                st.json(raw_clip_scores)
            else:
                st.info("Not available — no image was provided for this request.")

st.markdown('<div class="app-footer">Multimodal BERT + CLIP toxicity classifier</div>', unsafe_allow_html=True)