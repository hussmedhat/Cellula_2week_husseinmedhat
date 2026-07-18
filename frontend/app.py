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

st.set_page_config(page_title="Content Safety Guardrail", page_icon="🛡️", layout="centered")

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    .verdict-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0 1.25rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .verdict-safe {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
        border-color: rgba(34,197,94,0.35);
    }
    .verdict-flagged {
        background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(239,68,68,0.05));
        border-color: rgba(239,68,68,0.4);
    }
    .verdict-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.25rem; }
    .verdict-sub { font-size: 0.9rem; opacity: 0.85; }

    .caption-box {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
        font-style: italic;
        line-height: 1.5;
    }

    .score-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0; }
    .score-label { width: 130px; font-size: 0.88rem; font-weight: 600; }
    .score-pct { width: 52px; text-align: right; font-size: 0.85rem; font-variant-numeric: tabular-nums; opacity: 0.9; }

    .pill {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.15rem 0.3rem 0.15rem 0;
    }
    .pill-flag { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def bar_color(score: float) -> str:
    if score >= VIOLATION_THRESHOLD:
        return "#ef4444"  # red
    if score >= VIOLATION_THRESHOLD * 0.5:
        return "#f59e0b"  # amber, approaching threshold
    return "#22c55e"  # green


def render_score_bar(label_key: str, score: float):
    pct = score * 100
    color = bar_color(score)
    st.markdown(
        f"""
        <div class="score-row">
            <div class="score-label">{LABEL_DISPLAY[label_key]}</div>
            <div style="flex:1; background: rgba(255,255,255,0.08); border-radius: 6px; height: 10px; overflow: hidden;">
                <div style="width:{pct:.1f}%; background:{color}; height:100%; border-radius:6px;"></div>
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
st.title("🛡️ Content Safety Guardrail")
st.caption("Upload an image, enter a caption/comment, or both, to analyze for potential policy violations.")

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

analyze_clicked = st.button("🔍 Classify", type="primary", width='stretch', disabled=not has_input)
if not has_input:
    st.caption("Upload an image, enter a caption, or both, to enable classification.")

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
        pills = "".join(f'<span class="pill pill-flag">{LABEL_DISPLAY[c]}</span>' for c in triggered_categories)
        st.markdown(
            f"""
            <div class="verdict-card verdict-flagged">
                <div class="verdict-title">🚨 Flagged for review</div>
                <div class="verdict-sub">This content crossed the safety threshold in the following categories:</div>
                <div style="margin-top:0.5rem;">{pills}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="verdict-card verdict-safe">
                <div class="verdict-title">✅ Approved for distribution</div>
                <div class="verdict-sub">No category crossed the safety threshold.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    has_clip_signal = raw_clip_scores and any(v is not None for v in raw_clip_scores.values())
    if has_clip_signal:
        top_clip_label = max(raw_clip_scores, key=lambda k: raw_clip_scores[k] or 0.0)
        top_clip_score = raw_clip_scores[top_clip_label]
        st.caption(
            f"Strongest visual signal: **{LABEL_DISPLAY.get(top_clip_label, top_clip_label)}** "
            f"({top_clip_score:.2f} raw CLIP confidence)"
        )

    # --------------------------------------------------
    # Score breakdown
    # --------------------------------------------------
    with st.expander("📊 Full probability breakdown", expanded=is_flagged):
        st.markdown("**Fused multi-modal scores**")
        for label in LABEL_ORDER:
            render_score_bar(label, fused_scores.get(label, 0.0))

        st.markdown("&nbsp;", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Raw Text Model (BERT)", "Raw Visual Model (CLIP)"])
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
else:
    st.info("👆 Upload an image, enter a caption, or both, to get started.")