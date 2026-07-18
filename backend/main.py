import os
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import BertTokenizer

# Import custom modules
from model_arch import BertBiLSTM
from imagecaption import caption_engine
from image_classifier import get_clip_scores

app = FastAPI(title="Toxicity Classification API")

# 1. CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # "*" + credentials=True is invalid per the CORS spec; browsers reject it.
    allow_methods=["*"],      # If you need cookies/auth headers, replace "*" with real frontend origin(s)
    allow_headers=["*"],      # and set allow_credentials=True instead.
)

# 2. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Initialize and Load the Text Toxicity Model (BertBiLSTM)
model = BertBiLSTM(hidden_dim=256, output_dim=6)
model_path = os.path.join("weights", "bilstm_model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Toxicity model weights loaded successfully!")
else:
    print(f"⚠️ Warning: Model weights not found at {model_path}.")

model.to(device)
model.eval()

# 4. Load Tokenizer & Define Labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---------------------------------------------------------
# Fusion weighting config
# ---------------------------------------------------------
# BERT is the trained classifier here, so it's the default authority.
# CLIP is zero-shot and its "unsure" state sits near 0.5 (noise), so it
# only earns more influence on a label when IT ITSELF is confident about
# that label -- not when certain words show up in the caption.
BASE_BERT_WEIGHT = 0.8
BASE_CLIP_WEIGHT = 0.2

CLIP_CONFIDENCE_THRESHOLD = 0.75  # CLIP score needs to clear this to be "confident"
HIGH_CONF_CLIP_WEIGHT = 0.6       # weight CLIP gets on that label when confident


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read image bytes once to be used by both Moondream2 and CLIP
        image_bytes = await file.read()

        # ---------------------------------------------------------
        # PHASE A: Moondream (captioning) and CLIP (visual scoring) don't
        # depend on each other's output, so run them concurrently instead
        # of back-to-back. Both are sync/blocking calls, so we offload
        # them to threads rather than await-ing them serially.
        # ---------------------------------------------------------
        description, clip_scores = await asyncio.gather(
            asyncio.to_thread(caption_engine.generate_caption, image_bytes),
            asyncio.to_thread(get_clip_scores, image_bytes),
        )
        print(f"DEBUG: Moondream2 Generated Description: {description}")
        print(f"DEBUG: CLIP Visual Scores: {clip_scores}")

        # If captioning failed, don't silently tokenize the error string and
        # score it as if it were a real description — that produces a
        # confident-looking but meaningless bert_scores dict.
        if description == "failed to generate caption":
            bert_scores = {label: None for label in LABELS}
            fused_scores = clip_scores  # fall back to CLIP-only signal
            return {
                "status": "partial_failure",
                "description": description,
                "toxicity_scores": fused_scores,
                "raw_bert_scores": bert_scores,
                "raw_clip_scores": clip_scores,
                "warning": "Caption generation failed; scores are CLIP-only.",
            }

        # ---------------------------------------------------------
        # PHASE B: Linguistic Analysis (BertBiLSTM on the caption)
        # ---------------------------------------------------------
        inputs = tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        lengths = attention_mask.sum(dim=1).cpu()

        with torch.no_grad():
            logits = model(input_ids, lengths)
            probs = torch.sigmoid(logits)

        scores = probs.cpu().numpy().tolist()[0]
        bert_scores = {label: float(score) for label, score in zip(LABELS, scores)}

        # ---------------------------------------------------------
        # PHASE C: Multi-Modal Fusion (Per-Label Confidence-Weighted)
        # ---------------------------------------------------------
        # No keyword scanning of the caption text. Each label's blend is
        # decided independently, based on whether CLIP is actually
        # confident about THAT label -- not on whether some unrelated
        # word appeared in the description.
        fused_scores = {}
        for label in LABELS:
            clip_score = clip_scores[label]
            bert_score = bert_scores[label]

            if clip_score >= CLIP_CONFIDENCE_THRESHOLD:
                w_clip = HIGH_CONF_CLIP_WEIGHT
            else:
                w_clip = BASE_CLIP_WEIGHT
            w_bert = 1 - w_clip

            fused_scores[label] = (bert_score * w_bert) + (clip_score * w_clip)

        return {
            "status": "success",
            "description": description,
            "toxicity_scores": fused_scores,
            "raw_bert_scores": bert_scores,
            "raw_clip_scores": clip_scores
        }

    except Exception as e:
        print(f"DEBUG: Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))