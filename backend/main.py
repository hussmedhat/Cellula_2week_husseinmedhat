import os
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import BertTokenizer

# Import custom modules
from model_arch import BertBiLSTM
from imagecaption import caption_engine
from image_classifier import get_clip_scores
from db_logger import log_prediction

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


BASE_BERT_WEIGHT = 0.8
BASE_CLIP_WEIGHT = 0.2

CLIP_CONFIDENCE_THRESHOLD = 0.75  # CLIP score needs to clear this to be "confident"
HIGH_CONF_CLIP_WEIGHT = 0.6       # weight CLIP gets on that label when confident


def empty_clip_dict():
    return {label: None for label in LABELS}


@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    caption: Optional[str] = Form(None),
):
    has_image = file is not None and bool(file.filename)
    user_text = caption.strip() if caption else ""
    has_text = bool(user_text)

    if not has_image and not has_text:
        raise HTTPException(status_code=400, detail="Provide an image, a caption, or both.")

    if has_image and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read() if has_image else None

        # law wa5ed input soora bas
        generated_caption = None
        clip_scores = None

        if has_image:
            generated_caption, clip_scores = await asyncio.gather(
                asyncio.to_thread(caption_engine.generate_caption, image_bytes),
                asyncio.to_thread(get_clip_scores, image_bytes),
            )
            print(f"DEBUG: Moondream2 Generated Description: {generated_caption}")
            print(f"DEBUG: CLIP Visual Scores: {clip_scores}")

            if generated_caption == "failed to generate caption":
                if has_text:
                    # hena el user kaman da5al text fa lazem na5od da el caption elly da5aloh el user w n3ml scoring 3aleh
                    generated_caption = None
                else:
                    # ya2ema partial failure ya2ema howa el user mada5alsh ay text fa lazem n3ml scoring 3ala el clip scores bas
                    bert_scores = empty_clip_dict()
                    fused_scores = clip_scores
                    await asyncio.to_thread(
                        log_prediction, "failed to generate caption", bert_scores, clip_scores, fused_scores, "partial_failure"
                    )
                    return {
                        "status": "partial_failure",
                        "description": "failed to generate caption",
                        "toxicity_scores": fused_scores,
                        "raw_bert_scores": bert_scores,
                        "raw_clip_scores": clip_scores,
                        "warning": "Caption generation failed; scores are CLIP-only.",
                    }

        # ---------------------------------------------------------
        # Build the text BERT will actually score. By this point we're
        # guaranteed to have SOMETHING: either user text, a generated
        # caption, or both.
        # ---------------------------------------------------------
        if has_text and generated_caption:
            description = f"{user_text}. {generated_caption}"
        elif has_text:
            description = user_text
        else:
            description = generated_caption

        # PHASE B: Linguistic Analysis (BertBiLSTM) -- always runs
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

        # PHASE C: Fusion -- only blend in CLIP if we actually have image scores

        fused_scores = {}
        for label in LABELS:
            bert_score = bert_scores[label]
            if clip_scores is not None:
                clip_score = clip_scores[label]
                w_clip = HIGH_CONF_CLIP_WEIGHT if clip_score >= CLIP_CONFIDENCE_THRESHOLD else BASE_CLIP_WEIGHT
                w_bert = 1 - w_clip
                fused_scores[label] = (bert_score * w_bert) + (clip_score * w_clip)
            else:
                fused_scores[label] = bert_score

        response_clip_scores = clip_scores if clip_scores is not None else empty_clip_dict()

        await asyncio.to_thread(
            log_prediction, description, bert_scores, response_clip_scores, fused_scores, "success"
        )

        return {
            "status": "success",
            "description": description,
            "toxicity_scores": fused_scores,
            "raw_bert_scores": bert_scores,
            "raw_clip_scores": response_clip_scores
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))