import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import BertTokenizer

from model_arch import BertBiLSTM
from imagecaption import caption_engine

app = FastAPI(title="Toxicity Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explicitly setting output_dim=6 to match your training
model = BertBiLSTM(hidden_dim=256, output_dim=6)
model_path = os.path.join("weights", "bilstm_model.pth")

if os.path.exists(model_path):
    # map_location is crucial for loading trained GPU weights onto CPU or specific GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Toxicity model weights loaded successfully!")
else:
    print(f"⚠️ Warning: Model weights not found at {model_path}.")

model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
        
    try:
        image_bytes = await file.read()
        description = caption_engine.generate_caption(image_bytes)
        print(f"DEBUG: Description generated: {description}")
        
        inputs = tokenizer(
            description, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        input_ids = inputs["input_ids"].to(device)
        lengths = torch.tensor([input_ids.size(1)], dtype=torch.int64)
        
        with torch.no_grad():
            logits = model(input_ids, lengths)
            print(f"DEBUG: Raw Logits from model: {logits}") # Check if these are zeros
            probs = torch.sigmoid(logits)
        
        scores = probs.cpu().numpy().tolist()[0]
        results = {label: float(score) for label, score in zip(LABELS, scores)}
        
        return {
            "status": "success",
            "description": description,
            "toxicity_scores": results
        }
        
    except Exception as e:
        print(f"DEBUG: Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))