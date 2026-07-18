import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io

class CaptionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Moondream2 model on {self.device}...")
        
        model_id = "vikhyatk/moondream2"
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load the lightweight model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        self.model.eval()

    def generate_caption(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            prompt = "Describe exactly what is happening in this image in one or two neutral, factual sentences."
            
            with torch.no_grad():
                enc_image = self.model.encode_image(image)
                caption = self.model.answer_question(enc_image, prompt, self.tokenizer)
            
            return caption.strip()
            
        except Exception as e:
            print(f"Error in Moondream2 generation: {e}")
            return "failed to generate caption"

# Initialize the engine so main.py can use it seamlessly
caption_engine = CaptionEngine()