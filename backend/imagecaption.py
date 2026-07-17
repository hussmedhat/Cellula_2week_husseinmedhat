import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self):
        # ba5tar el model 
        self.model_id = "Salesforce/blip-image-captioning-base"
        
        # baload el processor 
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        
        # baload el model
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id)

    def generate_caption(self, image_bytes: bytes) -> str:
        # 1. ba7awel el bytes le image 3ashan a2dar astakhdemha ma3a el processor
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 2. Ba process el soora 3ashan atala3 tensors
        inputs = self.processor(image, return_tensors="pt")
        
        # 3. hena ba3mel generation lel tokens elly hayet7awel le caption
        out = self.model.generate(**inputs)
        
        # 4. hena 3amalt decode 3ashan atala3 el caption
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption

caption_engine = ImageCaptioner()