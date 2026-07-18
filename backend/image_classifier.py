import torch
import clip
from PIL import Image
import io

# 1. Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load the CLIP Model and Preprocessor
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 3. Each label gets a (positive, negative) prompt pair instead of a single
# prompt. This is the fix: comparing "toxic" only against "not toxic" keeps
# labels independent, instead of softmaxing across all 6 unrelated concepts
# (which forced them to compete for a shared probability budget — a threat
# image would mechanically suppress obscene/insult/etc. scores even if both
# were true).
CLIP_PROMPT_MAPPING = {
    "toxic": (
        "a scene showing danger signs, hazard warnings, radiation or biohazard symbols, or toxic/contaminated material",
        ["a nature photo", "a landscape photo", "an ordinary urban photo", "a peaceful scene"],
    ),
    "severe_toxic": (
        "an extremely violent scene with active fighting, blood, gore, fire, or explosions happening",
        ["a nature photo", "a calm outdoor scene", "an abandoned or decayed building with no violence"],
    ),
    "obscene": (
        "an obscene, vulgar, or sexually explicit scene",
        ["a nature photo", "a landscape or garden photo", "a fully clothed, non-sexual photo", "an ordinary everyday photo"],
    ),
    "threat": (
        "a person holding a weapon or making a physical threat toward someone",
        ["a nature photo with no people", "a peaceful photo", "a person relaxing or smiling"],
    ),
    "insult": (
        "a person making an offensive, mocking, or insulting gesture at someone",
        ["a nature photo with no people", "a person with a neutral expression", "an ordinary photo"],
    ),
    "identity_hate": (
        "a scene depicting racist symbols, hate speech text, or discriminatory imagery",
        ["a nature photo", "an ordinary diverse scene", "a photo with no symbols or text"],
    ),
}

LABELS = list(CLIP_PROMPT_MAPPING.keys())

# Encode positives normally, but average several negative anchors per label
# instead of relying on one under-specified negative phrase.
pos_prompts = [CLIP_PROMPT_MAPPING[label][0] for label in LABELS]
pos_tokens = clip.tokenize(pos_prompts).to(device)

with torch.no_grad():
    pos_features = model.encode_text(pos_tokens)
    pos_features /= pos_features.norm(dim=-1, keepdim=True)

    neg_features_per_label = []
    for label in LABELS:
        neg_prompts = CLIP_PROMPT_MAPPING[label][1]
        neg_tokens = clip.tokenize(neg_prompts).to(device)
        neg_feats = model.encode_text(neg_tokens)
        neg_feats /= neg_feats.norm(dim=-1, keepdim=True)
        neg_features_per_label.append(neg_feats.mean(dim=0, keepdim=True))  # average multiple negatives
    neg_features = torch.cat(neg_features_per_label, dim=0)
    neg_features /= neg_features.norm(dim=-1, keepdim=True)  # renormalize after averaging


def get_clip_scores(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            pos_logits = (logit_scale * image_features @ pos_features.T).squeeze(0)
            neg_logits = (logit_scale * image_features @ neg_features.T).squeeze(0)

        results = {}
        for i, label in enumerate(LABELS):
            pair_probs = torch.softmax(torch.stack([pos_logits[i], neg_logits[i]]), dim=0)
            results[label] = float(pair_probs[0])

        return results

    except Exception as e:
        print(f"Error in CLIP visual classification: {e}")
        return {label: 0.0 for label in LABELS}