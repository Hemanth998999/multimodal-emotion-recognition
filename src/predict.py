import os
import gdown
import torch
import json
import sys
from transformers import AutoTokenizer
from src.model import MultimodalEmotionModel, CrossModalAttention

# 🔥 map classes
sys.modules['__main__'].MultimodalEmotionModel = MultimodalEmotionModel
sys.modules['__main__'].CrossModalAttention = CrossModalAttention

MODEL_PATH = "model/best_model_full.pt"

# 🔥 download if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=17skLeBlw1rfJuFmEfcG063wGZOuXIhyv",
        MODEL_PATH,
        quiet=False
    )

# 🔥 load model
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

if hasattr(model, "module"):
    model = model.module

model.eval()

# labels
with open("model/labels.json", "r") as f:
    emotions = json.load(f)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def predict(audio_features, text):
    with torch.no_grad():
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        output = model(
            inputs["input_ids"],
            inputs["attention_mask"],
            audio_tensor
        )

        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

        return emotions[pred], confidence