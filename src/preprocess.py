import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# load once (global)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

SAMPLE_RATE = 16000
MAX_AUDIO_LEN = 80000  # 5 sec


def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(y) > MAX_AUDIO_LEN:
        y = y[:MAX_AUDIO_LEN]

    inputs = processor(
        y,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    ).input_values

    with torch.no_grad():
        output = model(inputs).last_hidden_state  # (1, T, 768)

    features = output.mean(dim=1).squeeze(0).numpy()  # (768,)

    return features