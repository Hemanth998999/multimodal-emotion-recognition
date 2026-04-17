import streamlit as st
from src.preprocess import extract_audio_features
from src.predict import predict
import sounddevice as sd
from scipy.io.wavfile import write

st.title("🎭 Multimodal Emotion Recognition")

uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
text_input = st.text_area("Enter Text")


def record_audio():
    fs = 22050
    seconds = 3
    st.info("Recording...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("live.wav", fs, audio)
    return "live.wav"


if st.button("Predict"):
    if uploaded_file and text_input:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        features = extract_audio_features("temp.wav")
        result, confidence = predict(features, text_input)

        st.success(f"Emotion: {result}")
        st.info(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Provide both audio and text")


if st.button("🎤 Record & Predict"):
    if text_input:
        file_path = record_audio()
        features = extract_audio_features(file_path)
        result, confidence = predict(features, text_input)

        st.success(f"Emotion: {result}")
        st.info(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Enter text first")