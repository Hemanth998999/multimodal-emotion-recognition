import streamlit as st
from src.preprocess import extract_audio_features
from src.predict import predict

st.title("🎭 Multimodal Emotion Recognition")

uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
text_input = st.text_area("Enter Text")

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