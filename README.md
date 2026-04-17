# 🎭 Multimodal Emotion Recognition System

A deep learning-based web application that detects human emotions using both **audio** and **text** inputs. This system leverages advanced multimodal learning techniques by combining speech features and transformer-based text embeddings.

---

## 🚀 Live Demo

👉 https://hemanth-multimodal-emotion-recognition.streamlit.app

---

## 📌 Features

* 🎤 Emotion detection from **speech (audio)**
* 📝 Emotion detection from **text input**
* 🔗 **Multimodal fusion** (audio + text)
* 🤖 Transformer-based text understanding (RoBERTa)
* 🎧 Deep audio feature extraction
* 📊 Confidence score for predictions
* 🌐 Deployed on Streamlit Cloud

---

## 🧠 Model Architecture

The system uses a **Multimodal Deep Learning Model** with:

* **Text Encoder**: RoBERTa (transformer-based NLP model)
* **Audio Features**: Extracted using Librosa (MFCC-based)
* **Fusion Mechanism**: Cross-modal attention + gated fusion
* **Classifier**: Fully connected neural network

---

## 🗂️ Project Structure

```
emotion-recognition/
│
├── model/
│   └── labels.json
│
├── src/
│   ├── model.py          # Model architecture
│   ├── predict.py        # Inference pipeline
│   └── preprocess.py     # Audio preprocessing
│
├── streamlit_app.py      # Web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Hemanth998999/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app locally

```bash
streamlit run streamlit_app.py
```

---

## 📦 Model Handling

Due to GitHub size limitations, the trained model is:

* ❌ Not stored in the repository
* ✅ Automatically downloaded from **Google Drive** using `gdown`

This ensures:

* Lightweight repo
* Scalable deployment

---

## 🎯 Supported Emotions

* 😠 Anger
* 😊 Joy
* 😐 Neutral
* 😢 Sadness

---

## 📊 Example Output

* **Input**: Audio + Text
* **Output**:

  * Emotion label
  * Confidence score

---

## ⚠️ Limitations

* Model accuracy depends on audio quality
* Limited emotion classes
* Cloud deployment does not support microphone input

---

## 🚀 Future Improvements

* 📊 Emotion probability visualization (bar chart)
* 🎥 Video-based emotion detection
* 🤖 Emotion-aware chatbot
* 📈 Improved multimodal fusion techniques

---

## 👨‍💻 Author

**Hemanth Yembuluri**
📧 [hemanthyembuluri777@gmail.com](mailto:hemanthyembuluri777@gmail.com)

---

## 📜 License

This project is for educational and research purposes.
