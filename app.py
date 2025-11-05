import streamlit as st
from deepface import DeepFace
import matplotlib.pyplot as plt

st.set_page_config(page_title="Emotion AI – The Human Mood Analyzer", layout="wide")

st.title("🧠 Emotion AI – The Human Mood Analyzer")
st.write("Upload an image to analyze emotions using AI.")

uploaded_file = st.file_uploader("📸 Upload a Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Analyzing emotions..."):
        result = DeepFace.analyze(uploaded_file, actions=['emotion'], enforce_detection=False)
    st.success("✅ Emotion Analysis Complete!")

    dominant_emotion = result[0]['dominant_emotion']
    st.subheader(f"Detected Emotion: **{dominant_emotion.upper()}**")

    emotions = result[0]['emotion']
    fig, ax = plt.subplots()
    ax.bar(emotions.keys(), emotions.values())
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Emotion Distribution")
    st.pyplot(fig)
