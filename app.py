# streamlit_app.py
import streamlit as st
from deepface import DeepFace
import cv2

st.title("🧠 Emotion AI – The Human Mood Analyzer")
st.write("Analyzing facial expressions to detect your emotion in real-time...")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Camera not found!")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        st.subheader(f"Detected Emotion: **{emotion.upper()}**")
    except:
        pass

    FRAME_WINDOW.image(frame)
