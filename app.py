import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

st.set_page_config(page_title="Emotion Detection App", layout="wide")
st.title("😊 Real-Time Emotion Detection App")
st.markdown("Detect your emotions live using your webcam!")

# Start/stop camera
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# Initialize camera
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access webcam.")
        break

    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)

    try:
        # Detect emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        # Draw emotion text
        cv2.putText(frame, f"Emotion: {dominant_emotion}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        cv2.putText(frame, "Detecting...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    st.write("✅ Click the checkbox to start the webcam.")
    camera.release()
