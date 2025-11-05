import streamlit as st
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="Emotion AI – The Human Mood Analyzer", layout="wide")

st.title("🧠 Emotion AI – The Human Mood Analyzer")
st.markdown("""
### 🔬 Real-time Emotion Detection using Artificial Intelligence
This system detects **human emotions through facial expressions** using Deep Learning.  
Developed using **Python, OpenCV, DeepFace, and Streamlit**.
""")

st.sidebar.header("📸 Camera Control")
run = st.sidebar.checkbox("Start Camera")
st.sidebar.markdown("Press **Stop Camera** to end live emotion detection.")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Store emotion history
emotion_history = []
timestamps = []

# ----------------------------
# Main Camera Loop
# ----------------------------
if run:
    st.sidebar.success("✅ Camera Started Successfully!")
    start_time = time.time()

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("⚠️ Camera not accessible! Try restarting the app.")
            break

        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Display emotion on Streamlit UI
            st.subheader(f"Detected Emotion: **{emotion.upper()}**")

            # Record emotion & time
            emotion_history.append(emotion)
            timestamps.append(round(time.time() - start_time, 1))

            # Show video frame
            FRAME_WINDOW.image(frame)

        except Exception as e:
            st.error(f"Detection error: {str(e)}")
            FRAME_WINDOW.image(frame)

        # Allow manual stop
        run = st.sidebar.checkbox("Start Camera", value=True)

    camera.release()
else:
    st.sidebar.info("🕹️ Turn ON the camera from the sidebar to start.")
    st.write("👆 Click **Start Camera** to begin emotion detection.")

# ----------------------------
# Emotion Summary Dashboard
# ----------------------------
if len(emotion_history) > 0:
    st.markdown("---")
    st.subheader("📊 Emotion Analysis Summary")

    # Convert history to DataFrame
    df = pd.DataFrame({
        'Time (s)': timestamps,
        'Emotion': emotion_history
    })

    # Count each emotion
    counts = df['Emotion'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(counts)

    with col2:
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    # Emotion over time graph
    st.markdown("### ⏱️ Emotion Timeline")
    fig2, ax2 = plt.subplots()
    ax2.plot(df['Time (s)'], df['Emotion'], 'o-', color='purple')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Detected Emotion")
    plt.grid(True)
    st.pyplot(fig2)

    st.success("✅ Analysis complete! You can now view your emotion trend and summary below.")
