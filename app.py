import streamlit as st
import cv2
import numpy as np
import face_recognition

st.set_page_config(page_title="Multi-Face Detection", layout="wide")

st.title("👥 Multi-Face Recognition using Streamlit")

st.sidebar.header("Input Options")
source = st.sidebar.radio("Select Input Source", ["Upload Image", "Webcam"])

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    return frame, len(face_locations)

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = np.frombuffer(uploaded_file.read(), np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        processed_frame, count = process_frame(frame)
        st.image(processed_frame, channels="BGR", caption=f"Detected Faces: {count}")
else:
    st.warning("Webcam access isn’t supported directly on Streamlit Cloud. Use ESP32 or local testing.")

st.info("✅ This app detects multiple faces using face_recognition library.")
