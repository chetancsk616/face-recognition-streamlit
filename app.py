import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Face Detection App", layout="centered")

# Load YOLO model (using pretrained face detector)
@st.cache_resource
def load_model():
    return YOLO("yolov8n-face.pt")

model = load_model()

st.title("😎 Real-Time Face Detection (No Pillow Version)")
st.markdown("This app uses your **laptop camera** or an uploaded image to detect faces.")

# Sidebar options
mode = st.sidebar.radio("Choose Mode", ["Webcam Feed", "Upload Image"])

# Load webcam feed
if mode == "Webcam Feed":
    st.subheader("📸 Live Webcam Feed")
    with open("frontend.html", "r") as f:
        components.html(f.read(), height=700)

# Image Upload
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image directly with OpenCV (no Pillow)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run YOLO face detection
        results = model(img)
        annotated_frame = results[0].plot()

        # Convert BGR → RGB for Streamlit
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Detected Faces", use_column_width=True)

        # Optional: Save detections
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            cv2.imwrite(tmp_file.name, annotated_frame)
            st.download_button(
                label="Download Result Image",
                data=open(tmp_file.name, "rb").read(),
                file_name="detected_faces.jpg",
                mime="image/jpeg"
            )
            os.unlink(tmp_file.name)
