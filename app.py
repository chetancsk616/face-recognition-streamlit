import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Real-Time Face Detection", layout="wide")
st.title("🧠 Real-Time Face Detection using YOLO (Streamlit + OpenCV)")

# Load pretrained face detection model
model = YOLO("yolov8n-face.pt")  # small, fast, accurate face model

st.info("Click 'Allow' when prompted to give camera permission.")

img_file_buffer = st.camera_input("Capture an image")

if img_file_buffer:
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    results = model.predict(img, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Faces", use_column_width=True)
