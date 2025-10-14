import streamlit as st
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

st.set_page_config(page_title="Face Recognition App", layout="wide")

st.title("🎥 Face Recognition using YOLOv8")

# Load YOLO model safely
@st.cache_resource
def load_model():
    return YOLO("yolov8n-face.pt")

model = load_model()

# JavaScript frontend loader
with open("frontend.html", "r") as f:
    frontend_html = f.read()

# Display webcam capture UI
st.components.v1.html(frontend_html, height=500)

# Streamlit receives image data from JS
img_data = st.session_state.get("img_data", None)

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

# Display image from either webcam or upload
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.captured_image = image

if st.session_state.captured_image is not None:
    st.image(st.session_state.captured_image, caption="Captured Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(st.session_state.captured_image)
    results = model(img_array)

    # Draw bounding boxes
    annotated = results[0].plot()
    st.image(annotated, caption="Detection Result", use_column_width=True)
