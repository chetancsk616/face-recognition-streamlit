import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Real-time Face Recognition", layout="wide")

st.title("🎯 Real-time Face Detection App")

# Load YOLO model (face detection)
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n-face.pt")  # Make sure this file exists in your repo
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()
    return model

model = load_model()

# Camera input (Streamlit widget)
st.sidebar.header("📷 Camera Settings")
use_camera = st.sidebar.toggle("Use Camera", value=True)

if use_camera:
    st.info("Turn on your camera and allow browser access.")
    camera_input = st.camera_input("Capture a Frame")
else:
    camera_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display and detect
if camera_input is not None:
    with st.spinner("Processing..."):
        img = Image.open(camera_input)
        img_array = np.array(img)

        # Run YOLO face detection
        results = model.predict(source=img_array, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Draw boxes on the image
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert BGR → RGB and display
        st.image(img_array, caption="Detected Faces", use_column_width=True)

else:
    st.warning("Please capture or upload an image to start detection.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + YOLOv8")
