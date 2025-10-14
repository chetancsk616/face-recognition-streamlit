import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import streamlit.components.v1 as components

st.set_page_config(page_title="Face Detection", layout="centered")

# ✅ Fix for PyTorch 2.6 "weights_only" issue
if hasattr(torch, "serialization"):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception as e:
        st.warning(f"Safe global registration skipped: {e}")

# Load YOLO model safely
@st.cache_resource
def load_model():
    try:
        return YOLO("yolov8n-face.pt")
    except Exception as e:
        st.error("⚠️ Error loading YOLO model. Check that yolov8n-face.pt exists.")
        st.stop()

model = load_model()

st.title("😎 Real-Time Face Detection (No Pillow Version)")
st.markdown("This app detects faces from your **laptop camera** or an **uploaded image** using YOLOv8.")

# Sidebar for mode selection
mode = st.sidebar.radio("Select Mode", ["Webcam Feed", "Upload Image"])

# Webcam mode
if mode == "Webcam Feed":
    st.subheader("📸 Live Webcam Feed")
    html_code = """
    <video id="video" autoplay playsinline style="width:100%;max-width:640px;border-radius:12px;border:2px solid #ccc;"></video>
    <p id="status" style="font-family:Arial;margin-top:10px;color:green;">Initializing camera...</p>
    <script>
      const video = document.getElementById('video');
      const status = document.getElementById('status');
      async function initCamera() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          video.srcObject = stream;
          status.textContent = "✅ Camera active!";
        } catch (err) {
          status.textContent = "❌ Unable to access camera: " + err.message;
        }
      }
      initCamera();
    </script>
    """
    components.html(html_code, height=600)

# Upload image mode
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Read image bytes directly
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(img)
        annotated_frame = results[0].plot()

        # Display results
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                 caption="Detected Faces",
                 use_column_width=True)
