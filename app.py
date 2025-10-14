import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🧠 Real-Time Face Detection using YOLO (Streamlit + OpenCV)")

# Load YOLOv8 face detection model
@st.cache_resource
def load_model():
    return YOLO("yolov8n-face.pt")

model = load_model()

# Load the webcam frontend
with open("frontend.html", "r") as f:
    components.html(f.read(), height=650)

# API-like handler: capture image from frontend and detect faces
image_data = st.experimental_get_query_params().get("frame", [None])[0]
if image_data:
    image_bytes = base64.b64decode(image_data.split(",")[1])
    img = Image.open(BytesIO(image_bytes))
    frame = np.array(img)

    # Run YOLO detection
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Faces", channels="BGR", use_column_width=True)
