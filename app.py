import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("🧠 Real-Time Face Detection using YOLO (Streamlit + OpenCV)")

# Load YOLOv8 face detection model
model = YOLO("yolov8n-face.pt")

# Render frontend webcam UI
with open("frontend.html", "r") as f:
    components.html(f.read(), height=700)
