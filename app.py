import streamlit as st
import streamlit.components.v1 as components
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ✅ Add this before loading the model
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])

@st.cache_resource
def load_model():
    return YOLO("yolov8n-face.pt")

model = load_model()

# Load the webcam frontend
with open("frontend.html", "r") as f:
    components.html(f.read(), height=700)
