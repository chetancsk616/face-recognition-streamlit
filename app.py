import streamlit as st
import streamlit.components.v1 as components
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ✅ Fix PyTorch 2.6 "weights_only" issue
from torch.serialization import add_safe_globals, safe_globals
from ultralytics.nn.tasks import DetectionModel

add_safe_globals([DetectionModel])

# Force YOLO to load using weights_only=False
def patched_torch_load(file, *args, **kwargs):
    kwargs["weights_only"] = False
    return torch.load(file, *args, **kwargs)

torch.load = patched_torch_load  # ⚡ monkey patch before YOLO load

# Load YOLOv8 face model safely
@st.cache_resource
def load_model():
    return YOLO("yolov8n-face.pt")

model = load_model()

# Load frontend
with open("frontend.html", "r") as f:
    components.html(f.read(), height=700)
