import streamlit as st
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Face Detection API", layout="wide")
st.title("📡 Face Detection API (via POST upload)")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    _, buf = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buf).decode('utf-8')
    return img_base64, len(faces)

# Simulated endpoint using Streamlit form
st.write("### Upload image manually to test")
file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if file:
    img_b64, count = detect_faces(file.read())
    st.image(Image.open(BytesIO(base64.b64decode(img_b64))), caption=f"{count} face(s) detected.")

# Simple API mock (since Streamlit doesn’t have native Flask-like endpoints)
from streamlit.web.server.websocket_headers import _get_websocket_headers
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.runtime import Runtime

# Define a fake REST-like route via query param
query_params = st.experimental_get_query_params()
if "api" in query_params and query_params["api"][0] == "analyze":
    import os, json
    if "file" in st.session_state:
        image_bytes = st.session_state["file"]
        img_b64, count = detect_faces(image_bytes)
        st.json({"faces": count, "image": img_b64})
