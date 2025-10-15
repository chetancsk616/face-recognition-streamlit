import streamlit as st
import cv2
import numpy as np
import base64

st.set_page_config(page_title="Remote Face Detection API")

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("🧠 Remote Face Detection API")
st.write("Send a POST request with an image — returns detected faces and annotated image (base64).")

def detect_faces(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8"), len(faces)

# UI testing (manual uploads)
file = st.file_uploader("Test locally", type=["jpg", "jpeg", "png"])
if file:
    img_b64, count = detect_faces(file.read())
    st.image(cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR),
             channels="BGR", caption=f"{count} face(s) detected")

# Handle POST from other Streamlit apps
import streamlit.web.server.websocket_headers
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.runtime import Runtime

# "API" simulation via POST (for your local app)
if st.query_params.get("api") == "analyze":
    import json, sys
    try:
        uploaded = st.file_uploader("Upload via API")
        if uploaded:
            img_b64, count = detect_faces(uploaded.read())
            st.json({"faces": count, "image": img_b64})
    except Exception as e:
        st.write(f"Error: {e}")
