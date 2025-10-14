import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import base64, io, json

st.set_page_config(page_title="Real-time Face Detection", layout="wide")
st.title("👁️ Real-time Face Detection using YOLO")

# Load YOLO face model
model = YOLO("yolov8n-face.pt")

# Endpoint for frame detection (Streamlit's experimental API feature)
if st.experimental_user.is_active():  # Normal UI rendering
    with open("frontend.html", "r") as f:
        components.html(f.read(), height=700)
else:
    # Handle API request (for POST /detect)
    import os
    from streamlit.web.server.websocket_headers import _get_websocket_headers

    def handle_request():
        headers = _get_websocket_headers()
        if "Content-Type" in headers and headers["Content-Type"] == "application/json":
            body = st.experimental_get_query_params().get("body", None)
            if not body:
                return
            frame = json.loads(body)
            img_data = base64.b64decode(frame["image"].split(",")[1])
            img = Image.open(io.BytesIO(img_data))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results = model(img_cv)
            annotated = results[0].plot()
            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_base64 = base64.b64encode(buffer).decode("utf-8")
            return {"image": "data:image/jpeg;base64," + annotated_base64}
    handle_request()
