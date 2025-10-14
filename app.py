import streamlit as st
import streamlit.components.v1 as components
import base64, cv2, numpy as np, io
from ultralytics import YOLO
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from threading import Thread
import uvicorn

# Initialize YOLO model
model = YOLO("yolov8n-face.pt")

# ---------- FASTAPI BACKEND ----------
app_fastapi = FastAPI()

class Frame(BaseModel):
    image: str

@app_fastapi.post("/detect")
def detect(frame: Frame):
    img_data = base64.b64decode(frame.image.split(",")[1])
    img = Image.open(io.BytesIO(img_data))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    results = model(img_cv)
    annotated = results[0].plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {"image": "data:image/jpeg;base64," + annotated_base64}

def run_fastapi():
    uvicorn.run(app_fastapi, host="0.0.0.0", port=7861)

# Run FastAPI in background thread
Thread(target=run_fastapi, daemon=True).start()

# ---------- STREAMLIT FRONTEND ----------
st.set_page_config(page_title="Realtime Face Recognition", layout="wide")
st.title("👁️ Realtime Face Detection via Webcam")

st.markdown(
    """
    This app uses your browser's webcam (via JavaScript) and sends frames to a YOLO model 
    running on the backend for real-time face detection.
    """
)

with open("frontend.html", "r") as f:
    components.html(f.read(), height=700)
