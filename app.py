from ultralytics import YOLO
import streamlit as st
import cv2

st.title("🧠 Real-Time Face Detection (YOLO + Streamlit)")

model = YOLO("yolov8n-face.pt")

st.info("Click 'Allow' to enable your webcam.")

run = st.checkbox("Start detection")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("No camera feed detected!")
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    FRAME_WINDOW.image(annotated_frame, channels="BGR")

camera.release()
