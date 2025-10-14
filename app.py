import streamlit as st
import cv2, numpy as np
import mediapipe as mp
from PIL import Image

st.title("Face Detection (Mediapipe)")

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    results = mp_face.process(img)
    out = img.copy()
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h,w,_ = out.shape
            x1 = int(bbox.xmin * w); y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w); y2 = int((bbox.ymin + bbox.height) * h)
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    st.image(out, use_column_width=True)
