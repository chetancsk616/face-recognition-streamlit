import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import tempfile

st.set_page_config(page_title="Face Recognition Demo", layout="wide")
st.title("🎥 Real-Time Face Recognition using Laptop Camera")

# Allow camera access
st.info("Click 'Allow' when prompted to give camera permission.")

# Capture a frame from the laptop camera
img_file_buffer = st.camera_input("Capture a photo")

if img_file_buffer is not None:
    # Convert the captured image to a format usable by OpenCV
    bytes_data = img_file_buffer.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.image(rgb_img, caption="Captured Image", use_column_width=True)

    st.write("🔍 Analyzing face... please wait")

    try:
        result = DeepFace.analyze(
            rgb_img,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False
        )
        st.success("✅ Face detected and analyzed!")
        st.json(result)
    except Exception as e:
        st.error(f"❌ Error: {e}")
