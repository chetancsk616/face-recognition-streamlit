import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="Face Detection with OpenCV", layout="wide")

st.title("👁️ Face Detection using OpenCV")

st.write("Upload an image to detect faces. This demo runs fully on Streamlit Cloud — no dlib, no heavy build!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's built-in Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption=f"Detected Faces: {len(faces)}",
        use_column_width=True,
    )
else:
    st.info("👆 Upload an image to start face detection.")
