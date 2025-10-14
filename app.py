import streamlit as st
import streamlit.components.v1 as components
import torch
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
import cv2
import numpy as np
from PIL import Image
import base64
import io

# ================================
# ✅ Fix for PyTorch 2.6+ (weights_only error)
# ================================
add_safe_globals([DetectionModel])
_original_torch_load = torch.load

def patched_torch_load(file, *args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(file, *args, **kwargs)

torch.load = patched_torch_load

# ================================
# ✅ Load YOLOv8 Face Model
# ================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n-face.pt")

model = load_model()

# ================================
# ✅ Streamlit Page Config
# ================================
st.set_page_config(page_title="Real-Time Face Detection", layout="wide")
st.title("👤 Real-Time Face Detection (YOLOv8 + Streamlit)")

st.markdown(
    """
    This demo uses **YOLOv8n-face** for real-time face detection directly in your browser.  
    Click **Start Camera** below to begin.
    """
)

# ================================
# ✅ JavaScript Camera Capture Frontend
# ================================
frontend_html = """
<div style="text-align:center;">
  <video id="video" width="640" height="480" autoplay playsinline></video><br>
  <button onclick="startCamera()">Start Camera</button>
  <button onclick="captureFrame()">Capture Frame</button>
  <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
  <p id="status" style="color:green;"></p>
</div>

<script>
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const statusText = document.getElementById('status');

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      statusText.innerText = "✅ Camera started successfully!";
    } catch (err) {
      statusText.innerText = "❌ Error: " + err.message;
    }
  }

  async function captureFrame() {
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg');
    const response = await fetch('/capture', {
      method: 'POST',
      body: dataUrl
    });
    const result = await response.text();
    statusText.innerText = result;
  }
</script>
"""

components.html(frontend_html, height=650)

# ================================
# ✅ Handle Backend / Capture API
# ================================
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web.server import routes

if not hasattr(routes, "added"):
    from fastapi import Request
    from starlette.responses import PlainTextResponse

    @routes.post("/capture")
    async def capture_image(request: Request):
        data_url = await request.body()
        header, encoded = data_url.split(b",", 1)
        image_data = base64.b64decode(encoded)

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_array = np.array(img)

        # Run YOLOv8 face detection
        results = model(img_array, verbose=False)

        # Count detected faces
        faces = len(results[0].boxes) if results[0].boxes is not None else 0
        return PlainTextResponse(f"Detected {faces} face(s)!")

    routes.added = True

# ================================
# ✅ Streamlit Footer
# ================================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit + YOLOv8 Face Model")
