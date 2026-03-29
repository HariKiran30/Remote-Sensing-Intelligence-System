import streamlit as st
from ultralytics import YOLO
from PIL import Image
import anthropic
import json
import tempfile
import os
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Object Detector",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Object Detection")
st.caption("Upload a satellite image → detect objects → get an AI-generated intelligence report")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence threshold", 0.1, 0.9, 0.3, 0.05)
    api_key = st.text_input("Anthropic API key", type="password")
    st.markdown("---")

# ── Upload FIRST ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

st.info(
    " No satellite image? Take a screenshot from Google Earth and upload it here."
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n-obb.pt")

model = load_model()

# ── Report generation ─────────────────────────────────────────────────────────
def generate_report(detections: dict, api_key: str) -> str:
    if not api_key:
        return " Add API key in sidebar."

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""
Detection results:
{json.dumps(detections, indent=2)}

Write a short 3-5 sentence report explaining what is observed.
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── Detection function ────────────────────────────────────────────────────────
def run_detection(image: Image.Image, conf: float):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        path = tmp.name
    results = model(path, conf=conf, verbose=False)
    os.unlink(path)
    return results[0]

# ── SHOW RESULTS IMMEDIATELY AFTER UPLOAD ─────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):
        results = run_detection(image, confidence)

    boxes = results.obb if results.obb is not None else results.boxes
    names = model.names

    detected_labels = []
    detection_data = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            conf_score = float(box.conf[0])

            detected_labels.append(label)
            detection_data.append({
                "object": label,
                "confidence": round(conf_score, 2)
            })

    counts = Counter(detected_labels)
    annotated = results.plot()

    with col2:
        st.subheader(f"Detections ({len(detected_labels)})")
        st.image(annotated, use_container_width=True)

    st.markdown("---")
    st.subheader("Detection Summary")

    if counts:
        cols = st.columns(3)
        for i, (label, count) in enumerate(counts.items()):
            cols[i % 3].metric(label.capitalize(), count)

        with st.expander("Details"):
            st.table(detection_data)
    else:
        st.warning("No objects detected. Try lowering confidence.")

    st.markdown("---")
    st.subheader("AI Report")

    summary = {
        "total": len(detected_labels),
        "counts": dict(counts),
        "detections": detection_data
    }

    if st.button("Generate Report"):
        with st.spinner("Generating..."):
            report = generate_report(summary, api_key)
        st.success("Done")
        st.markdown(f"> {report}")

# ── About section LAST ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**About this project**")
st.markdown(
    "This project uses YOLOv8 to detect objects in satellite images and generates "
    "a simple AI-based report. It shows how image data can be converted into useful insights."
)

st.markdown("**Key Features**")
st.markdown(
    "- Object detection using YOLOv8\n"
    "- AI-generated summary\n"
    "- Simple pipeline: image → detection → report"
)

st.markdown("**Topics Covered**")
st.markdown(
    "- Computer Vision\n"
    "- Remote Sensing\n"
    "- Deep Learning\n"
    "- Generative AI"
)

st.markdown("**Dataset:** DOTA")
st.markdown("**Model:** YOLOv8n")