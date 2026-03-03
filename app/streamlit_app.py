from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import List

import streamlit as st
import torch
from PIL import Image

# Ensure repo root is on sys.path so `import src...` works when Streamlit runs
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import build_fasterrcnn_resnet50_fpn, predict
from src.utils_viz import draw_detections

NUM_CLASSES = 12
CLASS_NAMES: List[str] = [
    "background",
    "Passenger Vehicle",
    "Truck",
    "Engineering Vehicle",
    "Railway Vehicle",
    "Maritime Vessel",
    "Building",
    "Helipad",
    "Storage Tank",
    "Shipping Container",
    "Pylon",
    "Aircraft",
]

# Default local weights, matching our release asset
DEFAULT_WEIGHTS_PATH = Path("models/FasterRCNN/ResNet50_best_model.pth")
MODEL_RELEASE_URL = "https://github.com/MarcSaghiah/xView_detect/releases/download/v1.0-baseline/ResNet50_best_model.pth"  # <- update <OWNER>/<REPO>

def ensure_weights_exist(weights_path: Path, model_release_url: str):
    """
    If weights_path does not exist, download from release URL using src/download_weights.py.
    """
    if not weights_path.exists():
        st.warning(f"Model weights not found at: {weights_path}\n\nDownloading from GitHub Release asset...")
        try:
            # Call the download script (silent mode unless error)
            result = subprocess.run(
                [sys.executable, "src/download_weights.py"],
                check=True,
                capture_output=True,
                text=True
            )
            st.info(result.stdout)
        except Exception as e:
            st.error(f"Unable to download weights automatically.\n"
                     f"Please run `python src/download_weights.py` yourself.\nError: {e}")
            st.stop()

@st.cache_resource
def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fasterrcnn_resnet50_fpn(NUM_CLASSES, Path(weights_path), device)
    return model, device


def main() -> None:
    st.set_page_config(page_title="xView_detect Demo", layout="wide")
    st.title("Satellite Object Detection (Faster R-CNN)")

    st.sidebar.header("Settings")
    weights_path = st.sidebar.text_input("Model weights path", value=str(DEFAULT_WEIGHTS_PATH))
    score_threshold = st.sidebar.slider(
        "Score threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )

    uploaded = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png", "tif", "tiff"])

    ensure_weights_exist(Path(weights_path), MODEL_RELEASE_URL)

    if not uploaded:
        st.info("Upload an image to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with st.spinner("Loading model..."):
        model, device = load_model(weights_path)

    with st.spinner("Running inference..."):
        detections = predict(model, image, device=device, score_threshold=score_threshold)

    with col2:
        st.subheader(f"Output ({len(detections)} detections)")
        if detections:
            vis = draw_detections(image, detections, CLASS_NAMES)
            st.image(vis, use_container_width=True)
        else:
            st.warning("No detections above the selected threshold.")

    st.subheader("Detections table")
    st.table(
        [
            {
                "label": CLASS_NAMES[d.label] if d.label < len(CLASS_NAMES) else d.label,
                "score": d.score,
                "x1": d.box_xyxy[0],
                "y1": d.box_xyxy[1],
                "x2": d.box_xyxy[2],
                "y2": d.box_xyxy[3],
            }
            for d in detections
        ]
    )


if __name__ == "__main__":
    main()