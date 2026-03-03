# xView_detect — Satellite Object Detection with Faster R-CNN

## Overview

This repository implements a **satellite imagery object detection** project using the **xView dataset** and a **Faster R-CNN (ResNet50-FPN)** detector. It provides:

- A **reusable trained model checkpoint** (stored under `models/` and available in Releases)
- **Notebooks** documenting the exploration, preprocessing, training, and evaluation process
- A **Streamlit app** allowing users to upload a satellite image and run object detection locally (and optionally online)
- A **command-line inference script** and auto-download of the model weights from GitHub Releases

The current model is a **baseline**. Predictions may be imperfect for some classes/scales, but the pipeline is complete, reproducible, and designed to allow future improvement.

---

## Why this project?

Object detection on satellite imagery is useful for:
- infrastructure monitoring (buildings, tanks, pylons)
- transportation and logistics analytics (vehicles, containers)
- maritime and aviation monitoring (vessels, aircraft)
- rapid mapping / situational awareness (disaster response, security, etc.)

The key difficulty is that satellite objects can be **small**, **dense**, and appear at different **scales** and **background contexts**.

---

## Dataset: xView (high-level)

This project is built around the **xView** dataset, which provides:
- satellite images
- object annotations (bounding boxes)
- multiple object categories (vehicles, buildings, aircraft, vessels, etc.)

> Note: xView is large. This repo focuses on the modeling pipeline, and expects users to have access to the dataset separately if they want to retrain.

---

## Project Journey (What was done)

### 1) Problem framing & approach selection
- Framed as a **supervised object detection** problem.
- **Faster R-CNN (ResNet50-FPN) chosen for**: strong baseline performance, stable training, multi-scale feature extraction via FPN, compatibility with torchvision.

### 2) Data preparation
- Converted xView annotations to `torchvision` detection targets (`boxes`, `labels` format).
- Conducted train/val splits and sanity-checks (visualization in notebooks/scripts).

### 3) Baseline modeling and training
- Initialized Faster R-CNN ResNet50-FPN architecture.
- Replaced classification head for correct class count.
- Trained, checkpointed, and saved the best model as a raw `state_dict`.

### 4) Inference packaging
- Modular inference code (`src/inference.py`) and visualization (`src/utils_viz.py`), usable via CLI or Streamlit.

### 5) Application (Streamlit)
- User-friendly demo app:
  - upload satellite image
  - adjust threshold
  - visualize bounding boxes + detection table

### 6) Model Delivery Strategy
- **Model weights are available both directly in the repository and as a Release asset.**
- An auto-download script ensures seamless setup for both local and cloud/deployable usage.

---

## Results

### What works well
- End-to-end pipeline (model loading → prediction → visualization).
- Modular code: easy adaptation, reuse, and retraining.
- Streamlit demo is accessible for easy testing and demonstration.

### Limitations (baseline)
- Detection quality varies across classes and scenes.
- Small objects and class imbalance decrease accuracy.
- Dense scenes/backgrounds may cause missed or spurious detections.
- No COCO-style mAP reported yet (can be added for future evaluation).

### Next Steps / Improvements
- Advanced data augmentation (random crop, scale jitter, etc.)
- Hyperparameter tuning, anchor optimization
- Longer training, better validation
- Improvements to handle class imbalance
- Adding evaluation scripts and metrics (e.g. COCO mAP)
- Stronger models as alternatives (RetinaNet, Mask R-CNN, etc.)

---

## Deliverables

### 1) Reusable inference code (`src/`)
- `src/inference.py` — loads model and runs predictions on any PIL image.
- `src/utils_viz.py` — draws bounding boxes, labels for visualization.
- Usable as a standalone Python module.

### 2) Trained model checkpoint (`models/`)
Your best model is distributed in two ways:

- Committed directly: `models/FasterRCNN/ResNet50_best_model.pth`
- Available in Releases: [Download release weights](https://github.com/MarcSaghiah/xView_detect/releases/download/v1.0-baseline/ResNet50_best_model.pth)

You can also auto-download using:

```bash
python src/download_weights.py
```

### 3) Notebooks (`notebooks/`)
- Notebooks document exploration, training, evaluation, and provide transparency.
- Useful for retraining/adaptation.

### 4) Streamlit app (`app/`)
- `app/streamlit_app.py`
- Upload images, experiment with thresholds, visualize predictions.

### 5) Inference CLI (`scripts/`)
- Optionally, use the same model for inference in a script or pipeline.
- Example (basic usage):

  ```python
  from pathlib import Path
  import torch
  from PIL import Image
  from src.inference import build_fasterrcnn_resnet50_fpn, predict

  NUM_CLASSES = 12
  WEIGHTS_PATH = Path("models/FasterRCNN/ResNet50_best_model.pth")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = build_fasterrcnn_resnet50_fpn(NUM_CLASSES, WEIGHTS_PATH, device)

  img = Image.open("your_image.jpg").convert("RGB")
  detections = predict(model, img, device=device, score_threshold=0.35)

  for d in detections:
      print(d)
  ```

---

## Quickstart — Local Setup

### 1) Clone the repository
```bash
git clone https://github.com/MarcSaghiah/xView_detect.git
cd xView_detect
```

### 2) Create and activate a virtual environment

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Obtain model weights
*There are two official ways:*

- **Method 1:** Clone with weights already present in `models/FasterRCNN/ResNet50_best_model.pth`
- **Method 2 (Recommended for cloud/deployment):** Download weights from [GitHub Releases](https://github.com/MarcSaghiah/xView_detect/releases/download/v1.0-baseline/ResNet50_best_model.pth):

```bash
python src/download_weights.py
```

### 5) Run the Streamlit demo
```bash
streamlit run app/streamlit_app.py
```

Go to `http://localhost:8501`, upload an image, and learn from detected objects.

---

## Using inference without Streamlit

Import and use the inference module in your scripts or analysis pipelines as shown above. See `src/inference.py` for API details.

---

## Configuration notes

### Classes and class mapping
- Ensure `NUM_CLASSES` matches your model.  
- `CLASS_NAMES[label_id]` must map correctly to training labels.

Update these in `app/streamlit_app.py` if your dataset has a different mapping.

### Model checkpoint compatibility
- Loader expects a raw `state_dict` (not whole model object).
- If you train a new model, save like:
  ```python
  torch.save(model.state_dict(), "ResNet50_best_model.pth")
  ```

---

## Reproducibility & environment

- Inference works on CPU (slower) and GPU.
- For exact results, use same dataset, preprocessing, and class mapping as documented in notebooks.

---

## FAQ / troubleshooting

- **Weights file not found:**  
  Run `python src/download_weights.py` to fetch official weights, or ensure the file exists at `models/FasterRCNN/ResNet50_best_model.pth`.

- **ImportError (module src):**  
  The app sets up `sys.path` for local/cloud use. If you change structure, ensure `src/__init__.py` exists and launch Streamlit from repo root.

---
