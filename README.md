# xView_detect

**Worked on:** June 2025 → August 2025

**Stack:** _Python, PyTorch, torchvision, Faster R-CNN, Streamlit, PIL_

**Satellite imagery object detection on the xView dataset using Faster R-CNN, packaged with reusable inference code and a Streamlit demo.**

---

## 🚀 Overview

xView_detect is an end-to-end **object detection** project for **satellite imagery** using the **xView dataset** and a **Faster R-CNN (ResNet50-FPN)** detector. It includes:

- A reusable **inference module** (`src/`) to load the model and run predictions on new images
- A **trained checkpoint** distributed in `models/` and via **GitHub Releases**
- A **Streamlit demo app** (`app/`) to upload an image and visualize detections
- Notebooks documenting exploration, preparation, training, and baseline evaluation

> Note: The model provided is a baseline and can be improved (augmentation, tuning, metrics, stronger architectures).

---

## 📂 Project Structure

```
xView_detect/
│
├── app/                    # Streamlit demo (upload image, run detection)
│   └── streamlit_app.py
│
├── src/                    # Inference + utilities
│   ├── inference.py
│   ├── utils_viz.py
│   └── download_weights.py
│
├── models/                 # Model weights (or downloaded here)
│   └── FasterRCNN/
│       └── ResNet50_best_model.pth
│
├── notebooks/              # Exploration / training / experiments
│
├── results/                # (Optional) outputs, figures, logs
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MarcSaghiah/xView_detect.git
   cd xView_detect
   ```

2. Create and activate a virtual environment:

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

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🔄 Inference

You can use the inference code directly from Python (without Streamlit).

Example:

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

## 📊 Streamlit Demo

Launch the app:

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` and:
- Upload a satellite image
- Adjust the score threshold
- Visualize bounding boxes and inspect detections

---

## 🧠 Model Weights

The baseline checkpoint is distributed in two ways:

- **Committed directly** in the repository:  
  `models/FasterRCNN/ResNet50_best_model.pth`

- **Available via GitHub Releases** (useful for cloud/deployment):  
  Download the release asset and place it under `models/FasterRCNN/`

Auto-download (recommended):

```bash
python src/download_weights.py
```

---

## 🗂️ Data (xView)

This project is built around the **xView** dataset (satellite images + bounding box annotations across multiple object categories).

Because xView is large, this repository focuses on the **modeling and inference pipeline** and assumes you have access to the dataset separately if you want to retrain.

---

## 🏆 Results & Next Steps

### Highlights
- End-to-end pipeline (load model → predict → visualize)
- Packaged for reuse (module + demo)
- Clean structure with clear separation between app, inference, and assets

### Limitations (baseline)
- Quality varies across classes and scenes
- Small objects, class imbalance, and dense backgrounds remain challenging
- COCO-style metrics (mAP) are not fully reported yet (can be added)

### Potential improvements
- Stronger augmentation (random crop, scale jitter)
- Hyperparameter tuning / anchor optimization
- Longer training + better validation
- Add evaluation scripts (COCO mAP)
- Explore alternatives (RetinaNet, Mask R-CNN)

---

## 📄 License

MIT

---

**Author: Marc Saghiah**