from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class Detection:
    """
    Single object detection result.
    """
    box_xyxy: Tuple[float, float, float, float]
    label: int
    score: float


def build_fasterrcnn_resnet50_fpn(num_classes: int, weights_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Build a Faster R-CNN ResNet50-FPN model and load trained weights.

    Args:
        num_classes: Number of classes including background (class 0).
        weights_path: Path to a .pth file containing a state_dict.
        device: Torch device.

    Returns:
        A ready-to-run model in eval mode.
    """
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    image_pil: Image.Image,
    device: torch.device,
    score_threshold: float = 0.35,
) -> List[Detection]:
    """
    Run Faster R-CNN inference on a PIL image and return detections above score_threshold.

    Args:
        model: Faster R-CNN model.
        image_pil: Input image as PIL.
        device: Torch device.
        score_threshold: Minimum confidence score to keep a detection.

    Returns:
        List of detections (boxes in absolute image pixel coordinates).
    """
    image_rgb = image_pil.convert("RGB")
    image_tensor = F.to_tensor(image_rgb).to(device)

    output = model([image_tensor])[0]
    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    detections: List[Detection] = []
    for box, label, score in zip(boxes, labels, scores):
        s = float(score.item())
        if s < score_threshold:
            continue
        x1, y1, x2, y2 = map(float, box.tolist())
        detections.append(Detection((x1, y1, x2, y2), int(label.item()), s))

    return detections