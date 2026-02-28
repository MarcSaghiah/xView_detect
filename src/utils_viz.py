from __future__ import annotations

from typing import List, Sequence

from PIL import Image, ImageDraw

from .inference import Detection


def draw_detections(
    image: Image.Image,
    detections: List[Detection],
    class_names: Sequence[str],
    box_width: int = 3,
) -> Image.Image:
    """
    Draw bounding boxes and labels on a copy of the image.
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        label_name = class_names[det.label] if det.label < len(class_names) else str(det.label)
        text = f"{label_name} {det.score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=box_width)
        draw.text((x1, max(0, y1 - 12)), text, fill="red")

    return img