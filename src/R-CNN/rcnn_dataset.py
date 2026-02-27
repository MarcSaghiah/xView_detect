import os
import random
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

active_region_folder_path = Path("../data/processed/active_regions_dataset")
proposal_json_path = active_region_folder_path / "proposals.json"
actproposal_json_path = active_region_folder_path / "active_regions.json"
image_folder_path = Path("../data/processed/images")
train_path = active_region_folder_path / "train.json"
val_path = active_region_folder_path / "val.json"
test_path = active_region_folder_path / "test.json"
coco_json_path = Path("../data/processed/coco_annotations_new.json")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return intersection / float(area1 + area2 - intersection + 1e-6)

def get_adaptive_threshold(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    area_media = {2: 651.14, 11: 6075.12}
    min_area = min(area_media.values())
    max_area = max(area_media.values())
    norm_area = (area - min_area) / (max_area - min_area)
    norm_area = max(0, min(1, norm_area))
    threshold = 0.3 + 0.4 * norm_area
    return threshold

def assign_positive_regions(proposal_json_path, coco_json_path, image_folder_path, output_dir, out_json_path):
    proposals = load_json(proposal_json_path)
    coco = load_json(coco_json_path)
    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        if len(ann['bbox']) == 4:
            annotations_by_image[ann['image_id']].append((ann['bbox'], ann['category_id']))
    os.makedirs(output_dir, exist_ok=True)
    counter = 0
    active_region_data = []
    for img_props in tqdm(proposals, desc="Assign Regions"):
        image_id = img_props['image_id']
        file_name = img_props['file_name']
        gt_data = annotations_by_image.get(image_id, [])
        gt_bboxes = [b[0] for b in gt_data]
        gt_categories = [b[1] for b in gt_data]
        region_bboxes = [p['coordinates'] for p in img_props['proposals']]
        for i, region_bbox in enumerate(region_bboxes):
            best_iou, best_cat = 0.0, 0
            for gt_bbox, gt_cat in zip(gt_bboxes, gt_categories):
                iou = get_iou(region_bbox, gt_bbox)
                adaptive_thresh = get_adaptive_threshold(gt_bbox)
                if iou >= adaptive_thresh and iou > best_iou:
                    best_iou = iou
                    best_cat = gt_cat
            add_bg = (best_cat == 0 and random.random() < 0.1)
            if best_iou > 0 or add_bg:
                img_path = image_folder_path / file_name
                orig_img = cv2.imread(str(img_path))
                x_min, y_min, x_max, y_max = [int(x) for x in region_bbox]
                cropped = orig_img[y_min:y_max, x_min:x_max]
                if cropped.size == 0: continue
                resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
                crop_path = output_dir / f"image_{counter:06d}.jpg"
                cv2.imwrite(str(crop_path), resized)
                active_region_data.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "category_id": best_cat,
                    "proposal_id": i,
                    "region_bbox": region_bbox,
                    "saved_path": str(crop_path)
                })
                counter += 1
    with open(out_json_path, 'w') as f:
        json.dump(active_region_data, f, indent=2)
    print(f"Saved {counter} cropped region proposals and metadata to {out_json_path}")

def split_dataset(active_region_json_path, train_path, val_path, test_path):
    data = load_json(active_region_json_path)
    df = pd.DataFrame(data)
    df['category_id_strat'] = df['category_id'].replace({7: 6})
    X = df.index
    y = df['category_id_strat']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED)
    train_data = df.loc[X_train].drop(columns=["category_id_strat"])
    val_data = df.loc[X_val].drop(columns=["category_id_strat"])
    test_data = df.loc[X_test].drop(columns=["category_id_strat"])
    train_data.to_json(train_path, orient="records")
    val_data.to_json(val_path, orient="records")
    test_data.to_json(test_path, orient="records")
    print("Splitting completed. Files saved: train.json, val.json, test.json.")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    assign_positive_regions(
        proposal_json_path=proposal_json_path,
        coco_json_path=coco_json_path,
        image_folder_path=image_folder_path,
        output_dir=active_region_folder_path / "proposals",
        out_json_path=actproposal_json_path
    )
    split_dataset(
        actproposal_json_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path
    )