import os
import random
import json
from pathlib import Path
from itertools import islice
import cv2
from tqdm import tqdm
import selectivesearch

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PROCESSED_DATA_PATH = Path("../data/processed")
image_folder_path = PROCESSED_DATA_PATH / "images"
coco_json_path = PROCESSED_DATA_PATH / "coco_annotations_new.json"
active_region_folder_path = PROCESSED_DATA_PATH / "active_regions_dataset"
proposal_json_path = active_region_folder_path / "proposals.json"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_and_process_proposals(image, img_width, img_height):
    _, regions = selectivesearch.selective_search(image, scale=200, sigma=0.5, min_size=10)
    proposals = []
    for region in regions:
        x, y, w, h = region['rect']
        area = w * h
        if w >= 10 and h >= 10 and 10 <= area <= 0.8 * (img_width * img_height):
            proposals.append([x, y, x + w, y + h])
    return proposals

def process_single_image(image_data, images_folder):
    img_id = image_data['id']
    img_name = image_data['file_name']
    img_path = images_folder / img_name
    if not img_path.exists():
        raise ValueError(f"Image not found: {img_path}")
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    proposals = generate_and_process_proposals(image, w, h)
    return {
        "image_id": img_id,
        "file_name": img_name,
        "original_size": [w, h],
        "proposals": [{"proposal_id": i, "coordinates": p} for i, p in enumerate(proposals)]
    }

def batch(iterable, n=1):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def generate_dataset_proposals(coco_json, image_folder_path, output_json, batch_size=500):
    coco_data = load_json(coco_json)
    images = coco_data['images']
    results = []
    total_batches = len(images) // batch_size + (1 if len(images) % batch_size else 0)
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for image_batch in batch(images, batch_size):
            batch_results = []
            for img in image_batch:
                try:
                    batch_results.append(process_single_image(img, image_folder_path))
                except Exception as e:
                    print(f"Error processing {img['file_name']}: {e}")
            results.extend(batch_results)
            pbar.update(1)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Created JSON file with region proposals: {output_json}")

if __name__ == "__main__":
    generate_dataset_proposals(
        coco_json=coco_json_path,
        image_folder_path=image_folder_path,
        output_json=proposal_json_path,
        batch_size=500
    )