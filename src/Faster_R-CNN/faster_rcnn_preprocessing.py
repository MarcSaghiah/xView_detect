import os
import random
import json
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def load_json(file_path):
    """Load a JSON file from disk."""
    with open(file_path, 'r') as f:
        return json.load(f)

def process_custom_coco_json(input_path, output_path, image_subsample_ratio=0.1):
    """
    Correct COCO categories, bounding boxes; subsample images.
    Args:
        input_path (str or Path)
        output_path (str or Path)
        image_subsample_ratio (float): fraction to keep
    """
    data = load_json(input_path)
    # Correct categories - Aircraft id 0 → 11, add background
    categories = []
    for cat in data.get('categories', []):
        for id_str, name in cat.items():
            categories.append({'id': int(id_str), 'name': name})
    for cat in categories:
        if cat['id'] == 0 and cat['name'] == 'Aircraft':
            cat['id'] = 11
    if not any(cat['id'] == 0 for cat in categories):
        categories.append({'id': 0, 'name': 'background'})
    # Clean/convert bboxes, filter invalid
    image_annots = {}
    valid_annots = []
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        image_annots.setdefault(img_id, []).append(ann)
    for ann in data.get('annotations', []):
        if ann['category_id'] == 0:
            ann['category_id'] = 11
        bbox = ann['bbox']
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        x, y, w, h = bbox
        if w <= 10 or h <= 10 or w < 0 or h < 0:
            continue
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        if xmin >= xmax or ymin >= ymax: continue
        ann['bbox'] = [xmin, ymin, xmax, ymax]
        valid_annots.append(ann)
    # Subsample images
    images = data.get('images', [])
    selected_images = random.sample(images, int(len(images) * image_subsample_ratio))
    selected_ids = {img['id'] for img in selected_images}
    # Filter annotations for sampled images
    data['images'] = selected_images
    data['annotations'] = [ann for ann in valid_annots if ann['image_id'] in selected_ids]
    data['categories'] = categories
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Processed and subsampled COCO JSON written to {output_path}")

def count_bboxes_per_category(json_path):
    """Count bounding boxes for each category in COCO."""
    data = load_json(json_path)
    category_mapping = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    bbox_counts = defaultdict(int)
    for annotation in data.get('annotations', []):
        category_id = annotation['category_id']
        bbox_counts[category_id] += 1
    return {category_mapping[cat_id]: count for cat_id, count in bbox_counts.items()}

def undersample_class(json_data_path, dataset_out, target_class=6, target_percentage=0.9):
    """Remove images with only/mostly one target class."""
    json_data = load_json(json_data_path)
    image_class_counts = Counter()
    total_class_counts = Counter()
    for annotation in json_data["annotations"]:
        image_class_counts[(annotation["image_id"], annotation["category_id"])] += 1
        total_class_counts[annotation["image_id"]] += 1
    # Find images to remove
    images_with_majority_target = {
        image_id for (image_id, category_id), count in image_class_counts.items()
        if category_id == target_class and count / total_class_counts[image_id] >= target_percentage
    }
    images_with_only_target = {
        image_id for (image_id, category_id), count in image_class_counts.items()
        if category_id == target_class and total_class_counts[image_id] == count
    }
    images_to_remove = images_with_majority_target | images_with_only_target
    # Filter
    remaining_images = [img for img in json_data["images"] if img["id"] not in images_to_remove]
    remaining_annotations = [ann for ann in json_data["annotations"] if ann["image_id"] not in images_to_remove]
    json_data["images"] = remaining_images
    json_data["annotations"] = remaining_annotations
    with open(dataset_out, "w") as file:
        json.dump(json_data, file, indent=4)
    print(f"Undersampling completed.")

def upsample_classes(json_data_path, dataset_out, classes_to_upsample, value):
    """Duplicate images & annotations for rare classes."""
    json_data = load_json(json_data_path)
    class_bbox_counts = Counter()
    image_annotations = {}
    for annotation in json_data["annotations"]:
        if annotation["category_id"] in classes_to_upsample:
            class_bbox_counts[annotation["category_id"]] += 1
        if annotation["image_id"] not in image_annotations:
            image_annotations[annotation["image_id"]] = []
        image_annotations[annotation["image_id"]].append(annotation)
    # Sort by priority
    class_images = {class_id: [] for class_id in classes_to_upsample}
    for img in json_data["images"]:
        image_id = img["id"]
        annotations = image_annotations.get(image_id, [])
        for ann in annotations:
            if ann["category_id"] in classes_to_upsample:
                class_images[ann["category_id"]].append((img, annotations))
    # Duplicate
    new_images, new_annotations = [], []
    image_id_offset = max(img["id"] for img in json_data["images"]) + 1
    annotation_id_offset = max(ann["id"] for ann in json_data["annotations"]) + 1
    for class_id in classes_to_upsample:
        class_data = class_images[class_id]
        idx = 0
        current_count = class_bbox_counts[class_id]
        target_count = value * class_bbox_counts[class_id]
        while current_count < target_count and class_data:
            img, annotations = class_data[idx % len(class_data)]
            new_img = dict(img)
            new_img["id"] = image_id_offset
            for ann in annotations:
                if ann["category_id"] == class_id:
                    new_ann = dict(ann)
                    new_ann["id"] = annotation_id_offset
                    new_ann["image_id"] = image_id_offset
                    new_annotations.append(new_ann)
                    annotation_id_offset += 1
            new_images.append(new_img)
            image_id_offset += 1
            current_count += len([ann for ann in annotations if ann["category_id"] == class_id])
            idx += 1
    json_data["images"].extend(new_images)
    json_data["annotations"].extend(new_annotations)
    with open(dataset_out, "w") as file:
        json.dump(json_data, file, indent=4)
    print(f"Upsampling completed.")

def split(json_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split COCO JSON into train/val/test sets with category stats."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']
    random.shuffle(images)
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = int(total_images * (train_ratio + val_ratio))
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    train_image_ids = {image['id'] for image in train_images}
    val_image_ids = {image['id'] for image in val_images}
    test_image_ids = {image['id'] for image in test_images}
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]
    def save_json(d, filename):
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)
    save_json({'images': train_images, 'annotations': train_annotations, 'categories': data['categories']}, 'train.json')
    save_json({'images': val_images, 'annotations': val_annotations, 'categories': data['categories']}, 'val.json')
    save_json({'images': test_images, 'annotations': test_annotations, 'categories': data['categories']}, 'test.json')
    print(f"Split complete: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test.")

if __name__ == "__main__":
    # Example usage:
    process_custom_coco_json('coco_annotations_new.json', 'mod_coco_annotations_new.json', image_subsample_ratio=0.1)
    undersample_class('mod_coco_annotations_new.json', 'mod_coco_annotations_new.json', target_class=6)
    upsample_classes('mod_coco_annotations_new.json', 'mod_coco_annotations_new.json', classes_to_upsample=[2,3,4,5,7,8,9,10,11], value=3)
    upsample_classes('mod_coco_annotations_new.json', 'mod_coco_annotations_new.json', classes_to_upsample=[7,8,10,11], value=10)
    upsample_classes('mod_coco_annotations_new.json', 'mod_coco_annotations_new.json', classes_to_upsample=[1], value=1.2)
    split('mod_coco_annotations_new.json')