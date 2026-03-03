import os
import json
import random
import torch
from pathlib import Path
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomCOCODataset(Dataset):
    """
    Custom dataset for Faster R-CNN using COCO-style JSON and images.
    """
    def __init__(self, json_file, img_dir, aug=False):
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        self.image_info = {image['id']: image['file_name'] for image in coco_data['images']}
        self.image_annotations = defaultdict(list)
        self.image_bboxes = defaultdict(list)
        self.classes = {int(category['id']): category['name'] for category in coco_data['categories']}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']
            self.image_annotations[image_id].append(annotation['category_id'])
            self.image_bboxes[image_id].append(bbox)
        self.img_dir = img_dir
        self.image_paths = []
        self.image_ids = []
        for image_id, file_name in self.image_info.items():
            if image_id in self.image_annotations:
                img_path = os.path.join(img_dir, file_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.image_ids.append(image_id)
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.aug_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.aug = aug
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_id = self.image_ids[index]
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size
        image_tensor = self.aug_transform(image) if self.aug else self.base_transform(image)
        categories = self.image_annotations[img_id]
        bboxes = self.image_bboxes[img_id]
        scale_x = 224 / original_width
        scale_y = 224 / original_height
        scaled_bboxes = [
            torch.tensor([
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ], dtype=torch.float32)
            for bbox in bboxes
        ]
        target = {
            "boxes": torch.stack(scaled_bboxes),
            "labels": torch.tensor(categories, dtype=torch.int64)
        }
        return image_tensor, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def validate_dataloader(dataloader):
    errors = []
    for batch_idx, (images, targets) in enumerate(dataloader):
        for idx, target in enumerate(targets):
            if target is None:
                errors.append(f"Batch {batch_idx}, Image {idx}: Target is None.")
            elif target["boxes"].numel() == 0 or target["labels"].numel() == 0:
                errors.append(
                    f"Batch {batch_idx}, Image {idx}: Target is empty or missing 'boxes'/'labels'.")
    return errors