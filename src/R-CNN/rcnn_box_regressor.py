import os
import pickle
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class BoundingBoxRegressor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
    def parse_json(self, json_data):
        proposals, ground_truths, categories = [], [], []
        for item in json_data:
            region_bbox = item.get("region_bbox", [])
            gt_bbox = item.get("original_bbox", []) if item.get("original_bbox", []) else region_bbox
            category_id = item.get("category_id", 0)
            if len(region_bbox) != 4 or len(gt_bbox) != 4: continue
            proposals.append(region_bbox)
            ground_truths.append(gt_bbox)
            categories.append(category_id)
        return (np.array(proposals, dtype=float),
                np.array(ground_truths, dtype=float),
                np.array(categories, dtype=int))
    def train(self, json_data, min_samples=10):
        proposals, ground_truths, categories = self.parse_json(json_data)
        unique_categories = np.unique(categories)
        for category in unique_categories:
            mask = categories == category
            cat_proposals = proposals[mask]
            cat_ground_truths = ground_truths[mask]
            if len(cat_proposals) < min_samples:
                continue
            offsets = cat_ground_truths - cat_proposals
            widths = cat_proposals[:,2] - cat_proposals[:,0]
            heights = cat_proposals[:,3] - cat_proposals[:,1]
            offsets[:, [0,2]] /= widths[:, None]
            offsets[:, [1,3]] /= heights[:, None]
            scaler = StandardScaler()
            normalized_proposals = scaler.fit_transform(cat_proposals)
            model = LinearRegression()
            model.fit(normalized_proposals, offsets)
            self.models[category] = model
            self.scalers[category] = scaler
    def predict(self, category_id, region_bbox, image_width=None, image_height=None):
        if category_id not in self.models:
            return region_bbox
        region_bbox = np.array(region_bbox, dtype=float).reshape(1, -1)
        scaler = self.scalers[category_id]
        normalized_bbox = scaler.transform(region_bbox)
        model = self.models[category_id]
        predicted_offsets = model.predict(normalized_bbox)
        width = region_bbox[0][2] - region_bbox[0][0]
        height = region_bbox[0][3] - region_bbox[0][1]
        predicted_offsets[0][[0,2]] *= width
        predicted_offsets[0][[1,3]] *= height
        refined_bbox = (region_bbox + predicted_offsets).flatten()
        if image_width and image_height:
            refined_bbox[0] = max(0, min(refined_bbox[0], image_width))
            refined_bbox[1] = max(0, min(refined_bbox[1], image_height))
            refined_bbox[2] = max(0, min(refined_bbox[2], image_width))
            refined_bbox[3] = max(0, min(refined_bbox[3], image_height))
        if refined_bbox[0] >= refined_bbox[2] or refined_bbox[1] >= refined_bbox[3]:
            return region_bbox.flatten().tolist()
        return refined_bbox.tolist()
    def save_models(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump({'models': self.models, 'scalers': self.scalers, 'metrics': self.metrics}, f)
    def load_models(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scalers = data['scalers']
            self.metrics = data.get('metrics', {})
    def predict_json(self, input_json_path, output_json_path):
        with open(input_json_path, "r") as f:
            data = json.load(f)
        refined_data = []
        for item in tqdm(data, desc="Refining BBoxes"):
            region_bbox = item.get("region_bbox", [])
            category_id = item.get("category_id", 0)
            image_width = item.get("original_size", [0,0])[0]
            image_height = item.get("original_size", [0,0])[1]
            if len(region_bbox) != 4:
                refined_data.append(item)
                continue
            refined_bbox = self.predict(category_id, region_bbox, image_width, image_height)
            item["refined_bbox"] = refined_bbox
            refined_data.append(item)
        with open(output_json_path, "w") as f:
            json.dump(refined_data, f, indent=4)

if __name__ == "__main__":
    regressor = BoundingBoxRegressor()
    train_json_path = Path("../data/processed/active_regions_dataset/train.json")
    train_json = load_json(train_json_path)
    regressor.train(train_json)
    regressor.save_models(Path("../models/ResNet50/regressor_best.pth"))
    test_json_path = Path("../data/processed/active_regions_dataset/test.json")
    regressor.predict_json(
        input_json_path=test_json_path,
        output_json_path=Path("../models/ResNet50/regressed_boxes.json")
    )