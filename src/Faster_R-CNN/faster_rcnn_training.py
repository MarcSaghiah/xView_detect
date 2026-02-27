import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from faster_rcnn_dataset import CustomCOCODataset, collate_fn, validate_dataloader

def train_and_validate(model, train_loader, val_loader, optimizer, device, num_epochs=10, num_classes=12, accumulation_steps=4):
    scaler = GradScaler()
    model.to(device)
    train_losses, val_losses = [], []
    train_precisions, train_recalls = [], []
    val_precisions, val_recalls = [], []
    train_confidences, val_confidences = [], []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss = 0.0
        for i, (images, targets) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast('cuda'):
                loss_dict = model(images, targets)
                if not isinstance(loss_dict, dict):
                    raise ValueError(f"Invalid loss_dict object: {type(loss_dict)}")
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_train_loss += losses.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average train loss: {avg_train_loss:.4f}")
        # Validation (loss only)
        model.train()
        total_val_loss = 0.0
        for i, (images, targets) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad(), autocast('cuda'):
                loss_dict = model(images, targets)
                if not isinstance(loss_dict, dict):
                    raise ValueError(f"Invalid loss_dict object: {type(loss_dict)}")
                losses = sum(loss for loss in loss_dict.values())
                total_val_loss += losses.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        print(f"Model saved: model_epoch_{epoch + 1}.pth")
    return train_losses, val_losses

def plot_metrics(train_losses, val_losses, num_epochs):
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    plt.tight_layout()
    plt.show()

def main():
    import json
    from pathlib import Path

    PROCESSED_DATA_PATH = Path("../data/processed")
    image_folder_path = PROCESSED_DATA_PATH / "images"
    train_path = PROCESSED_DATA_PATH / "train.json"
    val_path = PROCESSED_DATA_PATH / "val.json"
    test_path = PROCESSED_DATA_PATH / "test.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 12

    train_dataset = CustomCOCODataset(str(train_path), img_dir=str(image_folder_path), aug=True)
    val_dataset = CustomCOCODataset(str(val_path), img_dir=str(image_folder_path), aug=False)
    test_dataset = CustomCOCODataset(str(test_path), img_dir=str(image_folder_path), aug=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    # Model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Linear(in_features, num_classes * 4)  # For bbox regression
    )
    for param in model.backbone.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, optimizer, device, num_epochs=10)
    plot_metrics(train_losses, val_losses, num_epochs=10)

if __name__ == "__main__":
    main()