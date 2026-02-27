import os
import random
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

train_path = Path("../data/processed/active_regions_dataset/train.json")
val_path = Path("../data/processed/active_regions_dataset/val.json")
test_path = Path("../data/processed/active_regions_dataset/test.json")
alexnet_path = Path("../models/AlexNet/AlexNet_best.pth")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.data = load_json(json_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample["saved_path"]).convert("RGB")
        label = sample["category_id"]
        image = self.transform(image)
        return image, label

def plot_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", color="red", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(net, train_loader, val_loader, criterion, optimizer, device, epochs, path_min_loss, num_classes):
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    for epoch in range(epochs):
        net.train()
        train_loss, correct_train, total_train = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            total_train += labels.size(0)
            correct_train += outputs.argmax(1).eq(labels).sum().item()
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_acc = correct_train / total_train
        net.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                total_val += labels.size(0)
                correct_val += outputs.argmax(1).eq(labels).sum().item()
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_acc = correct_val / total_val
        if avg_val_loss < min_val_loss:
            print(f"Saving best model (val loss improved {min_val_loss:.4f} -> {avg_val_loss:.4f})")
            min_val_loss = avg_val_loss
            torch.save(net.state_dict(), path_min_loss)
        scheduler.step(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Train loss {avg_train_loss:.4f}, acc {train_acc:.2%}; Val loss {avg_val_loss:.4f}, acc {val_acc:.2%}")
    plot_loss(train_losses, val_losses)

def main():
    num_classes = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CustomDataset(train_path)
    val_ds = CustomDataset(val_path)
    test_ds = CustomDataset(test_path)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    in_features = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(in_features, num_classes)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    train_model(net, train_loader, val_loader, criterion, optimizer, device, epochs=10, path_min_loss=alexnet_path, num_classes=num_classes)

if __name__ == "__main__":
    main()