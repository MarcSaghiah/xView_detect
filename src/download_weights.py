import requests
from pathlib import Path

MODEL_URL = "https://github.com/MarcSaghiah/xView_detect/releases/download/v1.0-baseline/ResNet50_best_model.pth"
MODEL_PATH = Path("models/FasterRCNN/ResNet50_best_model.pth")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def download():
    print(f"Downloading model weights from:\n  {MODEL_URL}")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to: {MODEL_PATH.absolute()}")

if __name__ == "__main__":
    if MODEL_PATH.exists():
        print(f"Model already exists at: {MODEL_PATH.absolute()}")
    else:
        download()