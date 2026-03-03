import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix

def extract_categories_from_coco_json(json_path):
    """
    Extract category names from a COCO format JSON file, ordered by category ID.
    """
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    return [cat["name"] for cat in categories]

def visualize_predictions(image, boxes, labels, scores, class_names, threshold=0.35):
    """
    Visualize predicted bounding boxes (and scores) on a single image.

    Args:
        image (np.ndarray): Image in (H, W, 3) RGB format.
        boxes (np.ndarray): Bounding boxes, shape (N, 4).
        labels (np.ndarray): Predicted label indices.
        scores (np.ndarray): Prediction scores.
        class_names (list): List of class names.
        threshold (float): Confidence threshold for visualization.
    """
    unique_labels = np.unique(labels)
    colors = {label: tuple(np.random.rand(3)) for label in unique_labels}

    fig, ax = plt.subplots(1, figsize=(image.shape[1] / 30, image.shape[0] / 30))
    ax.imshow(image)
    ax.axis('off')
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            color = colors[label]
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                box[0], box[1] - 2, f'{class_names[label]}: {score:.2f}',
                color='black', fontsize=9, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.5, edgecolor='none')
            )
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def test_model(model, test_loader, device, class_names, num_classes=12, num_visualizations=8):
    """
    Iterate over test set, display predictions and gather outputs for further analysis.

    Args:
        model: Trained Faster R-CNN model.
        test_loader: DataLoader for the test set.
        device: The torch device to use.
        class_names: List of class names.
        num_classes: Number of classes including background.
        num_visualizations: Number of random images to visualize.

    Returns:
        List of dictionary with all predictions for further analysis.
    """
    model.to(device)
    model.eval()
    predictions = []
    visualizations_done = 0
    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, (image, pred) in enumerate(zip(images, outputs)):
                prediction_dict = {
                    'boxes': pred['boxes'].cpu().numpy(),
                    'labels': pred['labels'].cpu().numpy(),
                    'scores': pred['scores'].cpu().numpy()
                }
                predictions.append(prediction_dict)
                if visualizations_done < num_visualizations:
                    img_np = (image.cpu().numpy().transpose(1, 2, 0))
                    visualize_predictions(
                        img_np, prediction_dict['boxes'],
                        prediction_dict['labels'],
                        prediction_dict['scores'],
                        class_names
                    )
                    visualizations_done += 1
            if visualizations_done >= num_visualizations:
                break
    print("Testing completed.")
    return predictions

def plot_gt_label_distribution(test_loader, device, class_names, num_classes=12):
    """
    Plot distribution of ground-truth labels (across all target objects) in the test set.

    Args:
        test_loader: DataLoader for the test set.
        device: Torch device.
        class_names: List of class names.
        num_classes: Number of classes (including background).
    """
    gt_label_counts = torch.zeros(num_classes)
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="GT label distribution", leave=False):
            for target in targets:
                gt_labels = target['labels'].cpu()
                for label in gt_labels:
                    gt_label_counts[label] += 1
    gt_label_counts = gt_label_counts.numpy()
    class_names = np.array(class_names)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=gt_label_counts, palette="viridis")
    plt.title('Ground Truth Label Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, cmap='Blues'):
    """
    Visualize the confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (2D numpy array).
        class_names: List of class names.
        cmap: Matplotlib colormap string.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, annot_kws={'size': 12}, square=True
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_normalized_confusion_matrix(cm, class_names, cmap='Blues'):
    """
    Visualize the normalized confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix (2D numpy array).
        class_names: List of class names.
        cmap: Matplotlib colormap string.
    """
    cm_norm = cm.astype('float')
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / (row_sums + 1e-6)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Percent'}, annot_kws={'size': 12}, square=True
    )
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def evaluate_test_set_iou(model, test_loader, device, num_classes=12, iou_threshold=0.3, score_threshold=0.3):
    """
    Compute confusion matrix on test set based on IoU matching between predictions and ground-truth.

    Args:
        model: Trained Faster R-CNN model.
        test_loader: DataLoader for the test set.
        device: Torch device (cuda or cpu).
        num_classes: Number of classes.
        iou_threshold: IoU threshold for positive match.
        score_threshold: Detection score threshold.

    Returns:
        Confusion matrix (numpy array).
    """
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="IoU Testing", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                pred_boxes = output['boxes'].cpu()
                pred_labels = output['labels'].cpu()
                pred_scores = output['scores'].cpu()
                # Filter predictions by detection score
                high_score_idx = pred_scores > score_threshold
                pred_boxes = pred_boxes[high_score_idx]
                pred_labels = pred_labels[high_score_idx]
                # Perform IoU association: assign predicted boxes to gt boxes
                if len(pred_boxes) and len(gt_boxes):
                    ious = box_iou(pred_boxes, gt_boxes)
                    matched_gt = set()
                    for i_pred, box_pred_lbl in enumerate(pred_labels):
                        iou_vals = ious[i_pred]
                        max_iou_idx = torch.argmax(iou_vals)
                        if iou_vals[max_iou_idx] >= iou_threshold:
                            all_true.append(gt_labels[max_iou_idx].item())
                            all_pred.append(box_pred_lbl.item())
                            matched_gt.add(max_iou_idx.item())
                        else:
                            all_true.append(0)
                            all_pred.append(box_pred_lbl.item())
                    for idx in range(len(gt_labels)):
                        if idx not in matched_gt:
                            all_true.append(gt_labels[idx].item())
                            all_pred.append(0)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    return cm

# Entrypoint for testing/visualization utility
if __name__ == "__main__":
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader
    # Import your CustomCOCODataset and collate_fn accordingly
    from faster_rcnn_dataset import CustomCOCODataset, collate_fn

    # Paths (you must adapt these lines depending on your actual paths!)
    data_dir = Path("../data/processed")
    test_json = data_dir / "test.json"
    img_dir = data_dir / "images"
    num_classes = 12

    test_dataset = CustomCOCODataset(str(test_json), img_dir=str(img_dir), aug=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = extract_categories_from_coco_json(test_json)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('../models/FasterRCNN/ResNet50_best_model.pth', map_location=device))
    model.to(device)

    # Plot ground truth label distribution (test set)
    plot_gt_label_distribution(test_loader, device, class_names, num_classes=num_classes)
    # Visualization for a subset of test images and their predictions
    test_model(model, test_loader, device, class_names=class_names, num_classes=num_classes, num_visualizations=5)
    # Quantitative confusion matrix
    cm = evaluate_test_set_iou(model, test_loader, device, num_classes=num_classes)
    plot_confusion_matrix(cm, class_names)
    plot_normalized_confusion_matrix(cm, class_names)