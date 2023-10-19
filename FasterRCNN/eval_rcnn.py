import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np

def evaluate_faster_rcnn(model, dataloader, device):
    print("[LOG] Evaluating Faster R-CNN model...")
    model.eval()    

    # Initialize evaluation metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through test dataset
    for images, targets in dataloader:
        # Move images and targets to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Make predictions with model
        with torch.inference_mode():
            outputs = model(images)

        # Get predicted bounding boxes and object classes
        pred_boxes = outputs[0]['boxes'].cpu().numpy()
        pred_classes = outputs[0]['labels'].cpu().numpy()

        # Get ground truth bounding boxes and object classes
        gt_boxes = targets[0]['boxes'].cpu().numpy()
        gt_classes = targets[0]['labels'].cpu().numpy()
        
        # print(f"pred:{pred_boxes}")

        # Calculate intersection over union (IoU) between predicted and ground truth boxes
        ious = torchvision.ops.box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()

        # Match predicted and ground truth boxes based on highest IoU
        matched_indices = (-ious).argsort(axis=1)[:, 0]
        matched_ious = ious[np.arange(len(matched_indices)), matched_indices]
        matched_pred_boxes = pred_boxes[matched_indices]
        matched_pred_classes = pred_classes[matched_indices]

        # Calculate true positives, false positives, and false negatives
        true_positives += (matched_ious >= 0.5).sum()
        false_positives += (matched_ious < 0.5).sum()
        false_negatives += (len(gt_boxes) - (matched_ious >= 0.5).sum())

    # Calculate evaluation metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Output evaluation metrics
    print(f"[INFO] Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1_score:.4f}")
    
    return precision, recall, f1_score

    
