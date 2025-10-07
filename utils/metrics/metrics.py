import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

# Helper function to convert logits/probabilities to binary prediction (0 or 1)
def get_binary_prediction(SR, threshold=0.5):
    """
    Converts model output (assumed to be logits or probabilities) to binary mask.
    If SR is logits (unbounded), sigmoid must be applied first.
    We assume the input SR here is the raw model output (logits).
    """
    if SR.min() < 0 or SR.max() > 1: # Check if it looks like logits
        SR = torch.sigmoid(SR)
    
    # Binary prediction
    return (SR > threshold).float()

def get_accuracy(SR, GT):
    corr = torch.sum(SR == GT)
    tensor_size = SR.numel()
    acc = float(corr) / float(tensor_size)
    return acc

def get_sensitivity(SR, GT): # Recall
    TP = ((SR == 1) & (GT == 1)).sum()
    FN = ((SR == 0) & (GT == 1)).sum()
    return float(TP) / (float(TP + FN) + 1e-6)

def get_specificity(SR, GT):
    TN = ((SR == 0) & (GT == 0)).sum()
    FP = ((SR == 1) & (GT == 0)).sum()
    return float(TN) / (float(TN + FP) + 1e-6)

def get_precision(SR, GT):
    TP = ((SR == 1) & (GT == 1)).sum()
    FP = ((SR == 1) & (GT == 0)).sum()
    return float(TP) / (float(TP + FP) + 1e-6)

def get_miou(SR, GT):
    intersection_crack = ((SR == 1) & (GT == 1)).sum()
    union_crack = ((SR == 1) | (GT == 1)).sum()
    iou_crack = float(intersection_crack) / (float(union_crack) + 1e-6)

    # Intersection Over Union (IoU) is often computed only for the positive class in binary segmentation
    return iou_crack


def evaluation_metrics(outputs, targets, threshold=0.5):
    # CRITICAL: Calculate binary prediction mask using sigmoid and a threshold
    SR = get_binary_prediction(outputs, threshold=threshold) 
    
    # Flatten tensors for scikit-learn metrics
    target_flat = targets.cpu().numpy().astype(int).flatten()
    preds_flat = SR.cpu().numpy().astype(int).flatten()
    
    # --- Basic Metrics ---
    # Dice Coefficient (F1-Score)
    TP = ((SR == 1) & (targets == 1)).sum()
    FP = ((SR == 1) & (targets == 0)).sum()
    FN = ((SR == 0) & (targets == 1)).sum()
    dice = (2. * TP) / (2. * TP + FP + FN + 1e-6)
    
    # IoU is the same as miou in this context if background is ignored or computed as (TP / (TP+FP+FN))
    iou = get_miou(SR, targets)

    # Other computed metrics
    recall = get_sensitivity(SR, targets)
    precision = get_precision(SR, targets)
    specificity = get_specificity(SR, targets)
    accuracy = get_accuracy(SR, targets)
    f1 = 2 * recall * precision / (recall + precision + 1e-6)
    
    # miou is already calculated above as iou_crack, but we keep the name for consistency
    miou = iou
    
    # --- Advanced Metrics ---
    # Handle case where one class is completely missing (numpy must be stable)
    try:
        mcc = matthews_corrcoef(target_flat, preds_flat)
    except ValueError:
        # This happens if one class is missing entirely (e.g., all 0s)
        mcc = 0.0

    try:
        balanced_accuracy = balanced_accuracy_score(target_flat, preds_flat)
    except ValueError:
        balanced_accuracy = accuracy.item() if torch.is_tensor(accuracy) else accuracy
    
    return iou, dice, recall, precision, f1, specificity, accuracy, miou, mcc, balanced_accuracy

def calculate_f1_for_threshold(pred, target, threshold):
    # Note: pred is assumed to be raw logits/model output
    pred_binary = get_binary_prediction(pred, threshold=threshold)
    target_float = target.float()
    
    TP = ((pred_binary == 1) & (target_float == 1)).sum()
    FP = ((pred_binary == 1) & (target_float == 0)).sum()
    FN = ((pred_binary == 0) & (target_float == 1)).sum()
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1.item()

def calculate_ods(predictions, targets):
    thresholds = np.linspace(0.01, 0.99, 99)
    avg_f1_scores = []
    # Ensure targets are on CPU for numpy operations in helper function
    targets_cpu = [t.cpu() for t in targets] 
    
    for threshold in thresholds:
        f1_scores = [calculate_f1_for_threshold(p, t, threshold) for p, t in zip(predictions, targets_cpu)]
        avg_f1_scores.append(np.mean(f1_scores))
    
    if not avg_f1_scores:
        return 0.0 # Return 0 if no scores calculated
        
    return np.max(avg_f1_scores)
    
# OIS calculation is not modified but depends on the new evaluation_metrics thresholding being correct.
# Assuming calculate_ois implementation is correct elsewhere, we only show the modified file content.
