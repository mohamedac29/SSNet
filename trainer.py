"""
Author: Mohammed Abdelrahim
23 March 2026
"""
# import section
import logging
import torch
from tqdm import tqdm
import wandb
import numpy as np
from utils.metrics.metrics import evaluation_metrics, calculate_ods, calculate_ois, find_best_threshold
from utils.util import AverageMeter
from utils.figures.figure_utils import visualize_predictions
import config


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None, wandb_run=None):
    """
    Trains the model for one epoch with gradient clipping for stability.
    """
    model.train()
    avg_meters = {
        'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(),
        'recall': AverageMeter(), 'precision': AverageMeter(), 'f1': AverageMeter(),
        'specificity': AverageMeter(), 'accuracy': AverageMeter(), 'miou': AverageMeter(),
        'mcc': AverageMeter(), 'balanced_accuracy': AverageMeter()
    }

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]', leave=False)
    last_batch_images, last_batch_targets, last_batch_outputs = None, None, None

    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = batch['label'].to(device, dtype=torch.float)
        last_batch_images, last_batch_targets = images, targets

        outputs = model(images)
        last_batch_outputs = outputs
        loss = criterion(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN or Inf loss detected at epoch {epoch}, batch {i}. Skipping update.")
            continue

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics = evaluation_metrics(outputs, targets, threshold=config.METRICS.THRESHOLD)
        (iou, dice, recall, precision, f1, specificity, accuracy, miou, mcc, balanced_acc) = metrics

        for name, val in zip(avg_meters.keys(),
                             [loss.item(), iou, dice, recall, precision, f1, specificity, accuracy, miou, mcc,
                              balanced_acc]):
            avg_meters[name].update(val, images.size(0))

        pbar.set_postfix({k: f'{v.avg:.4f}' for k, v in avg_meters.items()})

    # Log metrics and visualizations
    if writer or wandb_run:
        log_data = {f'train/{k}': v.avg for k, v in avg_meters.items()}
        if writer:
            for k, v in log_data.items(): writer.add_scalar(k, v, epoch)
        if wandb_run:
            wandb_run.log(log_data, step=epoch)

        if last_batch_images is not None:
            visualize_predictions(
                images=last_batch_images, targets=last_batch_targets, outputs=last_batch_outputs,
                epoch=epoch, writer=writer, wandb_run=wandb_run, prefix='train'
            )

    return {k: v.avg for k, v in avg_meters.items()}


# Function: validate_one_epoch
@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, epoch, writer=None, wandb_run=None):
    """
    Validates the model for one epoch.
    """
    model.eval()
    avg_meters = {
        'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(),
        'recall': AverageMeter(), 'precision': AverageMeter(), 'f1': AverageMeter(),
        'specificity': AverageMeter(), 'accuracy': AverageMeter(), 'miou': AverageMeter(),
        'mcc': AverageMeter(), 'balanced_accuracy': AverageMeter()
    }

    first_batch_images, first_batch_targets, first_batch_outputs = None, None, None
    all_preds, all_targets = [], []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [VALID]', leave=False)
    for i, batch in enumerate(pbar):
        images, targets = batch['image'].to(device), batch['label'].to(device, dtype=torch.float)
        outputs = model(images)
        loss = criterion(outputs, targets)

        if i == 0:
            first_batch_images, first_batch_targets, first_batch_outputs = images, targets, outputs

        metrics = evaluation_metrics(outputs, targets, threshold=config.METRICS.THRESHOLD)
        (iou, dice, recall, precision, f1, specificity, accuracy, miou, mcc, balanced_acc) = metrics

        for name, val in zip(avg_meters.keys(),
                             [loss.item(), iou, dice, recall, precision, f1, specificity, accuracy, miou, mcc,
                              balanced_acc]):
            avg_meters[name].update(val, images.size(0))

        # Collect predictions/targets for ODS/OIS and threshold tuning
        # Always collect probabilities for ODS/OIS
        all_preds.extend(p.cpu() for p in torch.sigmoid(outputs))
        all_targets.extend(t.cpu() for t in targets)

        pbar.set_postfix({f'val_{k}': f'{v.avg:.4f}' for k, v in avg_meters.items()})

    # Compute ODS/OIS and best threshold on validation set
    val_ods = calculate_ods(all_preds, all_targets)
    val_ois = calculate_ois(all_preds, all_targets)
    best_threshold, _ = find_best_threshold(all_preds, all_targets)

    # Log metrics and visualizations
    if writer or wandb_run:
        log_data = {f'val/{k}': v.avg for k, v in avg_meters.items()}
        log_data['val/ods'] = val_ods
        log_data['val/ois'] = val_ois
        log_data['val/best_threshold'] = best_threshold

        if writer:
            for k, v in log_data.items(): writer.add_scalar(k, v, epoch)
        if wandb_run:
            wandb_run.log(log_data, step=epoch)

        if first_batch_images is not None:
            visualize_predictions(
                images=first_batch_images, targets=first_batch_targets, outputs=first_batch_outputs,
                epoch=epoch, writer=writer, wandb_run=wandb_run, prefix='val', threshold=best_threshold
            )

    result = {f'val_{k}': v.avg for k, v in avg_meters.items()}
    result['val_ods'] = val_ods
    result['val_ois'] = val_ois
    result['val_best_threshold'] = best_threshold
    return result

