import os
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
import argparse
import numpy as np
from builders.model_builder import get_model
import utils.losses.losses as losses
from trainer import train_one_epoch, validate_one_epoch
from utils.util import seed_torch
from utils.figures.figure_utils import save_metrics_plot, analyze_train_val_discrepancies
from datasets.crack_segmentation import CrackDataSets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


def main():
    parser = argparse.ArgumentParser(description="Training script for crack segmentation")
    parser.add_argument('--dataset', type=str, required=True, choices=config.DATASETS.keys(),
                        help='Dataset to train on')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='List of models to train. Defaults to the list in config.py. Use "all" to train all models.')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                        help='List of GPU IDs to use. Defaults to the one in config.py')
    args = parser.parse_args()

    seed_torch(config.GENERAL.SEED)

    # --- GPU Setup ---
    gpu_ids = args.gpu_ids if args.gpu_ids is not None else config.GENERAL.GPU_IDS
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu")

    # --- Model Selection ---
    if args.models is None:
        models_to_run = config.DEFAULT_MODELS_TO_TRAIN
    elif 'all' in args.models:
        models_to_run = config.AVAILABLE_MODELS
    else:
        models_to_run = [m for m in args.models if m in config.AVAILABLE_MODELS]

    if not models_to_run:
        logging.error("No valid models specified to run. Please check the model names in config.py.")
        return

    dataset_config = config.DATASETS[args.dataset]

    for model_name in models_to_run:
        print(f"\n{'=' * 20} Training {model_name} on {args.dataset} using {len(gpu_ids)} GPU(s) {'=' * 20}")

        # --- Setup logging and trackers ---
        log_dir = os.path.join(config.PATHS.LOGS_DIR, args.dataset, model_name)
        checkpoint_dir = os.path.join(config.PATHS.CHECKPOINT_DIR, args.dataset, model_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)

        wandb_run = None
        if config.TRACKING.WANDB_ENABLED:
            def class_to_dict(c):
                return {k: v for k, v in vars(c).items() if not k.startswith('__')}

            wandb_config = {
                "GENERAL": class_to_dict(config.GENERAL),
                "TRAIN": class_to_dict(config.TRAIN),
                "dataset": args.dataset,
                "model": model_name
            }

            wandb_run = wandb.init(
                project=config.TRACKING.WANDB_PROJECT_NAME,
                name=f"{model_name}_{args.dataset}",
                config=wandb_config
            )

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # --- DataLoaders ---
        if getattr(config, 'NORMALIZATION', None) and config.NORMALIZATION.MODE == 'simple':
            # Legacy normalization that many earlier repos used: scale to [0,1] then (x-0.5)/0.5
            normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0)
        else:
            normalize = A.Normalize(mean=dataset_config["mean"], std=dataset_config["std"], max_pixel_value=255.0)

        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.2),
            A.Resize(height=config.GENERAL.IMG_SIZE, width=config.GENERAL.IMG_SIZE),
            normalize,
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Resize(height=config.GENERAL.IMG_SIZE, width=config.GENERAL.IMG_SIZE),
            normalize,
            ToTensorV2(),
        ])

        # DataLoaders section
        train_dataset = CrackDataSets(
            base_dir=dataset_config["base_dir"],
            split="train",
            image_file=dataset_config["train_image_file"],
            mask_file=dataset_config["train_mask_file"],
            transform=train_transform,
        )
        val_dataset = CrackDataSets(
            base_dir=dataset_config["base_dir"],
            split="val",
            image_file=dataset_config["val_image_file"],
            mask_file=dataset_config["val_mask_file"],
            transform=val_transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE * len(gpu_ids), shuffle=True,
                                  num_workers=config.GENERAL.NUM_WORKERS, pin_memory=config.GENERAL.PIN_MEMORY)
        val_loader = DataLoader(val_dataset, batch_size=config.TEST.TEST_BATCH_SIZE * len(gpu_ids), shuffle=False,
                                num_workers=config.GENERAL.NUM_WORKERS, pin_memory=config.GENERAL.PIN_MEMORY)

        # --- Model, Optimizer, Scheduler, Loss ---
        model_args = argparse.Namespace(model=model_name, num_classes=config.GENERAL.NUM_CLASSES)
        model = get_model(model_args)

        if len(gpu_ids) > 1 and torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=gpu_ids)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.BASE_LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.EPOCHS, eta_min=1e-6)

        # Prefer precision-oriented Tversky configuration to lower recall
        if config.TRAIN.LOSS_FN == "TverskyLoss":
            criterion = losses.TverskyLoss(
                alpha=config.LOSS.TVERSKY_ALPHA,
                beta=config.LOSS.TVERSKY_BETA
            ).to(device)
        else:
            criterion = getattr(losses, config.TRAIN.LOSS_FN)().to(device)

        # --- Training Loop ---
        best_iou = 0
        metrics_history = {
            'loss': {'train': [], 'val': []}, 'iou': {'train': [], 'val': []}, 'dice': {'train': [], 'val': []},
            'recall': {'train': [], 'val': []}, 'precision': {'train': [], 'val': []}, 'f1': {'train': [], 'val': []},
            'specificity': {'train': [], 'val': []}, 'accuracy': {'train': [], 'val': []},
            'miou': {'train': [], 'val': []}, 'mcc': {'train': [], 'val': []},
            'balanced_accuracy': {'train': [], 'val': []},
            'ods': {'train': [], 'val': []}, 'ois': {'train': [], 'val': []}, 'best_threshold': {'train': [], 'val': []}
        }

        for epoch in range(config.TRAIN.EPOCHS):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, writer,
                                            wandb_run)

            if np.isnan(train_metrics['loss']):
                logging.error(f"Training stopped for {model_name} on {args.dataset} due to NaN loss.")
                break

            val_metrics = validate_one_epoch(model, val_loader, criterion, device, epoch + 1, writer, wandb_run)

            # **FIX:** Move scheduler step to after the validation loop for epoch-level schedulers.
            scheduler.step()

            for metric in train_metrics:
                # Convert tensor values to CPU and then to Python scalars
                value = train_metrics[metric]
                if torch.is_tensor(value):
                    value = value.cpu().item()
                metrics_history[metric]['train'].append(value)
            for metric in val_metrics:
                value = val_metrics[metric]
                if torch.is_tensor(value):
                    value = value.cpu().item()
                metrics_history[metric.replace('val_', '')]['val'].append(value)

            # Print concise epoch summary to terminal
            logging.info(
                f"Epoch {epoch + 1}/{config.TRAIN.EPOCHS} | "
                f"Train: loss={train_metrics['loss']:.4f} iou={train_metrics['iou']:.4f} dice={train_metrics['dice']:.4f} "
                f"prec={train_metrics['precision']:.4f} rec={train_metrics['recall']:.4f} f1={train_metrics['f1']:.4f} | "
                f"Val: loss={val_metrics['val_loss']:.4f} iou={val_metrics['val_iou']:.4f} dice={val_metrics['val_dice']:.4f} "
                f"prec={val_metrics['val_precision']:.4f} rec={val_metrics['val_recall']:.4f} f1={val_metrics['val_f1']:.4f} "
                f"ods={val_metrics.get('val_ods', float('nan')):.4f} ois={val_metrics.get('val_ois', float('nan')):.4f} "
                f"thr={val_metrics.get('val_best_threshold', config.METRICS.THRESHOLD):.3f}"
            )

            if val_metrics['val_iou'] > best_iou:
                best_iou = val_metrics['val_iou']
                save_path = os.path.join(checkpoint_dir, 'best_model.pth')
                state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({'model_state_dict': state_to_save}, save_path)
                # Save tuned threshold bound to this best model
                threshold_path = os.path.join(checkpoint_dir, 'best_threshold.txt')
                try:
                    with open(threshold_path, 'w') as f:
                        f.write(str(val_metrics.get('val_best_threshold', config.METRICS.THRESHOLD)))
                    logging.info(f"Saved best threshold {val_metrics.get('val_best_threshold', config.METRICS.THRESHOLD):.3f} to {threshold_path}")
                except Exception as e:
                    logging.warning(f"Could not save best threshold. Error: {e}")
                logging.info(f"Epoch {epoch + 1}: New best model saved with IoU: {best_iou:.4f} at {save_path}")

            save_metrics_plot(metrics_history, log_dir)
            try:
                analyze_train_val_discrepancies(metrics_history, log_dir)
            except Exception as e:
                logging.warning(f"Discrepancy analysis failed: {e}")
            pd.DataFrame.from_dict({(i, j): metrics_history[i][j]
                                    for i in metrics_history.keys()
                                    for j in metrics_history[i].keys()},
                                   orient='index').to_csv(os.path.join(log_dir, 'metrics_history.csv'))

        writer.close()
        if wandb_run:
            wandb_run.finish()

        logging.info(f"Training finished for {model_name} on {args.dataset}")


if __name__ == "__main__":
    main()

