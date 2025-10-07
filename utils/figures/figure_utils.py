import torch
import numpy as np
import matplotlib

# Use a non-interactive backend to prevent tkinter errors on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import precision_recall_curve, auc
import os
import warnings


def set_publication_style():
    """Sets a publication-quality style for matplotlib plots."""
    plt.style.use('seaborn-v0_8-paper')
    try:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
            'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9
        })
    except Exception:
        warnings.warn("Times New Roman not found. Falling back to default serif font.")


def save_metrics_plot(metrics_history, log_dir):
    """
    Saves a single consolidated plot with subplots for all training and validation metrics.
    """
    set_publication_style()

    metric_names = list(metrics_history.keys())
    num_metrics = len(metric_names)

    # Determine grid size for subplots (e.g., 3 columns)
    cols = 3
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.5))
    axes = axes.flatten()  # Flatten to make it easier to iterate

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        values = metrics_history[metric_name]

        # Check if there is data to plot
        if not values['train'] or not values['val']:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(f'{metric_name.capitalize()} Over Epochs')
            continue

        epochs = range(1, len(values['train']) + 1)
        
        # Ensure values are CPU-based for matplotlib
        train_values = values['train']
        val_values = values['val']
        
        if torch.is_tensor(train_values[0]):
            train_values = [v.cpu().item() if torch.is_tensor(v) else v for v in train_values]
        if torch.is_tensor(val_values[0]):
            val_values = [v.cpu().item() if torch.is_tensor(v) else v for v in val_values]
            
        ax.plot(epochs, train_values, 'o-', label=f'Train', markersize=4, lw=1.5)
        ax.plot(epochs, val_values, 's--', label=f'Validation', markersize=4, lw=1.5)

        ax.set_title(f'{metric_name.capitalize()} Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide any unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(log_dir, 'all_metrics_plot.png'), dpi=300)
    plt.close()


def visualize_predictions(images, targets, outputs, epoch, writer=None, wandb_run=None, prefix='train', max_samples=4):
    """
    Visualizes predictions in TensorBoard and W&B.
    """
    set_publication_style()
    outputs = torch.sigmoid(outputs)
    preds = (outputs > 0.5).float()

    num_samples = min(max_samples, images.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # Normalize for display
        target = targets[i].cpu().numpy().squeeze()
        pred = preds[i].cpu().detach().numpy().squeeze()

        axes[i][0].imshow(img)
        axes[i][0].set_title('Image')
        axes[i][1].imshow(target, cmap='gray')
        axes[i][1].set_title('Ground Truth')
        axes[i][2].imshow(pred, cmap='gray')
        axes[i][2].set_title('Prediction')
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()

    if writer:
        writer.add_figure(f'{prefix}/predictions', fig, global_step=epoch)
    if wandb_run:
        wandb_run.log({f'{prefix}/predictions': wandb.Image(fig)}, step=epoch)

    plt.close(fig)


def save_predictions_comparison(images, targets, preds, save_path):
    """Saves a side-by-side comparison of image, ground truth, and prediction."""
    set_publication_style()
    preds = (torch.sigmoid(preds) > 0.5).float()
    num_samples = images.size(0)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1: axes = [axes]

    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        target = targets[i].cpu().numpy().squeeze()
        pred = preds[i].cpu().numpy().squeeze()

        axes[i][0].imshow(img)
        axes[i][0].set_title('Image')
        axes[i][1].imshow(target, cmap='gray')
        axes[i][1].set_title('Ground Truth')
        axes[i][2].imshow(pred, cmap='gray')
        axes[i][2].set_title('Prediction')
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curves(model_predictions, save_dir):
    """
    Plots and saves professional, publication-quality precision-recall curves.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    colors = plt.get_cmap('tab10').colors
    line_styles = ['-', '--', '-.', ':']

    for i, (model_name, (preds, targets)) in enumerate(model_predictions.items()):
        y_true = (torch.cat(targets).cpu().numpy().flatten() > 0.5).astype(np.int32)
        y_pred = torch.cat(preds).cpu().numpy().flatten()
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        ax.plot(recall, precision, label=f'{model_name} (AUC={pr_auc:.3f})',
                color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], lw=1.5)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.pdf'), format='pdf')
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300)
    plt.close()


def plot_efficiency_comparison(model_metrics, model_complexities, save_dir):
    """
    Plots and saves a professional comparison of model efficiency.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    models = list(model_metrics.keys())
    ious = [model_metrics[m]['iou'] for m in models]
    params = [model_complexities[m]['params'] for m in models]

    scatter = ax.scatter(params, ious, c=range(len(models)), cmap='viridis', s=100, alpha=0.8, edgecolors='w')

    for i, model_name in enumerate(models):
        ax.text(params[i], ious[i] + 0.002, model_name, fontsize=8, ha='center')

    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Model Efficiency: Performance vs. Parameters')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_comparison.pdf'), format='pdf')
    plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'), dpi=300)
    plt.close()

