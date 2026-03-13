import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, auc

def save_masks_with_legend(models_dict, test_loader, device, num_classes, save_dir):
    it = iter(test_loader)
    images, masks = next(it)
    idx = 0
    img_tensor = images[idx].to(device)
    gt_mask = masks[idx].cpu().numpy()

    img_disp = np.clip(img_tensor.cpu().numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

    plt.figure(figsize=(25, 10))
    plt.subplot(1, len(models_dict)+2, 1); plt.imshow(img_disp); plt.title("Original"); plt.axis('off')
    plt.subplot(1, len(models_dict)+2, 2); plt.imshow(gt_mask, cmap='jet', vmin=0, vmax=num_classes-1); plt.title("GT"); plt.axis('off')

    legend_elements = [
        Patch(facecolor='yellow', label='Correct (TP)'),
        Patch(facecolor='magenta', label='Missed (FN)'),
        Patch(facecolor='green', label='False Positive (FP)')
    ]

    for i, (name, model) in enumerate(models_dict.items()):
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred = model(img_tensor.unsqueeze(0))
                p_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]

        error_map = np.zeros_like(img_disp)
        error_map[(gt_mask > 0) & (p_mask == 0)] = [1, 0, 1] # Magenta=FN
        error_map[(gt_mask == 0) & (p_mask > 0)] = [0, 1, 0] # Green=FP
        error_map[(gt_mask > 0) & (p_mask > 0)] = [1, 1, 0] # Yellow=TP

        blend = 0.6 * img_disp + 0.4 * error_map
        plt.subplot(1, len(models_dict)+2, i+3); plt.imshow(blend); plt.title(name); plt.axis('off')

    plt.figlegend(handles=legend_elements, loc='lower center', ncol=3, fontsize='large')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "legend_qualitative_masks.png"), dpi=300)
    plt.close()

def plot_pr_curves(results_data, save_dir):
    class_names = ['BG', 'MA', 'HE', 'EX', 'SE', 'OD']

    # 1. Per-Class PR Curve
    for exp_name, data in results_data.items():
        plt.figure(figsize=(10, 7))
        for i in range(1, len(class_names)): 
            precision, recall, _ = precision_recall_curve(data['y_true'][:, i], data['y_probs'][:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{class_names[i]} (AUC = {pr_auc:.4f})')

        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Per-Class Precision-Recall Curves: {exp_name}')
        plt.legend(loc='best'); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"pr_class_{exp_name}.png"), dpi=300)
        plt.close()

    # 2. Per-Model Comparison (Focus on MA and HE)
    focus_lesions = [1, 2] 
    for lesion_idx in focus_lesions:
        plt.figure(figsize=(10, 7))
        for exp_name, data in results_data.items():
            precision, recall, _ = precision_recall_curve(data['y_true'][:, lesion_idx], data['y_probs'][:, lesion_idx])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{exp_name} (AUC = {pr_auc:.4f})')

        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Model Comparison PR Curve: {class_names[lesion_idx]}')
        plt.legend(loc='best'); plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"pr_model_comp_{class_names[lesion_idx]}.png"), dpi=300)
        plt.close()