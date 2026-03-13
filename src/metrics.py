import time
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import f1_score, jaccard_score, average_precision_score

def get_combined_loss_fn(device, class_weights_list=[1.0, 35.0, 5.0, 5.0, 5.0, 5.0]):
    class_weights = torch.tensor(class_weights_list).to(device)
    loss_fn_ce = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    loss_fn_focal = smp.losses.FocalLoss(mode='multiclass')
    
    def combined_loss(y_pred, y_true):
        return 0.4 * loss_fn_ce(y_pred, y_true) + 0.3 * loss_fn_dice(y_pred, y_true) + 0.3 * loss_fn_focal(y_pred, y_true)
    
    return combined_loss

def evaluate_model(loader, model, device, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    start_inf = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs = np.transpose(probs, (0, 2, 3, 1)).reshape(-1, num_classes)
                all_probs.append(probs)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().flatten())
                all_labels.extend(masks.cpu().numpy().flatten())

    inf_time = time.time() - start_inf
    all_probs = np.concatenate(all_probs)
    all_labels = np.array(all_labels)
    y_onehot = np.eye(num_classes)[all_labels]

    return (
        average_precision_score(y_onehot, all_probs, average='macro'),
        f1_score(all_labels, all_preds, average='macro', zero_division=0),
        jaccard_score(all_labels, all_preds, average='macro', zero_division=0),
        average_precision_score(y_onehot, all_probs, average=None),
        f1_score(all_labels, all_preds, average=None, zero_division=0),
        jaccard_score(all_labels, all_preds, average=None, zero_division=0),
        inf_time
    )

def get_raw_eval(loader, model, device, num_classes):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for imgs, msks in loader:
                logits = model(imgs.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(np.transpose(probs, (0, 2, 3, 1)).reshape(-1, num_classes))
                all_labels.extend(msks.cpu().numpy().flatten())

    y_probs = np.concatenate(all_probs)
    y_true = np.eye(num_classes)[np.array(all_labels)]
    return y_true, y_probs