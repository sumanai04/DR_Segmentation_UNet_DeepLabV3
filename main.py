import os
import time
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import modularized components
from src.dataset import IDRiDDataset, train_transform, val_transform
from src.metrics import get_combined_loss_fn, evaluate_model, get_raw_eval
from src.visualize import save_masks_with_legend, plot_pr_curves

# --- CONFIGURATION ---
BASE_PATH = "./data/raw"
TRAIN_IMG_PATH = os.path.join(BASE_PATH, "1. Original Images/a. Training Set")
TEST_IMG_PATH = os.path.join(BASE_PATH, "1. Original Images/b. Testing Set")
GT_BASE_PATH = os.path.join(BASE_PATH, "2. All Segmentation Groundtruths")
TRAIN_GT_PATH = os.path.join(GT_BASE_PATH, "a. Training Set")
TEST_GT_PATH = os.path.join(GT_BASE_PATH, "b. Testing Set")

SAVE_DIR = "./results"
WEIGHTS_DIR = "./weights"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

CLASS_MAP = {'MA': 1, 'HE': 2, 'EX': 3, 'SE': 4, 'OD': 5}
NUM_CLASSES = len(CLASS_MAP) + 1
BATCH_SIZE = 8
EPOCHS = 100
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['BG', 'MA', 'HE', 'EX', 'SE', 'OD']

def get_paths(img_dir, gt_dir):
    if not os.path.exists(img_dir):
        print(f"Warning: Data directory {img_dir} not found. Please check README for setup.")
        return []
    
    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    data = []
    for f in files:
        name = os.path.splitext(f)[0]
        d = {'image': os.path.join(img_dir, f)}
        lesions = ['1. Microaneurysms', '2. Haemorrhages', '3. Hard Exudates', '4. Soft Exudates']
        for i, L in enumerate(lesions, 1):
            key = list(CLASS_MAP.keys())[i-1]
            p = os.path.join(gt_dir, L, f"{name}_{key}.tif")
            if os.path.exists(p): d[key] = p
        od = os.path.join(gt_dir, '5. Optic Disc', f"{name}_OD.tif")
        if os.path.exists(od): d['OD'] = od
        data.append(d)
    return data

def main():
    print(f"Using device: {DEVICE}")
    train_files = get_paths(TRAIN_IMG_PATH, TRAIN_GT_PATH)
    test_files = get_paths(TEST_IMG_PATH, TEST_GT_PATH)
    
    if not train_files:
        return

    train_split, val_split = train_test_split(train_files, test_size=0.1, random_state=42)

    train_loader = DataLoader(IDRiDDataset(train_split, CLASS_MAP, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_subset_loader = DataLoader(IDRiDDataset(test_files[:6], CLASS_MAP, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(IDRiDDataset(test_files, CLASS_MAP, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    experiments = [("U-Net", "resnet50"), ("U-Net", "resnet101"), ("DeepLabV3", "resnet50"), ("DeepLabV3", "resnet101")]
    macro_summaries, models_trained, csv_rows, loss_history, pr_data_storage = {}, {}, [], {}, {}
    combined_loss = get_combined_loss_fn(DEVICE)

    for arch, backbone in experiments:
        exp_name = f"{arch}_{backbone}"
        print(f"\n--- Training {exp_name} ---")

        if arch == "U-Net":
            model = smp.Unet(encoder_name=backbone, classes=NUM_CLASSES).to(DEVICE)
        elif arch == "DeepLabV3":
            model = smp.DeepLabV3(encoder_name=backbone, classes=NUM_CLASSES).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler = torch.amp.GradScaler('cuda')

        best_aupr, t_start = 0, time.time()
        current_model_losses = []

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
            for imgs, msks in loop:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = combined_loss(preds, msks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(train_loader)
            current_model_losses.append(avg_loss)

            v_aupr, _, _, _, _, _, _ = evaluate_model(val_subset_loader, model, DEVICE, NUM_CLASSES)

            if v_aupr > best_aupr:
                best_aupr = v_aupr
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"best_{exp_name}.pth"))

            print(f"Epoch {epoch+1} Complete | Loss: {avg_loss:.4f} | Subset Val AUPR: {v_aupr:.4f}")

        loss_history[exp_name] = current_model_losses
        t_train = (time.time() - t_start) / 60

        # FINAL EVALUATION
        model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, f"best_{exp_name}.pth")))
        t_aupr, t_f1, t_iou, c_aupr, c_f1, c_iou, t_inf = evaluate_model(test_loader, model, DEVICE, NUM_CLASSES)
        
        y_true, y_probs = get_raw_eval(test_loader, model, DEVICE, NUM_CLASSES)
        pr_data_storage[exp_name] = {'y_true': y_true, 'y_probs': y_probs}

        row = {'Model': exp_name, 'Macro_AUPR': t_aupr, 'Macro_Dice': t_f1, 'Macro_IoU': t_iou, 'Train_Time': t_train, 'Inf_Time': t_inf}
        for i, name in enumerate(CLASS_NAMES):
            row[f'{name}_AUPR'] = c_aupr[i]; row[f'{name}_Dice'] = c_f1[i]; row[f'{name}_IoU'] = c_iou[i]
        csv_rows.append(row)
        macro_summaries[exp_name] = {'AUPR': t_aupr, 'Dice': t_f1, 'IoU': t_iou}
        models_trained[exp_name] = model

    # --- FINAL EXPORTS & VISUALIZATIONS ---
    plot_pr_curves(pr_data_storage, SAVE_DIR)
    pd.DataFrame(csv_rows).to_csv(os.path.join(SAVE_DIR, "comprehensive_results.csv"), index=False)
    
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(os.path.join(SAVE_DIR, "training_loss_report.csv"), index_label="Epoch")

    plt.figure(figsize=(12, 6))
    for model_name in loss_df.columns:
        plt.plot(range(1, EPOCHS + 1), loss_df[model_name], label=model_name)
    plt.title("Training Loss Convergence (Hybrid Loss)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "loss_convergence_plot.png"), dpi=300); plt.close()

    df_macro = pd.DataFrame(macro_summaries).T
    plt.figure(figsize=(10, 6)); sns.heatmap(df_macro, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.savefig(os.path.join(SAVE_DIR, "macro_metrics_heatmap.png"), dpi=300); plt.close()
    
    save_masks_with_legend(models_trained, test_loader, DEVICE, NUM_CLASSES, SAVE_DIR)
    print("Pipeline Complete. Results saved in ./results/")

if __name__ == '__main__':
    main()