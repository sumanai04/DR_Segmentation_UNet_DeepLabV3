# Diabetic Retinopathy Segmentation (U-Net vs. DeepLabV3)

This repository contains a comprehensive benchmark comparing **U-Net** and **DeepLabV3** architectures (using ResNet50 and ResNet101 backbones) for multi-class medical image segmentation of Diabetic Retinopathy lesions.

The pipeline is optimized for high-resolution images ($1280 \times 1280$) and handles extreme class imbalance using a specialized **Hybrid Loss Function** (Cross-Entropy + Dice + Focal Loss).

## Key Features
* **High-Resolution Training:** Utilizes PyTorch Automatic Mixed Precision (`torch.amp`) to fit $1280 \times 1280$ images into GPU memory.
* **Modular Pipeline:** Decoupled data loading, loss calculation, evaluation, and plotting.
* **Clinical Evaluation Metrics:** Evaluates models using AUPR, Dice Coefficient (F1), and Intersection over Union (IoU) with a focus on macro-averaging to account for rare lesions.
* **Qualitative Error Maps:** Custom visualization scripts generate 'Legend Error Maps' highlighting True Positives, False Positives (Green), and False Negatives (Magenta).

## Dataset Setup
This project uses the **Indian Diabetic Retinopathy Image Dataset (IDRiD)**. 

1. Download the dataset from the [IEEE Dataport - IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) (CC BY 4.0).
2. Extract the files into the `data/raw/` directory.
3. Your folder structure should look exactly like this:
   ```text
   data/
   └── raw/
       ├── 1. Original Images/
       │   ├── a. Training Set/
       │   └── b. Testing Set/
       └── 2. All Segmentation Groundtruths/
           ├── a. Training Set/
           │   ├── 1. Microaneurysms/
           │   └── ...
           └── b. Testing Set/
