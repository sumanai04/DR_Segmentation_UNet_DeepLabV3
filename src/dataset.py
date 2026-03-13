import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from tqdm import tqdm

class IDRiDDataset(Dataset):
    def __init__(self, data_paths, class_map, img_size=(1280, 1280), transform=None):
        self.data_paths = data_paths
        self.class_map = class_map
        self.img_size = img_size
        self.transform = transform
        self.images, self.masks = [], []
        prioritized_order = ['HE', 'EX', 'SE', 'MA', 'OD']

        print(f"Pre-loading {len(data_paths)} images with 1:1 Center Cropping...")
        for paths in tqdm(data_paths):
            image_orig = np.array(Image.open(paths['image']).convert("RGB"))
            h, w = image_orig.shape[:2]
            start_x = (w - h) // 2 if w > h else 0
            image_cropped = image_orig[:, start_x:start_x + h] if w > h else image_orig
            image = cv2.resize(image_cropped, self.img_size)

            combined_mask = np.zeros(self.img_size, dtype=np.int64)
            for class_key in prioritized_order:
                if class_key in paths:
                    class_id = self.class_map[class_key]
                    mask_orig = np.array(Image.open(paths[class_key]).convert('L'))
                    m_h, m_w = mask_orig.shape[:2]
                    m_start_x = (m_w - m_h) // 2 if m_w > m_h else 0
                    mask_cropped = mask_orig[:, m_start_x:m_start_x + m_h] if m_w > m_h else mask_orig
                    mask = cv2.resize(mask_cropped, self.img_size, interpolation=cv2.INTER_NEAREST)

                    if class_key == 'MA':
                        mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=1)
                    combined_mask[mask > 0] = class_id

            self.images.append(image)
            self.masks.append(combined_mask)

    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()
        return image, mask

# Standard Transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5), 
    A.Rotate(limit=30, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.5), 
    A.Normalize()
])

val_transform = A.Compose([
    A.Normalize()
])