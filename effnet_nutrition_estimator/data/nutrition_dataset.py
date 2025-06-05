# nutrition_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class NutritionDataset(Dataset):
    def __init__(self, image_root, metadata_csv, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.meta = pd.read_csv(metadata_csv)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        dish_id = str(row['dish_id'])
        views = []
        for view in ['A', 'B', 'C', 'D']:
            img_path = os.path.join(
                self.image_root,
                f"dish_{dish_id}",
                "frames_sampled30",
                f"dish_{dish_id}_{view}.jpg"
            )
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            views.append(image)
        views = torch.stack(views)  # Shape: (4, C, H, W)

        targets = torch.tensor([
            row['calories'],
            row['protein'],
            row['fat'],
            row['carbs']
        ], dtype=torch.float32)

        return views, targets
