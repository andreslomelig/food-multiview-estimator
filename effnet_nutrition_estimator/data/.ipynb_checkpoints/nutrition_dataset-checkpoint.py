# nutrition_dataset.py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class NutritionDataset(Dataset):
    def __init__(self, json_path, transform=None, max_views=4):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.max_views = max_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        images = item['images'][:self.max_views]

        # Cargar y transformar cada imagen
        loaded_imgs = []
        for img_path in images:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            loaded_imgs.append(image)

        # Completar views si hay menos de max_views
        while len(loaded_imgs) < self.max_views:
            loaded_imgs.append(loaded_imgs[-1])

        image_tensor = torch.stack(loaded_imgs)

        # Targets: [calories, protein, fat, carbs]
        target = torch.tensor([
            item['calories'],
            item['protein'],
            item['fat'],
            item['carbs']
        ], dtype=torch.float32)

        return image_tensor, target
