import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re
from collections import defaultdict

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    sam_model_registry = None  # In case SAM isn't installed

class NutritionDataset(Dataset):
    def __init__(self, json_path: str, num_views: int = 3, apply_sam: bool = False, sam_model_type="vit_h", sam_ckpt_path=None):
        # Usar JSON segmentado si apply_sam está activado
        if apply_sam and "sam_segmented" not in json_path:
            json_path = json_path.replace("nutrition5k_multiview.json", "nutrition5k_sam_segmented.json")

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.num_views = num_views
        self.apply_sam = apply_sam
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.sam_generator = None
        if self.apply_sam and sam_model_registry is not None and sam_ckpt_path and os.path.exists(sam_ckpt_path):
            try:
                sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt_path)
                sam.to("cuda")
                self.sam_generator = SamAutomaticMaskGenerator(sam)
                print("✅ SAM generator loaded")
                print("🔧 SAM running on:", next(sam.parameters()).device)
            except Exception as e:
                print("❌ Failed to initialize SAM:", e)
        else:
            if self.apply_sam:
                print("⚠️ SAM enabled but generator not initialized — assuming pre-segmented images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        images_paths = item['images']
        selected_paths = images_paths[:self.num_views]

        while len(selected_paths) < self.num_views:
            selected_paths.append(selected_paths[-1])

        processed_images = []

        for path in selected_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            processed_images.append(img)

        stacked = torch.stack(processed_images)

        targets = {
            "calories": torch.tensor(item["calories"], dtype=torch.float32),
            "carbs": torch.tensor(item["carbs"], dtype=torch.float32),
            "protein": torch.tensor(item["protein"], dtype=torch.float32),
            "fat": torch.tensor(item["fat"], dtype=torch.float32),
            "weight": torch.tensor(item["weight"], dtype=torch.float32)
        }

        return stacked, targets
