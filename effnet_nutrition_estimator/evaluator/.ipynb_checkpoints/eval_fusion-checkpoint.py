# evaluator/eval_fusion.py
import os
import sys
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Add root path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.nutrition_dataset import NutritionDataset
from models.multiview_effnet import MultiviewEfficientNetB4
from fusion.fusionador import fuse_predictions
from model_inference import predict_masks  # devuelve (img_con_overlay, seg_data)

# --- CONFIG ---
EFFNET_MODEL_PATH = "checkpoints/multiview_effnet_b4.pth"
JSON_DATA_PATH = "../data/nutrition5k_multiview.json"
INGREDIENT_METADATA_PATH = "../data/nutrition5k_dataset_metadata_ingredients_metadata.csv"

IMAGE_SIZE = 380
MAX_VIEWS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

import csv

# --- Load Ingredient Metadata ---
metadata_lookup = {}
with open(INGREDIENT_METADATA_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        name = row["ingr"].strip().lower()
        metadata_lookup[name] = {
            "cal": float(row["cal/g"]),
            "fat": float(row["fat(g)"]),
            "carb": float(row["carb(g)"]),
            "prot": float(row["protein(g)"]),
        }

# --- Load model ---
model = MultiviewEfficientNetB4(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(EFFNET_MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Load dataset JSON ---
with open(JSON_DATA_PATH, 'r') as f:
    dataset_json = json.load(f)

# --- Run evaluation ---
total_mae = np.zeros(4)
count = 0

for entry in dataset_json:
    dish_id = entry["dish_id"]

    try:
        # Cargar imágenes
        images = []
        for img_path in entry["images"][:MAX_VIEWS]:
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
        while len(images) < MAX_VIEWS:
            images.append(images[-1])
        views = torch.stack(images).unsqueeze(0).to(DEVICE)  # (1, V, C, H, W)

        # EffNet predicción
        with torch.no_grad():
            effnet_pred = model(views).cpu().squeeze().numpy()  # (4,)

        # Segmentación
        _, seg_data = predict_masks(entry["images"][0])

        # Fusión
        fused_pred = fuse_predictions(effnet_pred, seg_data, metadata_lookup)

        # Ground truth
        gt = np.array([
            entry["calories"],
            entry["protein"],
            entry["fat"],
            entry["carbs"]
        ])

        total_mae += np.abs(fused_pred - gt)
        count += 1

    except Exception as e:
        print(f"Error with {dish_id}: {e}")
        continue

mae = total_mae / count
print("\n--- Fusion Evaluation MAE ---")
print(f"Calories MAE: {mae[0]:.2f} kcal")
print(f"Protein  MAE: {mae[1]:.2f} g")
print(f"Fat      MAE: {mae[2]:.2f} g")
print(f"Carbs    MAE: {mae[3]:.2f} g")
