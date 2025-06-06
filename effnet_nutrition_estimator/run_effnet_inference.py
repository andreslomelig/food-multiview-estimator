# run_effnet_inference.py
import os
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from data.nutrition_dataset import NutritionDataset
from models.multiview_effnet import MultiviewEfficientNetB4

# --- CONFIG ---
EFFNET_MODEL_PATH = "checkpoints/multiview_effnet_b4.pth"
JSON_DATA_PATH = "../data/nutrition5k_multiview.json"
IMAGE_SIZE = 380
MAX_VIEWS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --- Load model ---
model = MultiviewEfficientNetB4(pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(EFFNET_MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Load dataset JSON ---
with open(JSON_DATA_PATH, 'r') as f:
    dataset_json = json.load(f)

# --- Inference Function ---
def run_effnet_inference(dish_id):
    entry = next((e for e in dataset_json if e["dish_id"] == dish_id), None)
    if not entry:
        raise ValueError(f"Dish ID {dish_id} not found")

    # Load and preprocess views
    images = []
    for img_path in entry["images"][:MAX_VIEWS]:
        img = Image.open(img_path).convert('RGB')
        images.append(transform(img))
    while len(images) < MAX_VIEWS:
        images.append(images[-1])  # duplicate last
    views = torch.stack(images).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        pred = model(views).cpu().squeeze().numpy()

    # Ground truth
    real = np.array([
        entry["calories"],
        entry["protein"],
        entry["fat"],
        entry["carbs"]
    ])

    return pred, real

# --- Example usage ---
if __name__ == "__main__":
    dish_id = "dish_1550772454"   # cambia por uno válido
    pred, real = run_effnet_inference(dish_id)

    print(f"\nDish ID: {dish_id}")
    print(f"Predicción  -> Calories: {pred[0]:.2f}, Protein: {pred[1]:.2f}, Fat: {pred[2]:.2f}, Carbs: {pred[3]:.2f}")
    print(f"Ground Truth -> Calories: {real[0]:.2f}, Protein: {real[1]:.2f}, Fat: {real[2]:.2f}, Carbs: {real[3]:.2f}")
