# eval_multiview_effnet.py
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from nutrition_dataset import NutritionDataset
from multiview_effnet import MultiviewEfficientNetB4

# --- Load config ---
with open("../configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Configuration from config.yaml ---
data_root = config["paths"]["data_root"]
metadata_csv = config["paths"]["metadata_csv"]
model_path = config["paths"]["model_checkpoint"]
batch_size = config["training"]["batch_size"]
input_size = config["training"]["input_size"]
num_workers = config["training"].get("num_workers", 4)
device = torch.device("cuda" if config["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
])

# --- Dataset and DataLoader ---
dataset = NutritionDataset(data_root, metadata_csv, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# --- Model ---
model = MultiviewEfficientNetB4(pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Evaluation ---
mae_sum = torch.zeros(4).to(device)
count = 0

with torch.no_grad():
    for views, targets in dataloader:
        views = views.to(device)
        targets = targets.to(device)
        preds = model(views)
        mae_sum += torch.sum(torch.abs(preds - targets), dim=0)
        count += targets.size(0)

mae = mae_sum / count
print("\n--- MAE per Nutrient ---")
print(f"Calories MAE: {mae[0].item():.2f} kcal")
print(f"Protein  MAE: {mae[1].item():.2f} g")
print(f"Fat      MAE: {mae[2].item():.2f} g")
print(f"Carbs    MAE: {mae[3].item():.2f} g")
