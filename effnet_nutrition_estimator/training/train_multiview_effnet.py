# train_multiview_effnet.py
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from nutrition_dataset import NutritionDataset
from multiview_effnet import MultiviewEfficientNetB4

# --- Load config ---
with open("../configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Configuration ---
data_root = config["paths"]["data_root"]
metadata_csv = config["paths"]["metadata_csv"]
model_path = config["paths"]["model_checkpoint"]
batch_size = config["training"]["batch_size"]
num_epochs = config["training"]["num_epochs"]
lr = config["training"]["learning_rate"]
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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# --- Model ---
model = MultiviewEfficientNetB4(pretrained=True).to(device)

# --- Optimizer and Loss ---
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# --- Training Loop ---
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for views, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        views = views.to(device)  # shape: (B, 4, C, H, W)
        targets = targets.to(device)  # shape: (B, 4)

        preds = model(views)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * views.size(0)

    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

# --- Save final model ---
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print("Model saved to:", model_path)
