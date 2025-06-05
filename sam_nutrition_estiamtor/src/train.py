import os
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_loader import NutritionDataset
from model import NutritionEstimator

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)  # [B, V, C, H, W]
    targets_stacked = {k: torch.stack([d[k] for d in targets]) for k in targets[0]}
    return images, targets_stacked

def evaluate(model, loader, device):
    model.eval()
    mse_scores = {k: [] for k in ['calories', 'carbs', 'protein', 'fat', 'weight']}
    mae_scores = {k: [] for k in mse_scores}

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            target_tensor = torch.stack([targets[k] for k in mse_scores], dim=1).to(device)

            outputs = model(images)
            for i, key in enumerate(mse_scores):
                pred = outputs[:, i].cpu().numpy()
                true = target_tensor[:, i].cpu().numpy()
                mse_scores[key].append(mean_squared_error(true, pred))
                mae_scores[key].append(mean_absolute_error(true, pred))

    final_mse = {k: sum(v) / len(v) for k, v in mse_scores.items()}
    final_mae = {k: sum(v) / len(v) for k, v in mae_scores.items()}

    print("\n📊 Validation Metrics:")
    for k in final_mse:
        print(f"{k}: MSE={final_mse[k]:.2f}, MAE={final_mae[k]:.2f}")
    return final_mse, final_mae

def main():
    config = load_config("configs/default.yaml")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = NutritionDataset(
        json_path=config["data"]["json_path"],
        num_views=config["data"]["num_views"],
        apply_sam=config["data"]["apply_sam"],
        sam_model_type=config["data"]["sam_model_type"],
        sam_ckpt_path=config["data"]["sam_ckpt_path"]
    )

    # Split into train/val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    model = NutritionEstimator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = images.to(device)
            labels = torch.stack([targets[k] for k in ['calories', 'carbs', 'protein', 'fat', 'weight']], dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"\n🔁 Epoch {epoch}/{config['training']['num_epochs']} - Train Loss: {avg_loss:.4f}")
        evaluate(model, val_loader, device)

        # Save checkpoint
        os.makedirs(config["output_dir"], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config["output_dir"], f"model_epoch_{epoch}.pt"))

if __name__ == "__main__":
    main()
