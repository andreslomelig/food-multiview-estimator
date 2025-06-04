import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from model import NutritionEstimator
from data_loader import sam_model_registry, SamAutomaticMaskGenerator

TARGET_KEYS = ["calories", "carbs", "protein", "fat"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to model weights")
parser.add_argument("--images", nargs='+', required=True, help="List of image paths")
parser.add_argument("--use_sam", action='store_true', help="Apply SAM segmentation")
parser.add_argument("--sam_ckpt", default="sam_vit_h.pth", help="Path to SAM checkpoint")
parser.add_argument("--backbone", default="efficientnet_b4  ", help="Backbone name")
parser.add_argument("--feat_dim", type=int, default=512, help="Feature dimension")
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load SAM if needed
sam_generator = None
if args.use_sam:
    if not os.path.exists(args.sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found at {args.sam_ckpt}")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).to(device)
    sam_generator = SamAutomaticMaskGenerator(sam)
    print("✅ SAM loaded")

# Load and process images
processed_images = []
for path in args.images:
    img = Image.open(path).convert("RGB")
    original = np.array(img)

    if sam_generator:
        masks = sam_generator.generate(original)
        if masks:
            combined_mask = np.zeros_like(masks[0]['segmentation'], dtype=bool)
            for m in masks:
                combined_mask |= m['segmentation']
            combined_mask = combined_mask.astype(np.uint8)
            masked_img = original * combined_mask[:, :, None]
            img = Image.fromarray(masked_img.astype(np.uint8))

    processed_images.append(transform(img))

    # Show original and segmented
    fig, axs = plt.subplots(1, 2 if sam_generator else 1, figsize=(10, 5))
    axs = axs if isinstance(axs, np.ndarray) else [axs]
    axs[0].imshow(original)
    axs[0].set_title("Original")
    axs[0].axis("off")
    if sam_generator:
        axs[1].imshow(img)
        axs[1].set_title("Segmented")
        axs[1].axis("off")
    plt.show()

# Stack into batch of shape [1, V, C, H, W]
batch = torch.stack(processed_images).unsqueeze(0).to(device)

# Load model
model = NutritionEstimator(feat_dim=args.feat_dim, backbone_name=args.backbone).to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

with torch.no_grad():
    prediction = model(batch)[0].cpu().numpy()
    print("\n🍽️ Predicted Nutrition:")
    for key, value in zip(TARGET_KEYS, prediction):
        print(f"{key}: {value:.2f}")
