# run_fusion_inference.py
import os
import json
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

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

# --- Run inference on one sample ---
def run_fusion_inference(dish_id):
    # Buscar el platillo
    entry = next((e for e in dataset_json if e["dish_id"] == dish_id), None)
    if not entry:
        raise ValueError(f"Dish ID {dish_id} not found")

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

    # Usar solo la primera vista para segmentación
    overlay_img, seg_data = predict_masks(entry["images"][0])

    # Fusión
    fused_pred = fuse_predictions(effnet_pred, seg_data, metadata_lookup)

    return fused_pred, overlay_img, entry

# --- Ejemplo de uso ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dish_id = "dish_1550772454"  # reemplazá con un ID válido
    pred, overlay, entry = run_fusion_inference(dish_id)

    original_image_path = entry["images"][0]
    original = Image.open(original_image_path)

    plt.figure(figsize=(12, 6))

    # Imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    # Imagen segmentada
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Segmentación + Nutrientes")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nPredicción final fusionada:")
    print(f"Calories: {pred[0]:.2f} kcal")
    print(f"Protein : {pred[1]:.2f} g")
    print(f"Fat     : {pred[2]:.2f} g")
    print(f"Carbs   : {pred[3]:.2f} g")

    # Crear carpeta si no existe
    output_dir = "outputs/segmentations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar la imagen segmentada
    output_path = os.path.join(output_dir, f"{dish_id}.jpg")
    overlay.save(output_path)
    print(f"Imagen segmentada guardada en: {output_path}")
