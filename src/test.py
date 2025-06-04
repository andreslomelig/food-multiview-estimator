from data_loader import NutritionDataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import os

# Dataset con SAM activo
dataset = NutritionDataset(
    json_path="data/nutrition5k_multiview.json",
    num_views=2,
    apply_sam=True,
    sam_model_type="vit_h",
    sam_ckpt_path="sam_vit_h.pth"
)

# Usamos la data original para obtener rutas
sample = dataset.data[0]
original_paths = sample['images'][:2]
original_images = [Image.open(p).convert("RGB") for p in original_paths]

# Aplicamos SAM manualmente
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam.to("cuda")
mask_gen = SamAutomaticMaskGenerator(sam)

segmented_images = []
masks_bin = []
img_path = "data/nutrition5k-dataset-side-angle-images/versions/2/dish_1550704750/frames_sampled30/camera_A_frame_001.jpeg"
for img in original_images:
    img_np = np.array(img)
    masks = mask_gen.generate(img_np)
    print(f"SAM generó {len(masks)} máscaras")

    if masks:
        combined = np.zeros_like(masks[0]['segmentation'], dtype=bool)
        for m in masks:
            combined |= m['segmentation']

        # Guardar la máscara binaria (0/255)
        masks_bin.append(combined.astype(np.uint8) * 255)

        # Aplicar máscara a la imagen
        masked = img_np * combined[:, :, None]
        segmented_images.append(Image.fromarray(masked.astype(np.uint8)))
    else:
        print("⚠️ SAM no encontró nada")
        masks_bin.append(np.zeros_like(img_np[:, :, 0]))
        segmented_images.append(img)

# Visualizar las 3 filas
views = len(original_images)
fig, axs = plt.subplots(3, views, figsize=(6 * views, 10))

for i in range(views):
    axs[0, i].imshow(original_images[i])
    axs[0, i].set_title(f"Original View {i+1}")
    axs[1, i].imshow(segmented_images[i])
    axs[1, i].set_title(f"SAM Segment View {i+1}")
    axs[2, i].imshow(masks_bin[i], cmap='gray')
    axs[2, i].set_title(f"Mask View {i+1}")
    for ax in axs[:, i]:
        ax.axis("off")

plt.tight_layout()
plt.show()
