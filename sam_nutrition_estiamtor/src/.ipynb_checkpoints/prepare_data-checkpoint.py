import os
import json

# === CONFIG ===
BASE_DIR = "/workspace/food-multiview-estimator/data"
images_dir = os.path.join(BASE_DIR, "nutrition5k-dataset-side-angle-images", "versions", "2")
metadata_files = [
    os.path.join(BASE_DIR, "nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"),
    os.path.join(BASE_DIR, "nutrition5k_dataset_metadata_dish_metadata_cafe2.csv")
]
output_json = os.path.join(BASE_DIR, "nutrition5k_multiview.json")

def build_json():
    dish_data = {}

    # 1. Read metadata
    for file in metadata_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                try:
                    dish_id = parts[0]
                    dish_data[dish_id] = {
                        "calories": float(parts[1]),
                        "weight": float(parts[2]),
                        "fat": float(parts[3]),
                        "carbs": float(parts[4]),
                        "protein": float(parts[5])
                    }
                except ValueError:
                    continue

    from collections import defaultdict
    import re

    records = []

    for dish_folder in os.listdir(images_dir):
        frames_path = os.path.join(images_dir, dish_folder, "frames_sampled30")

        if os.path.isdir(frames_path) and dish_folder in dish_data:
            image_files = [
                os.path.join(frames_path, f)
                for f in os.listdir(frames_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            # Agrupar imágenes por letra de cámara
            camera_groups = defaultdict(list)
            for path in image_files:
                match = re.search(r'camera_([A-Z])_', os.path.basename(path))
                if match:
                    camera_groups[match.group(1)].append(path)

            # Guardar solo si hay al menos 2 cámaras diferentes
            if len(camera_groups) >= 2:
                # Seleccionar una imagen por cámara
                images_one_per_camera = [group[0] for group in camera_groups.values()]
                
                records.append({
                    "dish_id": dish_folder,
                    "images": images_one_per_camera,
                    **dish_data[dish_folder]
                })


    # 3. Write JSON
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"Created: {output_json} with {len(records)} dishes")

if __name__ == "__main__":
    build_json()
