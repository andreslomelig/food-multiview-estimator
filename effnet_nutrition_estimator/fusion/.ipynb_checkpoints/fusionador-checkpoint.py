# fusionador.py
import numpy as np

def fuse_predictions(effnet_pred, segmentation_data=None, metadata_lookup=None, strategy="confidence_weighted"):
    """
    Combina la predicción del modelo multiview con una inferida desde segmentación + metadata.

    Parameters:
    - effnet_pred: np.array shape (4,), [cal, fat, carb, prot]
    - segmentation_data: dict or None, debe contener 'segments': [{class_name, mask_area}], 'image_area'
    - metadata_lookup: dict mapping ingredient -> {cal, fat, carb, prot} por 100g
    - strategy: "confidence_weighted" | "only_segmentation" | "only_effnet"

    Returns:
    - np.array shape (4,), resultado final fusionado
    """

    if segmentation_data is None or metadata_lookup is None or not segmentation_data.get("segments"):
        return effnet_pred

    # Construir predicción basada en segmentación
    seg_total = np.zeros(4)
    total_area = segmentation_data.get("image_area", 1)

    for seg in segmentation_data["segments"]:
        ingr = seg["class_name"]
        area_ratio = seg["mask_area"] / total_area

        if ingr in metadata_lookup:
            nutr = metadata_lookup[ingr]  # dict con keys: cal, fat, carb, prot
            nutr_array = np.array([nutr["cal"], nutr["fat"], nutr["carb"], nutr["prot"]])
            seg_total += nutr_array * area_ratio  # peso por área

    if strategy == "only_segmentation":
        return seg_total
    elif strategy == "only_effnet":
        return effnet_pred
    elif strategy == "confidence_weighted":
        # Promedio ponderado simple (puedes mejorar esto con una red luego)
        # Si hay muchas clases reconocidas, damos más peso a la segmentación
        seg_weight = min(1.0, len(segmentation_data["segments"]) / 5)
        effnet_weight = 1.0 - seg_weight
        return effnet_pred * effnet_weight + seg_total * seg_weight
    else:
        raise ValueError("Unknown fusion strategy")
