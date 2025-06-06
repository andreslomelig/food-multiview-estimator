# Food Multiview Estimator

**Food Multiview Estimator** is a deep learning system that estimates calories and macronutrients (protein, carbohydrates, fat, and weight) from multiple RGB images of a meal. It requires no depth sensors, calibration objects, or manual segmentation.

The project combines multiview inference, automatic semantic segmentation, and ingredient metadata to perform nutritional estimation directly from food photos.

---

## Associated Paper

**Title**: *Seeing Through the Plate: Estimating Nutrition from Multiview Food Images*  
**Author**: Andrés Lomelí (Universidad Panamericana, Mexico)

**Abstract**:  
This work proposes and compares two pipelines for nutrition prediction from RGB food images. The first uses a multiview transformer-like model for direct regression. The second fuses predictions from a CNN with semantic segmentation and ingredient metadata. We highlight strengths, limitations, and the role of segmentation quality and ingredient coverage in final predictions.

> The full paper is available inside the `/docs/` folder.

---

## Features

- Multiview architecture based on EfficientNet  
- Direct prediction of calories, protein, carbohydrates, fat, and total weight  
- Supports 2 to 4 RGB views per dish  
- Ingredient-based fusion using segmentation + metadata  
- Mask2Former segmentation with FoodSeg103 labels  
- Modular codebase with training, evaluation, and inference scripts

---

## Results

**Validation Performance Summary**:

| Model                            | Calories MAE | Protein MAE | Fat MAE | Carbs MAE |
|----------------------------------|--------------|-------------|---------|-----------|
| EfficientNet-B0 (3 views)        | 154.20       | 11.32       | 10.63   | 13.41     |
| EfficientNet-B4 (4 views)        | 87.08        | 7.74        | 7.09    | 9.65      |
| B4 + SAM Segmentation            | 78.13        | 6.90        | 6.89    | 10.68     |
| EfficientNet-B4 (new pipeline)   | 46.32        | 6.85        | 5.03    | 10.68     |
| Fusion (EffNet + M2F + metadata) | In progress  | In progress | In progress | In progress |

---

## Datasets

We use the following datasets:

- **Nutrition5k**:  
  [https://github.com/apple/ml-nutrition5k](https://github.com/apple/ml-nutrition5k)  
  Contains multiview RGB food images and ground truth nutritional values.

- **FoodSeg103**:  
  [https://github.com/NimaVahdat/foodseg103](https://github.com/NimaVahdat/foodseg103)  
  Used for ingredient segmentation via pretrained Mask2Former.

To train and run the code:
/data/
├── nutrition5k_multiview.json
├── nutrition5k_dataset_metadata_ingredients_metadata.csv
├── (image folders organized as in Apple’s repo)


---

## Installation

```bash
# Create environment
conda create -n food_estimator python=3.10 -y
conda activate food_estimator

# Install dependencies
pip install -r requirements.txt
