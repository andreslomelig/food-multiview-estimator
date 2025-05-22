# 🍽️ Food Multiview Estimator

**Food Multiview Estimator** is a deep learning system that predicts calories and macronutrients (protein, carbohydrates, fat, and weight) from multiple RGB images of a meal — no depth sensors or calibration objects required.

This project combines computer vision, automatic segmentation, and a multiview transformer architecture to perform accurate, real-time nutritional estimation from food photos.

---

## 📄 Associated Paper

**Title**: *Seeing Through the Plate: Estimating Nutrition from Multiview Food Images*  
**Author**: Andrés Lomelí (Universidad Panamericana, Mexico)  
**Abstract**: This work proposes a lightweight, sensor-free multiview framework for estimating detailed nutritional values from RGB images of meals. The system uses a CNN backbone and a cross-view transformer to infer food volume and composition implicitly through image attention, enabling direct regression of nutritional values.

> You can find the full paper inside the `/docs/` folder of this repository.

---

## 📦 Features

- 🧠 Multiview transformer-based architecture  
- 🎯 Direct prediction of calories, protein, carbohydrates, fat, and total weight  
- 📸 Supports 2–4 RGB views per meal  
- ✂️ Integration with [SAM (Segment Anything)](https://segment-anything.com/) for zero-shot food segmentation  
- ⚙️ Professional modular structure and training scripts

---

## 🚀 Installation

```bash
# Create a new environment
conda create -n food_estimator python=3.10 -y
conda activate food_estimator

# Install required packages
pip install -r requirements.txt
