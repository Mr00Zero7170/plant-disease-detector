# 🌿 Plant Disease Detector

CNN that classifies tomato leaves as **healthy**, **early_blight**, or **late_blight** using the PlantVillage dataset.

## Quickstart
```bash
pip install -r requirements.txt
python prepare_dataset.py
python train.py
python evaluate.py
python predict.py path/to/leaf.jpg
```

## Dataset
Download [PlantVillage from Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease), rename folders to `healthy`, `early_blight`, `late_blight` and place inside `PlantVillage_raw/`.
