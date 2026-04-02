import os
import shutil
import random
import kagglehub

print("Downloading PlantVillage dataset from Kaggle...")
path = kagglehub.dataset_download("emmarex/plantdisease")
print(f"Downloaded to: {path}\n")

def find_plantvillage_root(base):
    for root, dirs, files in os.walk(base):
        subdirs = [d for d in dirs if any(
            f.lower().endswith((".jpg", ".jpeg", ".png"))
            for f in os.listdir(os.path.join(root, d))
        )]
        if len(subdirs) >= 3:
            return root
    return base

dataset_root = find_plantvillage_root(path)
print(f"Dataset root found: {dataset_root}")
print(f"Folders inside: {os.listdir(dataset_root)}\n")

CLASS_MAP = {
    "Tomato__Tomato_healthy":           "healthy",
    "Tomato_healthy":                   "healthy",
    "Tomato___healthy":                 "healthy",
    "Tomato__Tomato_Early_blight":      "early_blight",
    "Tomato_Early_blight":              "early_blight",
    "Tomato___Early_blight":            "early_blight",
    "Tomato__Tomato_Late_blight":       "late_blight",
    "Tomato_Late_blight":               "late_blight",
    "Tomato___Late_blight":             "late_blight",
}

DEST_DIR = "dataset"
SPLIT    = 0.8
random.seed(42)

for split in ["train", "test"]:
    for cls in ["healthy", "early_blight", "late_blight"]:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

found = []
for folder in os.listdir(dataset_root):
    clean_name = CLASS_MAP.get(folder)
    if clean_name is None:
        continue
    src = os.path.join(dataset_root, folder)
    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    cut = int(len(images) * SPLIT)
    splits = {"train": images[:cut], "test": images[cut:]}
    for split, files in splits.items():
        for f in files:
            shutil.copy(os.path.join(src, f), os.path.join(DEST_DIR, split, clean_name, f))
    print(f"  {clean_name:15s}  {cut} train  /  {len(images) - cut} test")
    found.append(clean_name)

if not found:
    print("\n[!] No matching folders found. Available folders:")
    for f in sorted(os.listdir(dataset_root)):
        print(f"    {f}")
    print("\nPaste the folder names here and I'll fix the CLASS_MAP.")
else:
    print(f"\nDone! Dataset ready in ./dataset/")
    for split in ["train", "test"]:
        for cls in ["healthy", "early_blight", "late_blight"]:
            p = os.path.join(DEST_DIR, split, cls)
            count = len(os.listdir(p)) if os.path.exists(p) else 0
            print(f"  dataset/{split}/{cls}/  ({count} images)")
