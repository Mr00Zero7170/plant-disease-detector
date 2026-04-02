import os
import shutil
import random

SOURCE_DIR = "PlantVillage_raw"
DEST_DIR   = "dataset"
CLASSES    = ["healthy", "early_blight", "late_blight"]
SPLIT      = 0.8

random.seed(42)

print("Creating folder structure...")
for split in ["train", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

print("Splitting images...\n")
for cls in CLASSES:
    src = os.path.join(SOURCE_DIR, cls)
    if not os.path.exists(src):
        print(f"  [MISSING] {src} — skipping")
        continue
    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    cut = int(len(images) * SPLIT)
    splits = {"train": images[:cut], "test": images[cut:]}
    for split, files in splits.items():
        for f in files:
            shutil.copy(os.path.join(src, f), os.path.join(DEST_DIR, split, cls, f))
    print(f"  {cls:15s}  {cut} train  /  {len(images) - cut} test")

print("\nDone!")
