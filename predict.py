import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys, os

MODEL_PATH  = "plant_disease_model.keras"
CLASS_NAMES = ["early_blight", "healthy", "late_blight"]
IMG_SIZE    = (128, 128)

if not os.path.exists(MODEL_PATH):
    print("Model not found. Run train.py first.")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
img_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ").strip()
img = image.load_img(img_path, target_size=IMG_SIZE)
arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
preds = model.predict(arr, verbose=0)[0]
print(f"\n  Prediction : {CLASS_NAMES[np.argmax(preds)].upper()}")
print(f"  Confidence : {preds.max()*100:.1f}%")
for name, prob in zip(CLASS_NAMES, preds):
    print(f"    {name:15s}  {prob*100:5.1f}%  {'█' * int(prob*20)}")
