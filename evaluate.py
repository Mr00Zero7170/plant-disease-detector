import os, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

MODEL_PATH  = "plant_disease_model.keras"
CLASS_NAMES = ["early_blight", "healthy", "late_blight"]
IMG_SIZE    = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)
gen = ImageDataGenerator(rescale=1./255)
test_data = gen.flow_from_directory("dataset/test", target_size=IMG_SIZE,
    batch_size=32, class_mode="categorical", shuffle=False)
preds = model.predict(test_data, verbose=1)
y_pred, y_true = np.argmax(preds, axis=1), test_data.classes
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved → confusion_matrix.png")
