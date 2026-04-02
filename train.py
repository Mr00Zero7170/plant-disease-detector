import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

DATA_DIR    = "dataset"
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 20
NUM_CLASSES = 3
MODEL_PATH  = "plant_disease_model.keras"

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
    horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
test_data = test_gen.flow_from_directory(os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(), layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
]

history = model.fit(train_data, validation_data=test_data, epochs=EPOCHS, callbacks=callbacks)
print(f"\nBest val accuracy: {max(history.history['val_accuracy']):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val", linewidth=2, linestyle="--")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history["loss"], label="Train", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val", linewidth=2, linestyle="--")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Plot saved → training_curves.png")
