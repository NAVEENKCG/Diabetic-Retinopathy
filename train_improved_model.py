"""
Improved Diabetic Retinopathy Model Training Script
====================================================
Uses MobileNetV2 (transfer learning) + heavy data augmentation
to produce a much more robust and accurate model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ==================== Configuration ==================== #
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30  # EarlyStopping will prevent overfitting
DATASET_DIR = "diabetic_retinopathy_dataset"
MODEL_SAVE_PATH = "model-folder/diabetic-retino-model.h5"
BACKUP_MODEL_PATH = "model-folder/diabetic-retino-model-backup.h5"


# ==================== Backup Old Model ==================== #
if os.path.exists(MODEL_SAVE_PATH):
    import shutil
    shutil.copy2(MODEL_SAVE_PATH, BACKUP_MODEL_PATH)
    print(f"[INFO] Old model backed up to: {BACKUP_MODEL_PATH}")


# ==================== Data Augmentation ==================== #
# This is the KEY improvement. Heavy augmentation teaches the model
# to handle different brightness, contrast, colors, and orientations.

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,              # Random rotation up to 30 degrees
    width_shift_range=0.15,         # Horizontal shift
    height_shift_range=0.15,        # Vertical shift
    shear_range=0.1,                # Shearing
    zoom_range=0.2,                 # Random zoom in/out
    horizontal_flip=True,           # Flip horizontally
    vertical_flip=True,             # Flip vertically (fundus images can be either eye)
    brightness_range=[0.7, 1.3],    # Random brightness changes (KEY for your issue!)
    fill_mode="nearest",
)

# Validation/Test data should NOT be augmented — only rescaled.
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


# ==================== Data Loaders ==================== #
print("\n[INFO] Loading training data...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
)

print("\n[INFO] Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "valid"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("\n[INFO] Loading test data...")
test_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# Print class mapping
print(f"\n[INFO] Class indices: {train_generator.class_indices}")
print(f"[INFO] Training samples: {train_generator.samples}")
print(f"[INFO] Validation samples: {val_generator.samples}")
print(f"[INFO] Test samples: {test_generator.samples}")


# ==================== Build Improved Model ==================== #
print("\n[INFO] Building improved model with MobileNetV2 backbone...")

# Load MobileNetV2 pre-trained on ImageNet (without the top classification layer)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# Freeze the base model initially (we'll fine-tune later)
base_model.trainable = False

# Build custom classification head
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)           # Much better than Flatten!
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid", name="preds")(x)

model = Model(inputs, outputs, name="DR_MobileNetV2")
model.summary()


# ==================== Phase 1: Train Classification Head ==================== #
print("\n" + "=" * 60)
print("PHASE 1: Training classification head (base frozen)")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase1 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

history_phase1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks_phase1,
)


# ==================== Phase 2: Fine-tune Top Layers ==================== #
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning top layers of MobileNetV2")
print("=" * 60)

# Unfreeze the last 30 layers of MobileNetV2 for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a much lower learning rate to avoid destroying learned weights
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
)


# ==================== Evaluate on Test Set ==================== #
print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==================== Classification Report ==================== #
from sklearn.metrics import classification_report, confusion_matrix

test_generator.reset()
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# ==================== Save Final Model ==================== #
model.save(MODEL_SAVE_PATH)
print(f"\n[SUCCESS] Improved model saved to: {MODEL_SAVE_PATH}")
print("[INFO] Your old model is backed up at:", BACKUP_MODEL_PATH)
print("\nDone! You can now restart your Streamlit app to use the improved model.")
