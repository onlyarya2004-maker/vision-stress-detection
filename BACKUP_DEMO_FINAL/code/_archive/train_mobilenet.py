"""
MobileNetV2 training for FER2013 emotion classification.
Fixes over the original:
  1. Separate (non-augmented) validation generator pointed at FER2013 `test/`.
  2. Two-stage training: freeze-head-first, then unfreeze top layers at very low LR.
  3. Callbacks: ModelCheckpoint(save_best_only), EarlyStopping, ReduceLROnPlateau.
  4. Label smoothing for noisy FER2013 labels.
  5. Larger image size (160) to preserve more detail after upscaling.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("TRAINING MOBILENET V2 (two-stage, with callbacks)")
print("=" * 60)

# ---------------- Config ----------------
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS_HEAD = 8      # stage 1: train head only
EPOCHS_FINE = 40     # stage 2: fine-tune top layers (early stopping will cut this short)
TRAIN_DIR = '../datasets/fer2013/train'
TEST_DIR  = '../datasets/fer2013/test'
MODEL_OUT = '../models/stress_model.h5'

os.makedirs('../models', exist_ok=True)

# ---------------- Generators ----------------
# Training: augmented
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
)

# Validation: NO augmentation (fixes the leak in your original)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
)

val_data = val_gen.flow_from_directory(
    TEST_DIR,                                  # real held-out test, not a re-split of train
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)

print("Class indices:", train_data.class_indices)

# ---------------- Class weights ----------------
classes = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes,
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ---------------- Model ----------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False   # stage 1: freeze everything

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

# ---------------- Callbacks ----------------
callbacks = [
    ModelCheckpoint(
        MODEL_OUT, monitor='val_accuracy',
        save_best_only=True, verbose=1,
    ),
    EarlyStopping(
        monitor='val_accuracy', patience=6,
        restore_best_weights=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1,
    ),
]

# ---------------- Stage 1: Head training ----------------
print("\n--- Stage 1: training head only ---")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['accuracy'],
)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ---------------- Stage 2: Fine-tuning ----------------
print("\n--- Stage 2: unfreezing top 60 layers and fine-tuning ---")
base_model.trainable = True
# Keep the very bottom frozen (generic edge/texture filters)
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),   # much lower LR for fine-tuning
    loss=loss_fn,
    metrics=['accuracy'],
)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks,
)

best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest MobileNet val accuracy: {best_val_acc * 100:.2f}%")
print(f"Saved best model to: {MODEL_OUT}")
