"""
Train a BINARY stress classifier (not_stressed / stressed) on DaiSEE.
This is the PRIMARY stress model.

Inputs come from daisee_preprocessing.py + rebucket_to_binary.py:
  processed_daisee/
    train/{not_stressed,stressed}/
    val/{not_stressed,stressed}/
    test/{not_stressed,stressed}/

Outputs:
  ../models/daisee_mobilenet_stress.h5
  ../models/daisee_efficientnet_stress.h5

The continuous stress probability (model output for the "stressed" class) is
what the inference pipeline uses; the dashboard maps it into LOW/MODERATE/HIGH
display zones using the WESAD-calibrated thresholds at runtime.
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIG ----------------
DATA_ROOT = Path('../datasets/processed_daisee')
MODEL_DIR = Path('../models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE   = 32
EPOCHS_HEAD  = 6
EPOCHS_FINE  = 30
LABEL_SMOOTH = 0.1

CLASS_ORDER = ['not_stressed', 'stressed']   # fixed order so index 1 == stressed


def build_generators(preprocess_fn, img_size):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.08,
        height_shift_range=0.08,
        fill_mode='nearest',
    )
    eval_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_data = train_gen.flow_from_directory(
        DATA_ROOT / 'train',
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_ORDER,
        shuffle=True,
    )
    val_data = eval_gen.flow_from_directory(
        DATA_ROOT / 'val',
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_ORDER,
        shuffle=False,
    )
    test_data = eval_gen.flow_from_directory(
        DATA_ROOT / 'test',
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_ORDER,
        shuffle=False,
    )
    return train_data, val_data, test_data


def build_head(base_model, num_classes):
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=out)


def train_one(backbone_name, img_size, unfreeze_top_n, out_path):
    print("\n" + "=" * 64)
    print(f"TRAINING {backbone_name.upper()} stress classifier (binary)")
    print("=" * 64)

    if backbone_name == 'mobilenet':
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        base = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(img_size, img_size, 3))
    elif backbone_name == 'efficientnet':
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input
        base = EfficientNetB0(weights='imagenet', include_top=False,
                              input_shape=(img_size, img_size, 3))
    else:
        raise ValueError(backbone_name)

    train_data, val_data, test_data = build_generators(preprocess_input, img_size)
    print("Class indices:", train_data.class_indices)

    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes,
    )
    class_weights = dict(enumerate(cw))
    print("Class weights:", class_weights)

    base.trainable = False
    model = build_head(base, num_classes=len(CLASS_ORDER))
    loss_fn = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

    callbacks = [
        ModelCheckpoint(str(out_path), monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\n--- Stage 1: head-only training ---")
    model.compile(optimizer=Adam(1e-3), loss=loss_fn, metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data,
              epochs=EPOCHS_HEAD, class_weight=class_weights,
              callbacks=callbacks)

    print(f"\n--- Stage 2: unfreezing top {unfreeze_top_n} layers ---")
    base.trainable = True
    for layer in base.layers[:-unfreeze_top_n]:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-5), loss=loss_fn, metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data,
              epochs=EPOCHS_FINE, class_weight=class_weights,
              callbacks=callbacks)

    print("\n--- Held-out TEST set evaluation ---")
    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    preds = model.predict(test_data, verbose=0).argmax(axis=1)
    true  = test_data.classes
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification report:")
    print(classification_report(true, preds, target_names=CLASS_ORDER))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(true, preds))
    return test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['mobilenet', 'efficientnet'],
                        help='Train only one backbone (default: both).')
    args = parser.parse_args()

    results = {}
    if args.only in (None, 'mobilenet'):
        results['mobilenet'] = train_one(
            'mobilenet', img_size=160, unfreeze_top_n=60,
            out_path=MODEL_DIR / 'daisee_mobilenet_stress.h5',
        )
    if args.only in (None, 'efficientnet'):
        results['efficientnet'] = train_one(
            'efficientnet', img_size=224, unfreeze_top_n=30,
            out_path=MODEL_DIR / 'daisee_efficientnet_stress.h5',
        )

    print("\n=== FINAL RESULTS ===")
    for name, acc in results.items():
        print(f"  {name:<13s} test accuracy = {acc*100:.2f}%")


if __name__ == '__main__':
    main()