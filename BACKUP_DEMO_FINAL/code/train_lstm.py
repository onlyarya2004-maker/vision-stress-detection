"""
Train CNN+LSTM Temporal Stress Classifier
=========================================
Architecture:
  Input: (16, 96, 96, 3)  -- 16 frames per clip, RGB
       │
       ▼  TimeDistributed(MobileNetV2 frozen)
  Per-frame features: (16, 1280)  -- 1280-d feature vec per frame
       │
       ▼  TimeDistributed(GAP) -> already pooled inside MobileNet head
  Sequence: (16, 1280)
       │
       ▼  LSTM(64, return_sequences=False)
  Temporal embedding: (64,)
       │
       ▼  Dense(32, relu) -> Dropout(0.3) -> Dense(2, softmax)
  Output: P(not_stressed), P(stressed)

Why frozen MobileNet?
  We have ~6500 training sequences. That's not enough to fine-tune a
  CNN backbone end-to-end without overfitting. Using frozen MobileNet
  pretrained on ImageNet, we only train the LSTM + dense head (~50K
  params), which fits the data size and trains in 8-15 hours on CPU.

Why 96x96 input?
  MobileNetV2 supports 96x96 as a smaller native resolution. This cuts
  per-frame feature extraction time ~5x vs the 224x224 we use for
  single-frame DaiSEE. Critical for CPU training.

Outputs:
  ../models/lstm_stress.h5  -- the trained model
  ../models/lstm_test_report.txt -- evaluation summary

Run AFTER lstm_preprocessing.py.

Estimated training time: 18-25 hours on CPU. Run before bed.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input as mobilenet_preprocess
)
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, TimeDistributed,
    GlobalAveragePooling2D, BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)
from tensorflow.keras.utils import to_categorical, Sequence
import cv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
INDEX_PATH    = Path('../datasets/processed_daisee/lstm_index.pkl')
MODEL_OUT     = Path('../models/lstm_stress.h5')
REPORT_OUT    = Path('../models/lstm_test_report.txt')
CSV_LOG_OUT   = Path('../models/lstm_training_log.csv')
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

SEQ_LEN     = 16
IMG_SIZE    = 96
BATCH_SIZE  = 4          # SMALL - sequences are 16x larger than single frames
EPOCHS      = 25
LABEL_SMOOTH = 0.1


# ---------------- DATA LOADER ----------------
class SequenceDataLoader(Sequence):
    """
    Loads (batch_size, 16, 96, 96, 3) batches from disk on demand.
    Reading 16 frames per sample x batch_size is heavy I/O, so we use
    a small batch and let TF parallelize where possible.
    """

    def __init__(self, sequences, batch_size, shuffle=True, augment=False):
        self.sequences = sequences
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(sequences))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, batch_idx):
        batch_indices = self.indices[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        X = np.zeros((len(batch_indices), SEQ_LEN, IMG_SIZE, IMG_SIZE, 3),
                     dtype='float32')
        y = np.zeros(len(batch_indices), dtype='int32')

        for i, seq_idx in enumerate(batch_indices):
            clip_id, label, frame_paths = self.sequences[seq_idx]
            y[i] = label
            for t, fp in enumerate(frame_paths[:SEQ_LEN]):
                img = cv2.imread(fp)
                if img is None:
                    # Fall back to zeros for missing frames - shouldn't happen
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X[i, t] = img.astype('float32')

            if self.augment:
                # Apply same horizontal flip to whole sequence (preserve temporal coherence)
                if np.random.rand() < 0.5:
                    X[i] = X[i, :, :, ::-1, :]

        # MobileNet preprocessing (handles all frames uniformly)
        X = mobilenet_preprocess(X)
        Y = to_categorical(y, num_classes=2)
        return X, Y


# ---------------- MODEL ----------------
def build_lstm_model():
    inputs = Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))

    # Frozen MobileNet feature extractor
    cnn_base = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    cnn_base.trainable = False
    cnn_features = Model(cnn_base.input, GlobalAveragePooling2D()(cnn_base.output),
                         name='mobilenet_features')

    # Apply CNN to each timestep
    x = TimeDistributed(cnn_features, name='td_cnn')(inputs)
    x = BatchNormalization(name='bn1')(x)

    # Temporal modeling
    x = LSTM(64, dropout=0.3, recurrent_dropout=0.0,
             return_sequences=False, name='lstm')(x)

    # Classification head
    x = Dense(32, activation='relu', name='fc1')(x)
    x = Dropout(0.3, name='drop1')(x)
    outputs = Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='cnn_lstm_stress')
    return model


# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Train on small subset for 2 epochs to verify pipeline')
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        raise SystemExit(f"Cannot find {INDEX_PATH}. Run lstm_preprocessing.py first.")

    print(f"Loading sequence index from {INDEX_PATH}...")
    with open(INDEX_PATH, 'rb') as f:
        index = pickle.load(f)
    train_seqs = index['train']
    val_seqs   = index['val']
    test_seqs  = index['test']
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}")

    if args.smoke_test:
        print("\n[smoke-test mode] Using small subsets, 2 epochs only")
        train_seqs = train_seqs[:50]
        val_seqs   = val_seqs[:20]
        args.epochs = 2

    # Class weights (DaiSEE is imbalanced - more not_stressed than stressed)
    train_labels = np.array([s[1] for s in train_seqs])
    cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(cw))
    print(f"\nClass weights: {class_weights}")

    train_loader = SequenceDataLoader(train_seqs, args.batch_size,
                                       shuffle=True, augment=True)
    val_loader   = SequenceDataLoader(val_seqs,   args.batch_size,
                                       shuffle=False, augment=False)

    print("\nBuilding model...")
    model = build_lstm_model()
    model.compile(
        optimizer=Adam(1e-3),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy'],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(str(MODEL_OUT), monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-7, verbose=1),
        CSVLogger(str(CSV_LOG_OUT), append=False),
    ]

    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"NOTE: each epoch processes {len(train_seqs)} sequences * 16 frames each.")
    print(f"      On CPU this takes ~30-60 minutes per epoch. EarlyStopping will")
    print(f"      cut training short once val_accuracy plateaus.\n")

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------- TEST EVALUATION ----------------
    print("\n" + "=" * 60)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 60)
    test_loader = SequenceDataLoader(test_seqs, args.batch_size,
                                      shuffle=False, augment=False)

    test_loss, test_acc = model.evaluate(test_loader, verbose=1)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    print(f"Test loss:     {test_loss:.4f}")

    # Detailed report
    print("\nGenerating classification report...")
    preds = model.predict(test_loader, verbose=1)
    y_pred = preds.argmax(axis=1)
    y_true = np.array([s[1] for s in test_seqs])

    report = classification_report(
        y_true, y_pred,
        target_names=['not_stressed', 'stressed'], digits=3
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save report file
    with open(REPORT_OUT, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LSTM STRESS CLASSIFIER - TEST EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test set size: {len(test_seqs)} sequences\n")
        f.write(f"Test accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Test loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(cm) + "\n")

    print(f"\nReport saved to: {REPORT_OUT}")
    print(f"Model saved to:  {MODEL_OUT}")
    print(f"Training log:    {CSV_LOG_OUT}")


if __name__ == '__main__':
    main()
