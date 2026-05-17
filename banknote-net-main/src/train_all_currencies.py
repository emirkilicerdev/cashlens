"""
Train a single classifier for ALL currencies and denominations in the BankNote-Net dataset.
Outputs:
  - src/trained_models/all_currencies_classifier.h5
  - src/trained_models/all_currencies_classifier.tflite  (mobile-ready)
  - src/trained_models/labels.json                       (class index -> "CURRENCY_DENOMINATION")
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model

OUT_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(OUT_DIR, exist_ok=True)


def build_model(num_classes: int) -> Model:
    inp = Input(shape=(256,))
    x = Dense(256, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inp, outputs=x)


def main():
    parser = argparse.ArgumentParser(description="Train on all currencies.")
    parser.add_argument("--dpath", default="../data/banknote_net.feather")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bsize", type=int, default=256)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_feather(args.dpath)

    # Create combined label: CURRENCY_DENOMINATION (e.g. "TRY_100_1")
    df["label"] = df["Currency"] + "_" + df["Denomination"]

    labels = df["label"].astype("category")
    label_names = list(labels.cat.categories)
    num_classes = len(label_names)
    print(f"Total classes: {num_classes} across {df['Currency'].nunique()} currencies")

    labels_ohe = pd.get_dummies(labels).values
    # Keep only embedding columns (256 floats)
    embeddings = df.iloc[:, :256].values.astype(np.float32)

    # Shuffle
    idx = np.random.permutation(len(embeddings))
    embeddings, labels_ohe = embeddings[idx], labels_ohe[idx]

    model = build_model(num_classes)
    model.summary()

    h5_path = os.path.join(OUT_DIR, "all_currencies_classifier.h5")
    callbacks = [
        ModelCheckpoint(
            filepath=h5_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ]

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.fit(
        x=embeddings,
        y=labels_ohe,
        batch_size=args.bsize,
        epochs=args.epochs,
        validation_split=0.2,
        callbacks=callbacks,
    )

    # Save label mapping
    labels_path = os.path.join(OUT_DIR, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"classes": label_names}, f, indent=2)
    print(f"Labels saved to {labels_path}")

    # Export TFLite for mobile
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic range quantization
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUT_DIR, "all_currencies_classifier.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    print("\nDone! Output files:")
    print(f"  {h5_path}")
    print(f"  {tflite_path}")
    print(f"  {labels_path}")


if __name__ == "__main__":
    main()
