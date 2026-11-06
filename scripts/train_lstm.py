"""Train the character-level Keras LSTM on the project's shared corpus."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(7)
    text = (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")
    split = int(len(text) * 0.85)
    train_text = text[:split]
    chars = sorted(set(text))
    to_id = {c: i for i, c in enumerate(chars)}
    ids = np.array([to_id[c] for c in train_text], dtype=np.int32)
    width = 80
    starts = range(0, len(ids) - width, 3)
    x = np.stack([ids[i:i + width] for i in starts])
    y = np.stack([ids[i + 1:i + width + 1] for i in range(0, len(ids) - width, 3)])
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(min(len(x), 2048), seed=7).batch(64).prefetch(tf.data.AUTOTUNE)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(width,)),
        tf.keras.layers.Embedding(len(chars), 64),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dense(len(chars)),
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(ds, epochs=args.epochs, verbose=1)
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    (out / "char_vocab.json").write_text(json.dumps({"chars": chars, "to_id": to_id}), encoding="utf-8")
    model.save_weights(out / "char_lstm.weights.h5")
    print(f"Training characters: {len(train_text)}; held-out characters: {len(text) - split}")
    print(f"Saved Keras model to {out}")


if __name__ == "__main__":
    main()

