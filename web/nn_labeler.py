#!/usr/bin/env python3
"""
Neural network based character labeler with active learning.
Labels characters one at a time, trains a CNN, then prioritizes uncertain predictions.

Usage: python nn_labeler.py
Then open http://localhost:5000
"""

import sys
import json
import random
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Config
CHAR_DIR = Path("chars")
LABELS_FILE = Path("nn_labels.json")

# Global state
g_char_files: list[Path] = []
g_images: list[np.ndarray] = []
g_labels: dict[str, str] = {}  # relpath -> char
g_model: "CharCNN | None" = None
g_predictions: dict[str, tuple[str, float]] = {}  # relpath -> (char, confidence)
g_training = False


class CharCNN(nn.Module):
    """Simple CNN for character classification."""

    def __init__(self, num_classes: int, input_height: int, input_width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        h, w = input_height, input_width
        for _ in range(3):
            h, w = h // 2, w // 2
        self.flat_size = 64 * h * w

        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flat_size)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def relpath(f: Path) -> str:
    return str(f.relative_to(CHAR_DIR))


def load_images() -> None:
    global g_char_files, g_images
    g_char_files = sorted(CHAR_DIR.rglob("*.png"))

    print(f"Loading {len(g_char_files)} character images...")
    for i, f in enumerate(g_char_files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Failed to load image: {f}"
        g_images.append(img)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1} / {len(g_char_files)}...\r", end="")
            sys.stdout.flush()
    print(f"Loaded {len(g_images)} images.")


def load_labels() -> dict[str, str]:
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")


def save_labels(labels: dict[str, str]) -> None:
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def train_model_sync() -> None:
    global g_model, g_predictions, g_training

    if len(g_labels) < 10:
        print("Need at least 10 labels to train.")
        g_training = False
        return

    # Build char <-> idx mapping
    unique_chars = sorted(set(g_labels.values()))
    char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    num_classes = len(char_to_idx)

    # Build training data
    X, y = [], []
    relpath_to_idx = {relpath(f): i for i, f in enumerate(g_char_files)}

    for rp, char in g_labels.items():
        if rp in relpath_to_idx:
            img = g_images[relpath_to_idx[rp]]
            X.append(img)
            y.append(char_to_idx[char])

    X = np.array(X, dtype=np.float32) / 255.0
    X = X.reshape(-1, 1, X.shape[1], X.shape[2])
    y = np.array(y, dtype=np.int64)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create and train model
    h, w = g_images[0].shape
    g_model = CharCNN(num_classes, h, w)
    optimizer = torch.optim.Adam(g_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    g_model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(g_model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/10, loss: {total_loss / len(loader):.4f}")

    # Run predictions
    print("Training complete. Running predictions...")
    g_model.eval()
    g_predictions = {}

    with torch.no_grad():
        for i, f in enumerate(g_char_files):
            rp = relpath(f)
            if rp in g_labels:
                continue

            img = g_images[i].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            probs = F.softmax(g_model(img_tensor), dim=1)
            confidence, pred_idx = probs.max(dim=1)
            g_predictions[rp] = (idx_to_char[int(pred_idx.item())], float(confidence.item()))

            if i % 10000 == 0:
                print(f"  Predicted {i} / {len(g_char_files)}...\r", end="")
                sys.stdout.flush()

    print(f"Predicted {len(g_predictions)} unlabeled images.")
    g_training = False


def train_model() -> None:
    global g_training
    if g_training:
        return
    g_training = True
    threading.Thread(target=train_model_sync).start()


def get_next_to_label(uncertain: bool = False) -> str | None:
    """Get relpath of next image to label. If uncertain=True, pick lowest confidence."""
    if uncertain and g_predictions:
        unlabeled = [rp for rp in g_predictions if rp not in g_labels]
        if unlabeled:
            return min(unlabeled, key=lambda rp: g_predictions[rp][1])
    return relpath(random.choice(g_char_files))


@app.route("/")
def index():
    return render_template("nn_label.html")


@app.route("/api/next")
def api_next():
    mode = request.args.get("mode", "random")  # "random" or "uncertain"
    rp = get_next_to_label(uncertain=(mode == "uncertain"))
    if rp is None:
        return jsonify({"done": True})

    prediction, confidence = g_predictions.get(rp, (None, 0.0))
    return jsonify({
        "done": False,
        "filename": rp,
        "image_url": f"/image/{rp}",
        "prediction": prediction,
        "confidence": confidence,
        "labeled_count": len(g_labels),
        "total_count": len(g_char_files),
        "model_trained": g_model is not None,
    })


@app.route("/image/<path:rp>")
def serve_image(rp: str):
    return send_file(CHAR_DIR / rp, mimetype="image/png")


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.json
    g_labels[data["filename"]] = data["char"]
    save_labels(g_labels)
    return jsonify({"success": True, "labeled_count": len(g_labels)})


@app.route("/api/undo", methods=["POST"])
def api_undo():
    if not g_labels:
        return jsonify({"success": False, "error": "Nothing to undo"})
    rp, char = g_labels.popitem()
    save_labels(g_labels)
    return jsonify({"success": True, "undone": rp, "char": char, "labeled_count": len(g_labels)})


@app.route("/api/train", methods=["POST"])
def api_train():
    if g_training:
        return jsonify({"success": False, "error": "Already training"})
    print(f"Starting training on {len(g_labels)} labels...")
    train_model()
    return jsonify({"success": True})


@app.route("/api/export")
def api_export():
    threshold = float(request.args.get("threshold", 0.95))
    output = {}

    for rp, char in g_labels.items():
        output[rp] = {"char": char, "source": "human"}

    for rp, (char, conf) in g_predictions.items():
        if conf >= threshold:
            output[rp] = {"char": char, "source": "model", "confidence": conf}

    export_file = Path("nn_export.json")
    with open(export_file, "w") as f:
        json.dump(output, f, indent=2)

    return jsonify({
        "success": True,
        "file": str(export_file),
        "human_labels": len(g_labels),
        "model_predictions": len(output) - len(g_labels),
        "total": len(output),
    })


def main():
    load_images()
    global g_labels
    g_labels = load_labels()

    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, use_reloader=False)


if __name__ == "__main__":
    main()
