#!/usr/bin/env python3
"""
Neural network based character labeler with active learning.
Labels characters one at a time, trains a CNN, then prioritizes uncertain predictions.

Usage: python nn_labeler.py ../output_chars/
Then open http://localhost:5000
"""

import sys
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Base64 alphabet for reference
BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

# Global state
g_char_files: list[Path] = []
g_images: list[np.ndarray] = []  # Raw images (H, W) grayscale
g_labels: dict[str, str] = {}  # filename -> character label
g_model: "CharCNN | None" = None
g_predictions: dict[str, tuple[str, float]] = {}  # filename -> (predicted_char, confidence)
g_char_to_idx: dict[str, int] = {}
g_idx_to_char: dict[int, str] = {}
g_labels_file = Path("nn_labels.json")


class CharCNN(nn.Module):
    """Simple CNN for character classification."""

    def __init__(self, num_classes: int, input_height: int, input_width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calculate flattened size after convolutions
        h, w = input_height, input_width
        for _ in range(3):  # 3 pooling layers
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


def load_images(char_dir: str) -> None:
    """Load all character images from directory."""
    global g_char_files, g_images

    char_path = Path(char_dir)
    g_char_files = sorted(char_path.rglob("*.png"))

    print(f"Loading {len(g_char_files)} character images...")
    for i, f in enumerate(g_char_files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Failed to load image: {f}"
        g_images.append(img)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1} / {len(g_char_files)}...\r", end="")
            sys.stdout.flush()
    print(f"Loaded {len(g_images)} images.")


def load_existing_labels() -> None:
    """Load any existing labels from disk."""
    global g_labels
    if g_labels_file.exists():
        with open(g_labels_file) as f:
            g_labels = json.load(f)
        print(f"Loaded {len(g_labels)} existing labels.")


def save_labels() -> None:
    """Save labels to disk."""
    with open(g_labels_file, "w") as f:
        json.dump(g_labels, f, indent=2)


def build_char_mapping() -> None:
    """Build character <-> index mapping from current labels."""
    global g_char_to_idx, g_idx_to_char
    unique_chars = sorted(set(g_labels.values()))
    g_char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    g_idx_to_char = {i: c for c, i in g_char_to_idx.items()}


def train_model() -> None:
    """Train the CNN on current labels."""
    global g_model, g_predictions

    if len(g_labels) < 10:
        print("Need at least 10 labels to train.")
        return

    build_char_mapping()
    num_classes = len(g_char_to_idx)

    # Build training data
    X, y = [], []
    filename_to_idx = {f.name: i for i, f in enumerate(g_char_files)}

    for filename, char in g_labels.items():
        if filename in filename_to_idx:
            idx = filename_to_idx[filename]
            img = g_images[idx]
            X.append(img)
            y.append(g_char_to_idx[char])

    X = np.array(X, dtype=np.float32) / 255.0
    X = X.reshape(-1, 1, X.shape[1], X.shape[2])  # Add channel dim
    y = np.array(y, dtype=np.int64)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    h, w = g_images[0].shape
    g_model = CharCNN(num_classes, h, w)

    optimizer = torch.optim.Adam(g_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    g_model.train()
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = g_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}, loss: {total_loss / len(loader):.4f}")

    print("Training complete. Running predictions...")
    predict_all()


def predict_all() -> None:
    """Run predictions on all unlabeled images."""
    global g_predictions

    if g_model is None:
        return

    g_model.eval()
    g_predictions = {}

    with torch.no_grad():
        for i, f in enumerate(g_char_files):
            if f.name in g_labels:
                continue  # Skip already labeled

            img = g_images[i].astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

            outputs = g_model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_idx = probs.max(dim=1)

            pred_char = g_idx_to_char[pred_idx.item()]
            g_predictions[f.name] = (pred_char, confidence.item())

    print(f"Predicted {len(g_predictions)} unlabeled images.")


def get_next_to_label() -> Path | None:
    """Get the next image to label, prioritizing low confidence predictions."""
    unlabeled = [f for f in g_char_files if f.name not in g_labels]

    if not unlabeled:
        return None

    if g_predictions:
        # Sort by confidence (ascending = least confident first)
        unlabeled_with_conf = [
            (f, g_predictions.get(f.name, (None, 0.5))[1]) for f in unlabeled
        ]
        unlabeled_with_conf.sort(key=lambda x: x[1])
        return unlabeled_with_conf[0][0]
    else:
        # No predictions yet, return random
        return random.choice(unlabeled)


@app.route("/")
def index():
    return render_template("nn_label.html")


@app.route("/api/next")
def api_next():
    """Get next image to label."""
    next_file = get_next_to_label()

    if next_file is None:
        return jsonify({"done": True})

    prediction = g_predictions.get(next_file.name)

    return jsonify({
        "done": False,
        "filename": next_file.name,
        "image_url": f"/image/{next_file.parent.name}/{next_file.name}",
        "prediction": prediction[0] if prediction else None,
        "confidence": prediction[1] if prediction else None,
        "labeled_count": len(g_labels),
        "total_count": len(g_char_files),
        "model_trained": g_model is not None,
    })


@app.route("/image/<path:filename>")
def serve_image(filename: str):
    """Serve a character image directly."""
    return send_file('chars/' + filename, mimetype="image/png")


@app.route("/api/label", methods=["POST"])
def api_label():
    """Label an image."""
    data = request.json
    filename = data["filename"]
    char = data["char"]

    g_labels[filename] = char
    save_labels()

    return jsonify({"success": True, "labeled_count": len(g_labels)})


@app.route("/api/train", methods=["POST"])
def api_train():
    """Train the model on current labels."""
    print(f"Training on {len(g_labels)} labels...")
    train_model()
    return jsonify({
        "success": True,
        "num_classes": len(g_char_to_idx),
        "predictions_count": len(g_predictions),
    })


@app.route("/api/stats")
def api_stats():
    """Get labeling statistics."""
    # Count predictions by confidence bucket
    conf_buckets = {"high": 0, "medium": 0, "low": 0}
    for _, (_, conf) in g_predictions.items():
        if conf > 0.9:
            conf_buckets["high"] += 1
        elif conf > 0.7:
            conf_buckets["medium"] += 1
        else:
            conf_buckets["low"] += 1

    return jsonify({
        "labeled_count": len(g_labels),
        "total_count": len(g_char_files),
        "model_trained": g_model is not None,
        "unique_chars": len(g_char_to_idx),
        "confidence_buckets": conf_buckets,
    })


@app.route("/api/export")
def api_export():
    """Export all labels (including model predictions above threshold)."""
    threshold = float(request.args.get("threshold", 0.95))

    output = {}

    # Add human labels
    for filename, char in g_labels.items():
        output[filename] = {"char": char, "source": "human"}

    # Add high-confidence predictions
    for filename, (char, conf) in g_predictions.items():
        if conf >= threshold:
            output[filename] = {"char": char, "source": "model", "confidence": conf}

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
    char_dir = 'chars/'
    load_images(char_dir)
    load_existing_labels()

    if len(g_labels) >= 10:
        print("Found existing labels, training model...")
        train_model()

    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")

    app.run(debug=True, port=5000, use_reloader=False)


if __name__ == "__main__":
    main()
