#!/usr/bin/env python3
"""
Interactive web-based character labeler for clustering results.
Usage: python app.py output_chars/
Then open http://localhost:5000 in your browser
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import base64
import json

app = Flask(__name__)

# Global state
g_char_files: list[Path] = []
g_images: list[np.ndarray] = []
g_labels: np.ndarray = np.array([], dtype=int)
g_cluster_labels: dict[int, str] = {}  # cluster_id -> character label
g_current_cluster = 0
g_n_clusters = 64


def load_and_cluster(
    char_dir: str,  # Directory containing extracted character images
    n_clusters: int = 64,  # Number of clusters (should match base64 alphabet size)
) -> None:
    """Load all character images and perform clustering."""
    global g_char_files, g_images, g_labels, g_n_clusters, g_kmeans

    char_path = Path(char_dir)
    g_char_files = sorted(char_path.rglob("*.png"))

    print(f"Loading {len(g_char_files)} character images...")
    images = []
    for f in g_char_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        images.append(img.flatten())
        if len(images) % 1000 == 0:
            print(f"Loaded {len(images)} / {len(g_char_files)}...")

    g_images = images
    X = np.array(images)

    print(f"Clustering into {n_clusters} clusters...")
    g_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    g_labels = g_kmeans.fit_predict(X)
    g_n_clusters = n_clusters

    print("Clustering complete!")


def find_next_unlabeled(start_from: int = 0) -> int:
    """Find the next unlabeled cluster, wrapping around if needed."""
    # First try from start_from to end
    for i in range(start_from, g_n_clusters):
        if i not in g_cluster_labels:
            return i
    # Then try from beginning to start_from
    for i in range(0, start_from):
        if i not in g_cluster_labels:
            return i
    # All labeled, just return start_from
    return start_from


def get_cluster_examples(
    cluster_id: int,  # Which cluster to get examples from
    n_examples: int = 10,  # Number of example images to return
) -> list[str]:
    """Get example images from a cluster as base64-encoded PNGs."""
    indices = np.where(g_labels == cluster_id)[0]

    # Sample up to n_examples
    sample_size = min(n_examples, len(indices))
    if sample_size == 0:
        return []
    sampled_indices = np.random.choice(indices, sample_size, replace=False)

    examples = []
    for idx in sampled_indices:
        img_path = g_char_files[idx]
        img = cv2.imread(str(img_path))

        # Encode as PNG then base64 for web display
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        examples.append(f"data:image/png;base64,{img_base64}")

    return examples


@app.route('/')
def index():
    """Main labeling interface."""
    return render_template('label.html')


@app.route('/api/cluster_info')
def cluster_info():
    """Get information about current cluster."""
    cluster_id = g_current_cluster
    examples = get_cluster_examples(cluster_id, n_examples=10)

    # Count how many chars in this cluster
    count = np.sum(g_labels == cluster_id)

    return jsonify({
        'cluster_id': cluster_id,
        'total_clusters': g_n_clusters,
        'examples': examples,
        'count': int(count),
        'current_label': g_cluster_labels.get(cluster_id, ''),
        'labeled_count': len(g_cluster_labels),
    })


@app.route('/api/label_cluster', methods=['POST'])
def label_cluster():
    """Label the current cluster."""
    global g_current_cluster

    data = request.json
    label = data.get('label', '')

    cluster_id = g_current_cluster
    g_cluster_labels[cluster_id] = label

    # Move to next unlabeled cluster
    g_current_cluster = find_next_unlabeled(cluster_id + 1)

    return jsonify({'success': True})


@app.route('/api/skip_cluster', methods=['POST'])
def skip_cluster():
    """Skip to next unlabeled cluster."""
    global g_current_cluster

    # Find next unlabeled, starting after current
    g_current_cluster = find_next_unlabeled(g_current_cluster + 1)
    return jsonify({'success': True})


@app.route('/api/goto_cluster', methods=['POST'])
def goto_cluster():
    """Jump to a specific cluster."""
    global g_current_cluster

    data = request.json
    cluster_id = data.get('cluster_id', 0)

    if 0 <= cluster_id < g_n_clusters:
        g_current_cluster = cluster_id
        return jsonify({'success': True})

    return jsonify({'success': False, 'error': 'Invalid cluster ID'})


@app.route('/api/split_cluster', methods=['POST'])
def split_cluster():
    """Split a cluster into sub-clusters (for when clustering merged distinct chars)."""
    global g_labels, g_n_clusters, g_cluster_labels

    data = request.json
    cluster_id = data.get('cluster_id', g_current_cluster)
    n_splits = data.get('n_splits', 2)

    # Find all indices belonging to this cluster
    indices = np.where(g_labels == cluster_id)[0]
    if len(indices) < n_splits:
        return jsonify({'success': False, 'error': 'Not enough samples to split'})

    # Get the image data for just this cluster
    cluster_images = np.array([g_images[i] for i in indices])

    # Re-cluster into n_splits
    sub_kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
    sub_labels = sub_kmeans.fit_predict(cluster_images)

    # First sub-cluster keeps the original ID, others get new IDs
    new_cluster_ids = [cluster_id] + list(range(g_n_clusters, g_n_clusters + n_splits - 1))

    # Update labels
    for i, idx in enumerate(indices):
        g_labels[idx] = new_cluster_ids[sub_labels[i]]

    # Update cluster count
    g_n_clusters += (n_splits - 1)

    # Remove label for the split cluster (it's now different)
    if cluster_id in g_cluster_labels:
        del g_cluster_labels[cluster_id]

    return jsonify({
        'success': True,
        'new_cluster_ids': new_cluster_ids,
        'new_total_clusters': g_n_clusters,
    })


@app.route('/api/save_labels', methods=['POST'])
def save_labels():
    """Save cluster labels to JSON file."""
    output_file = 'cluster_labels.json'

    with open(output_file, 'w') as f:
        json.dump(g_cluster_labels, f, indent=2)

    return jsonify({'success': True, 'file': output_file})


@app.route('/api/export_labeled_data')
def export_labeled_data():
    """Export full labeled dataset (cluster assignments + labels)."""
    output = []

    for idx, (char_file, cluster_id) in enumerate(zip(g_char_files, g_labels)):
        label = g_cluster_labels.get(cluster_id, None)
        if label:  # Only export if cluster is labeled
            output.append({
                'file': str(char_file.name),
                'cluster': int(cluster_id),
                'label': label,
            })

    output_file = 'labeled_characters.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    return jsonify({
        'success': True,
        'file': output_file,
        'count': len(output),
    })


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python app.py <char_directory>")
        sys.exit(1)

    char_dir = sys.argv[1]
    load_and_cluster(char_dir)

    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser to begin labeling")

    app.run(debug=True, port=5000)


if __name__ == '__main__':
    main()
