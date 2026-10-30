"""
Model Evaluation Script for Music Genre Classification CNN
===========================================================

This script performs a full academic evaluation of the trained 2D-CNN model
using the GTZAN dataset. It applies the exact same preprocessing pipeline
used during training (Log Mel-Spectrograms) and generates:

  1. Classification Report (accuracy, precision, recall, F1, per-genre metrics)
  2. Confusion Matrix (10x10 heatmap saved as PNG)
  3. Evaluation summary saved as JSON

IMPORTANT: This script splits data at the TRACK level before segmenting,
ensuring no segments from the same audio track leak between train and test sets.
This provides a more rigorous evaluation than random segment-level splitting.

Usage:
    python scripts/evaluate_model.py

Output files are saved to the 'evaluation/' directory.
"""

import json
import math
import os
import sys
import warnings
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Use non-interactive backend for headless PNG rendering
matplotlib.use("Agg")

# ==============================================================================
# Configuration (must match training parameters exactly)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "genre_classifier_cnn.keras"
GENRE_LABELS_PATH = BASE_DIR / "models" / "genre_labels.json"
DATASET_PATH = BASE_DIR / "data" / "raw" / "genres_original"
OUTPUT_DIR = BASE_DIR / "evaluation"

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # seconds
NUM_SEGMENTS = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
EXPECTED_FRAMES = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)  # 130
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_genre_labels():
    """Load genre labels from the saved JSON file."""
    with open(GENRE_LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_segments_for_track(file_path, genre_idx):
    """
    Extract all valid Log Mel-Spectrogram segments from a single audio track.
    Returns list of (spectrogram, label) tuples.
    """
    segments = []
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"  [Skipped corrupted audio] {Path(file_path).name}: {e}")
        return segments

    for s in range(NUM_SEGMENTS):
        start = SAMPLES_PER_SEGMENT * s
        finish = start + SAMPLES_PER_SEGMENT
        segment = signal[start:finish]

        mel_spec = librosa.feature.melspectrogram(
            y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] == EXPECTED_FRAMES:
            segments.append((mel_spec_db, genre_idx))

    return segments


def load_dataset_with_track_split():
    """
    Load GTZAN dataset with track-level train/test splitting.

    Splits audio TRACKS into train/test sets BEFORE generating segments,
    ensuring no data leakage between train and test.
    """
    genres = sorted(
        [
            d
            for d in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, d))
        ]
    )
    label_mapping = {genre: idx for idx, genre in enumerate(genres)}

    print(f"Detected {len(genres)} genre classes: {label_mapping}\n")

    # Collect all track file paths and their labels
    all_tracks = []
    all_track_labels = []

    for genre in genres:
        genre_folder = os.path.join(DATASET_PATH, genre)
        audio_files = sorted([f for f in os.listdir(genre_folder) if f.endswith(".wav")])
        print(f"  {genre:<12} {len(audio_files)} tracks")

        for file_name in audio_files:
            file_path = os.path.join(genre_folder, file_name)
            all_tracks.append(file_path)
            all_track_labels.append(label_mapping[genre])

    print(f"\nTotal tracks found: {len(all_tracks)}")

    # Split at the TRACK level (stratified by genre)
    train_tracks, test_tracks, train_labels, test_labels = train_test_split(
        all_tracks,
        all_track_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_track_labels,
    )

    print(f"Track-level split: {len(train_tracks)} train / {len(test_tracks)} test\n")

    # Extract segments from test tracks only (for evaluation)
    print("Extracting mel-spectrogram segments from test tracks...")
    X_test = []
    y_test = []
    skipped = 0

    for track_path, label in zip(test_tracks, test_labels):
        segments = extract_segments_for_track(track_path, label)
        if not segments:
            skipped += 1
            continue
        for spec, lbl in segments:
            X_test.append(spec)
            y_test.append(lbl)

    X_test = np.array(X_test)[..., np.newaxis]
    y_test = np.array(y_test)

    print(f"Test segments extracted: {X_test.shape[0]}")
    if skipped:
        print(f"Tracks skipped (corrupted): {skipped}")

    return X_test, y_test, genres, len(train_tracks), len(test_tracks)


def generate_confusion_matrix_plot(y_true, y_pred, genre_labels, output_path):
    """Generate and save a dark-themed confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    # Use a green-tinted dark colormap
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad(color="#0D1117")

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#8B949E")
    cbar.outline.set_edgecolor("#30363D")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8B949E", fontsize=9)

    # Labels
    tick_marks = np.arange(len(genre_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(
        [g.capitalize() for g in genre_labels],
        rotation=45,
        ha="right",
        fontsize=10,
        color="#C9D1D9",
    )
    ax.set_yticklabels(
        [g.capitalize() for g in genre_labels],
        fontsize=10,
        color="#C9D1D9",
    )

    ax.set_xlabel("Predicted Genre", fontsize=12, color="#C9D1D9", labelpad=12)
    ax.set_ylabel("Actual Genre", fontsize=12, color="#C9D1D9", labelpad=12)
    ax.set_title(
        "Confusion Matrix – GTZAN Genre Classification (Track-Level Test Split)",
        fontsize=13,
        color="#E6EDF3",
        pad=16,
    )

    # Cell annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if val > 0 else "normal",
                color="black" if val > thresh else "#C9D1D9",
            )

    # Grid styling
    ax.tick_params(axis="both", colors="#8B949E", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#0D1117", bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to: {output_path}")

    return cm.tolist()


def main():
    print("=" * 70)
    print("  Music Genre Classification – Model Evaluation")
    print("=" * 70)
    print()

    # Validate paths
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not GENRE_LABELS_PATH.exists():
        print(f"ERROR: Genre labels not found at {GENRE_LABELS_PATH}")
        sys.exit(1)
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading trained CNN model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"Model loaded: {MODEL_PATH.name}\n")

    # Load genre labels
    genre_labels = load_genre_labels()
    print(f"Genre labels: {genre_labels}\n")

    # Load and split dataset
    X_test, y_test, genres, n_train_tracks, n_test_tracks = (
        load_dataset_with_track_split()
    )

    # Run predictions
    print("\nRunning batch inference on test segments...")
    predictions = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Classification Report
    print("\n" + "=" * 70)
    print("  CLASSIFICATION REPORT")
    print("=" * 70)

    report_text = classification_report(
        y_test, y_pred, target_names=[g.capitalize() for g in genre_labels], digits=4
    )
    print(report_text)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=[g.capitalize() for g in genre_labels],
        digits=4,
        output_dict=True,
    )

    # Overall metrics
    overall_accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(y_test, y_pred, average="weighted")
    )

    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"Weighted F1:      {f1_weighted:.4f}")

    # Confusion Matrix
    print("\nGenerating confusion matrix heatmap...")
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    cm_data = generate_confusion_matrix_plot(y_test, y_pred, genre_labels, cm_path)

    # Per-genre metrics
    per_genre = {}
    for genre in genre_labels:
        genre_key = genre.capitalize()
        if genre_key in report_dict:
            per_genre[genre] = {
                "precision": round(report_dict[genre_key]["precision"], 4),
                "recall": round(report_dict[genre_key]["recall"], 4),
                "f1_score": round(report_dict[genre_key]["f1-score"], 4),
                "support": int(report_dict[genre_key]["support"]),
            }

    # Assemble full evaluation results
    evaluation_results = {
        "model_file": MODEL_PATH.name,
        "dataset": "GTZAN",
        "split_method": "track-level stratified (no segment leakage)",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "num_train_tracks": n_train_tracks,
        "num_test_tracks": n_test_tracks,
        "num_test_segments": int(X_test.shape[0]),
        "preprocessing": {
            "sample_rate": SAMPLE_RATE,
            "segment_duration_sec": TRACK_DURATION // NUM_SEGMENTS,
            "segments_per_track": NUM_SEGMENTS,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "input_shape": [N_MELS, EXPECTED_FRAMES, 1],
        },
        "overall_metrics": {
            "accuracy": round(overall_accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(precision_weighted, 4),
            "recall_weighted": round(recall_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
        },
        "per_genre_metrics": per_genre,
        "confusion_matrix": cm_data,
        "genre_labels": genre_labels,
    }

    # Save evaluation JSON
    eval_json_path = OUTPUT_DIR / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to: {eval_json_path}")

    # Save classification report as text
    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Music Genre Classification – Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {MODEL_PATH.name}\n")
        f.write(f"Dataset: GTZAN ({n_test_tracks} test tracks, {X_test.shape[0]} segments)\n")
        f.write(f"Split: Track-level stratified ({TEST_SIZE*100:.0f}% test)\n\n")
        f.write(report_text)
        f.write(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%\n")
    print(f"Classification report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  Accuracy:     {overall_accuracy * 100:.2f}%")
    print(f"  Macro F1:     {f1_macro:.4f}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
