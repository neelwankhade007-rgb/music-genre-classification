# ==============================================================================
# Requirements (save as requirements.txt or install via pip):
# ==============================================================================
# fastapi>=0.110.0
# uvicorn[standard]>=0.28.0
# python-multipart>=0.0.9
# tensorflow>=2.15.0
# librosa>=0.10.0
# soundfile>=0.12.1
# numpy>=1.24.0
# pydantic>=2.0.0
# ==============================================================================

import base64
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import matplotlib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("genre-classifier")

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "genre_classifier_cnn.keras"
GENRE_LABELS_PATH = BASE_DIR / "models" / "genre_labels.json"

# Audio processing constants (must match training parameters)
SAMPLE_RATE = 22050
DURATION_SECONDS = 3
SAMPLES_PER_SEGMENT = SAMPLE_RATE * DURATION_SECONDS  # 66,150 samples
MAX_SEGMENTS = 10
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_FRAMES = 130  # Expected time bins for 3-second audio: 1 + (66150 // 512) = 130

# Global variables for loaded model and labels
model: Optional[tf.keras.Model] = None
genre_labels: List[str] = []


def compute_waveform_envelope(signal: np.ndarray, num_points: int = 200) -> List[float]:
    """
    Computes a normalized amplitude envelope of the audio signal
    by dividing the signal into evenly spaced windows and extracting peak absolute amplitudes.
    Returns a list of num_points floats in range [0.0, 1.0].
    """
    if len(signal) == 0:
        return [0.0] * num_points
    splits = np.array_split(signal, num_points)
    envelope = [float(np.max(np.abs(split))) if len(split) > 0 else 0.0 for split in splits]
    max_val = max(envelope) if envelope else 0.0
    if max_val > 1e-6:
        return [round(float(v / max_val), 4) for v in envelope]
    return [0.0] * num_points


def generate_spectrogram_image_data(mel_db: np.ndarray) -> str:
    """
    Renders a 128x130 Log Mel-Spectrogram matrix to a base64 PNG data URL using the magma colormap.
    Flips the frequency axis so low frequencies are at the bottom and high frequencies are at the top.
    """
    min_db = float(mel_db.min())
    max_db = float(mel_db.max())
    norm = (mel_db - min_db) / (max_db - min_db + 1e-6)
    norm_flipped = np.flipud(norm)

    cmap = matplotlib.colormaps["magma"]
    rgba = (cmap(norm_flipped) * 255).astype(np.uint8)
    img = Image.fromarray(rgba)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ------------------------------------------------------------------------------
# Pydantic Schemas for Swagger Documentation & Response Validation
# ------------------------------------------------------------------------------
class AudioFeatures(BaseModel):
    spectral_centroid_hz: float
    rms_energy: float
    zero_crossing_rate: float
    duration_seconds: float
    sample_rate_hz: int
    num_segments_analyzed: int


class SpectrogramSegment(BaseModel):
    segment_index: int
    label: str
    start_time: float
    end_time: float
    image_data: str


class PredictionResponse(BaseModel):
    filename: str
    predicted_genre: str
    confidence: float
    all_probabilities: Dict[str, float]
    waveform: Optional[List[float]] = None
    audio_features: Optional[AudioFeatures] = None
    spectrogram_segments: Optional[List[SpectrogramSegment]] = None


class WaveformResponse(BaseModel):
    filename: str
    duration: float
    num_points: int
    waveform: List[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_genres: int
    genres: List[str]


# ------------------------------------------------------------------------------
# Application Lifespan (Startup & Shutdown)
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the model and genre labels on startup.
    """
    global model, genre_labels

    logger.info("Starting up Music Genre Classification Service...")

    # 1. Load genre labels
    if not GENRE_LABELS_PATH.exists():
        logger.error(f"Genre labels file not found at {GENRE_LABELS_PATH}")
        raise FileNotFoundError(f"Missing genre labels file: {GENRE_LABELS_PATH}")

    try:
        with open(GENRE_LABELS_PATH, "r", encoding="utf-8") as f:
            genre_labels = json.load(f)
        logger.info(f"Loaded {len(genre_labels)} genre labels: {genre_labels}")
    except Exception as e:
        logger.error(f"Failed to read genre labels: {e}")
        raise RuntimeError(f"Error loading genre labels: {e}") from e

    # 2. Load trained Keras model
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Missing trained model file: {MODEL_PATH}")

    try:
        logger.info(f"Loading Keras model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info("Keras model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Keras model: {e}")
        raise RuntimeError(f"Error loading model: {e}") from e

    yield

    # Clean up resources on shutdown if necessary
    logger.info("Shutting down Music Genre Classification Service...")


# ------------------------------------------------------------------------------
# FastAPI App Initialization & CORS Configuration
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Music Genre Classification API",
    description="FastAPI service for predicting music genres using a 2D-CNN trained on GTZAN Mel-Spectrograms.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS to allow all origins for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    """
    Serve index.html at root if available, otherwise return health status.
    """
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(index_path)
    return {"status": "healthy", "service": "Music Genre Classifier"}


@app.get("/test.wav", include_in_schema=False)
async def serve_sample_audio():
    """
    Serve sample test.wav for quick UI demonstration.
    """
    test_path = BASE_DIR / "test.wav"
    if test_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(test_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Sample audio file not found.")


@app.get("/evaluation", tags=["Evaluation"])
async def get_evaluation_results():
    """
    Serve pre-generated model evaluation results (classification report,
    confusion matrix data, per-genre metrics). Results are generated
    offline by running: python scripts/evaluate_model.py
    """
    eval_path = BASE_DIR / "evaluation" / "evaluation_results.json"
    if not eval_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run 'python scripts/evaluate_model.py' to generate them.",
        )
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading evaluation results: {str(exc)}",
        ) from exc


@app.get("/evaluation/confusion-matrix", include_in_schema=False)
async def get_confusion_matrix_image():
    """
    Serve the confusion matrix heatmap PNG generated by the evaluation script.
    """
    cm_path = BASE_DIR / "evaluation" / "confusion_matrix.png"
    if cm_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(cm_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Confusion matrix image not found.")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check the health status of the API and verify that the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        num_genres=len(genre_labels),
        genres=genre_labels,
    )


@app.post(
    "/waveform",
    response_model=WaveformResponse,
    summary="Extract Audio Waveform",
    tags=["Audio"],
)
async def extract_waveform(file: UploadFile = File(...)):
    """
    Accepts an audio file upload and returns a compact normalized amplitude
    envelope (200 points) computed via librosa.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload.",
        )
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded audio file is empty.",
            )
        audio_stream = io.BytesIO(file_bytes)
        signal, sr = librosa.load(audio_stream, sr=SAMPLE_RATE)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error loading audio file for waveform: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to process audio file for waveform: {str(exc)}",
        ) from exc

    duration = float(len(signal) / sr)
    waveform = compute_waveform_envelope(signal, num_points=200)

    return WaveformResponse(
        filename=file.filename,
        duration=round(duration, 2),
        num_points=len(waveform),
        waveform=waveform,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify Music Genre",
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, slices it into 3-second segments,
    extracts Log Mel-Spectrograms, runs batch inference, and returns
    soft-voted genre probabilities.
    """
    # Verify that model and genre labels are loaded
    if model is None or not genre_labels:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or genre labels are not loaded. Service is unavailable.",
        )

    # Validate uploaded file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload.",
        )

    try:
        # Read uploaded file content into memory
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded audio file is empty.",
            )

        # Load audio using BytesIO and librosa at 22050 Hz
        audio_stream = io.BytesIO(file_bytes)
        signal, sr = librosa.load(audio_stream, sr=SAMPLE_RATE)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error loading audio file '{file.filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to process audio file '{file.filename}'. Ensure it is a valid audio format (e.g., WAV, MP3, OGG). Error: {str(exc)}",
        ) from exc

    # Slicing: calculate number of available 3-second segments (up to 10)
    num_slices = min(MAX_SEGMENTS, len(signal) // SAMPLES_PER_SEGMENT)

    if num_slices == 0:
        duration_sec = len(signal) / SAMPLE_RATE
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Audio file is too short ({duration_sec:.2f}s). "
                f"At least {DURATION_SECONDS} seconds of audio is required for classification."
            ),
        )

    slices: List[np.ndarray] = []
    spectrogram_segments: List[SpectrogramSegment] = []

    # Process each 3-second segment to Log Mel-Spectrogram
    for s in range(num_slices):
        start = s * SAMPLES_PER_SEGMENT
        segment = signal[start : start + SAMPLES_PER_SEGMENT]

        # Compute Mel-Spectrogram
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

        # Convert to Decibels (Log Mel-Spectrogram)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Verify time dimension matches expected input (128, 130)
        if mel_db.shape[1] == TARGET_FRAMES:
            target_slice = mel_db
        elif mel_db.shape[1] > TARGET_FRAMES:
            # Trim if slightly larger due to boundary precision
            target_slice = mel_db[:, :TARGET_FRAMES]
        elif mel_db.shape[1] < TARGET_FRAMES:
            # Pad if slightly shorter
            pad_width = TARGET_FRAMES - mel_db.shape[1]
            target_slice = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")

        slices.append(target_slice)

        # Extract real Mel-Spectrogram image for this 3-second CNN segment
        start_sec = round(s * DURATION_SECONDS, 1)
        end_sec = round((s + 1) * DURATION_SECONDS, 1)
        spectrogram_segments.append(
            SpectrogramSegment(
                segment_index=s,
                label=f"Segment {s + 1} ({start_sec:.1f}s – {end_sec:.1f}s)",
                start_time=start_sec,
                end_time=end_sec,
                image_data=generate_spectrogram_image_data(target_slice),
            )
        )

    if not slices:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to generate valid mel-spectrogram slices from the audio.",
        )

    # Convert slices to batch input format: (Batch, 128, 130, 1)
    # Using np.newaxis to add the channel dimension
    X_input = np.array(slices)[..., np.newaxis]

    try:
        # Run inference on all segments
        predictions = model.predict(X_input, verbose=0)

        # Soft voting: average predicted probabilities across all segments
        avg_probs = predictions.mean(axis=0)

        # Determine top prediction
        top_index = int(np.argmax(avg_probs))
        predicted_genre = genre_labels[top_index]
        confidence = float(avg_probs[top_index])

        # Map each genre to its probability as standard Python float
        all_probabilities = {
            genre: float(avg_probs[i])
            for i, genre in enumerate(genre_labels)
        }

        # Compute normalized amplitude waveform envelope (200 points)
        waveform_envelope = compute_waveform_envelope(signal, num_points=200)

        # Compute supplementary acoustic features via librosa
        spectral_centroid_hz = round(float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=SAMPLE_RATE))), 1)
        rms_energy = round(float(np.mean(librosa.feature.rms(y=signal))), 4)
        zcr = round(float(np.mean(librosa.feature.zero_crossing_rate(y=signal))), 4)
        duration_sec = round(float(len(signal) / SAMPLE_RATE), 2)

        audio_features = AudioFeatures(
            spectral_centroid_hz=spectral_centroid_hz,
            rms_energy=rms_energy,
            zero_crossing_rate=zcr,
            duration_seconds=duration_sec,
            sample_rate_hz=SAMPLE_RATE,
            num_segments_analyzed=num_slices,
        )

        return PredictionResponse(
            filename=file.filename,
            predicted_genre=predicted_genre,
            confidence=confidence,
            all_probabilities=all_probabilities,
            waveform=waveform_envelope,
            audio_features=audio_features,
            spectrogram_segments=spectrogram_segments,
        )

    except Exception as exc:
        logger.error(f"Inference error on file '{file.filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during model prediction: {str(exc)}",
        ) from exc


# ------------------------------------------------------------------------------
# Server Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
