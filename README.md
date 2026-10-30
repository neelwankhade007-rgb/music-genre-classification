# 🎵 Music Genre Classifier (2D-CNN + FastAPI)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio_DSP-2C3E50?style=for-the-badge)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

An end-to-end music genre classification system powered by a **2D Convolutional Neural Network (CNN)** and **Librosa audio feature extraction**. It classifies audio files into **10 distinct genres** with an interactive audio studio frontend.

---

## 🛠️ Tech Stack

### Backend & Audio Processing
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) with [Uvicorn](https://www.uvicorn.org/) ASGI server
- **Deep Learning:** [TensorFlow 2.x](https://www.tensorflow.org/) / Keras
- **Audio DSP:** [Librosa](https://librosa.org/) & [SoundFile](https://python-soundfile.readthedocs.io/) (STFT, Mel filterbanks, dynamic decibel scaling, spectral features)
- **Data & Schemas:** [NumPy](https://numpy.org/), [Pydantic v2](https://docs.pydantic.dev/)

### Frontend (Interactive Studio)
- **Architecture:** Single-Page Application (SPA) served directly from the FastAPI root (`/`)
- **Styling:** [Tailwind CSS](https://tailwindcss.com/) (Studio Dark palette)
- **Icons & Typography:** [Lucide Icons](https://lucide.dev/), Inter & JetBrains Mono
- **Visualization:** Dynamic HTML5 canvas waveform, interactive Mel-spectrogram segment inspector, and confidence distribution charts

### Machine Learning Model
- **Pre-trained Model:** `models/genre_classifier_cnn.keras`
- **Trained Classes (10):** `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

> **Note on Training Notebook:**  
> The CNN model is already fully trained and packaged in `models/genre_classifier_cnn.keras`. You do **not** need to run `classification.ipynb` or retrain the network to use the application.

---

## 🚀 Quick Start Guide

### 1. Clone & Navigate to Project

```bash
git clone <your-repo-url>
cd genre-classification
```

### 2. Set Up Virtual Environment

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Running Backend & Frontend

The FastAPI backend automatically serves the frontend interface (`index.html`) at the root URL.

### Launch Server

```bash
python main.py
```

*Or run with Uvicorn directly:*

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access Application

Once started, open your browser:

| Interface | URL | Description |
|---|---|---|
| **Web Studio** | [http://localhost:8000](http://localhost:8000) | Full UI to upload tracks, view waveforms, inspect spectrograms, and classify genres |
| **API Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI documentation to test endpoints directly |
| **API Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | Status endpoint verifying model and genre labels loading |

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive frontend web application (`index.html`) |
| `POST` | `/predict` | Upload audio (`.wav`, `.mp3`, `.ogg`) $\rightarrow$ Slices into 3s chunks $\rightarrow$ Extracts Log Mel-Spectrograms $\rightarrow$ Returns top genre, soft-voted probabilities, acoustic metrics, and spectrogram images |
| `POST` | `/waveform` | Computes a normalized 200-point amplitude envelope for client-side rendering |
| `GET` | `/health` | Returns health status, model readiness, and the 10 target genre labels |
| `GET` | `/test.wav` | Serves a built-in sample audio file for instant 1-click testing in the UI |

---

## 📂 Project Structure

```text
genre-classification/
├── index.html                           # Single-page interactive web studio
├── main.py                              # FastAPI application & inference engine
├── requirements.txt                     # Python package dependencies
├── MODEL_WORKFLOW_AND_ARCHITECTURE.md   # Detailed model workflow & CNN architecture
├── README.md                            # Project setup & run documentation
├── test.wav                             # Built-in sample audio for testing
├── models/
│   ├── genre_classifier_cnn.keras       # Pre-trained 2D-CNN Keras model
│   └── genre_labels.json                # Ordered list of the 10 genre classes
└── classification.ipynb                 # Reference training notebook (already trained)
```

---

## 🔬 How Inference Works

1. **Audio Loading:** Audio is resampled to $22,050\text{ Hz}$ mono.
2. **Segmentation:** Divided into $3.0\text{-second}$ segments ($66,150$ samples each, up to 10 slices).
3. **Feature Extraction:** Librosa calculates a $128 \times 130$ Log Mel-Spectrogram matrix for each segment.
4. **Batch Prediction:** The CNN predicts class probabilities for each slice.
5. **Ensemble Soft Voting:** Probabilities are averaged across all slices to produce the final genre classification and confidence score.
