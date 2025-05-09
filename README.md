# 🎬 Multi-Modal Genre Classification System

This is a Flask-based web application for **multi-label movie genre classification** using both **textual (movie plot)** and **visual (poster image)** inputs. It combines LSTM and ResNet-based models trained on a balanced TMDB dataset across the top 10 genres.

🔗 **Live Demo 1**: [Launch Website](https://myapp-284861369113.us-central1.run.app/)
🔗 **Live Demo 2**: [Launch Website](https://movie-genre-classification-sys.onrender.com)


📦 **GitHub Repository**: [ujwalwag/Movie-Genre-Classification-Sys](https://github.com/ujwalwag/Movie-Genre-Classification-Sys.git)

---

## 🧠 Features

- 🔤 **Text-based Genre Classification** using LSTM and GloVe embeddings (input: movie plot)
- 🖼️ **Image-based Genre Classification** using ResNet-18 (input: movie poster)
- 🔀 Unified **multi-modal predictions** for accurate multi-label genre output
- 🌐 Web interface to enter a plot and/or upload a poster
- 📊 Visual genre probability predictions
- 🐳 Docker support for containerized deployment

---

## 🗂️ Repository Structure

```
.
├── app.py                      # Flask app entry point
├── models/├── tokenizer.json   # Contains tokenizer, weights, embedding matrix,# Saved tokenizer for inference
├── templates/                  # HTML templates (Jinja2)
├── static/                     # Static assets like CSS, JS
├── scripts/                    # Training scripts
├── notebook/                   # Jupyter notebooks for analysis
├── plots/                      # Model evaluation plots
├── webapp/                     # Web deployment configs
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container instructions
```

---

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `pip`
- Git
- tensorflow
- torchvisiob
- keras
- Docker(for web deployment)

---

### 🔧 Installation

```bash
git clone https://github.com/ujwalwag/Movie-Genre-Classification-Sys.git
cd ujwalwag/Movie-Genre-Classification-Sys
```

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 🧪 Run Locally

```bash
python app.py
```

Open your browser and go to `http://127.0.0.1:5000`

---

### 🐳 Run with Docker

```bash
docker build -t movie-genre-classifier .
docker run -p 5000:5000 movie-genre-classifier
```

---

### 💡 How to Use

1. Visit the web app.
2. Enter a **movie plot (overview)** to predict based on text.
3. Upload a **poster image** to predict based on movie postre.
4. Click **Predict** to get genre predictions from:
   - The **Text model**, the **Image model**.

---

## 🧠 Model Overview

### 🔤 Text Model (LSTM)

- Pre-trained **GloVe (100d)** embeddings
- Tokenizer saved in `tokenizer.json`
- BiLSTM → Mean Pooling → Dense layers
- Uses `embedding_matrix.npy`

### 🖼️ Image Model (ResNet-18)

- Based on torchvision’s pretrained ResNet-18
- Final FC layer replaced for 10-class sigmoid output
- Poster inputs are normalized and resized

---

## 📈 Supported Genres

- Drama
- Comedy
- Romance
- Thriller
- Action
- Horror
- Documentary
- Animation
- Music
- Crime

---

## 🤝 Acknowledgements

- EAS 510 Basics of AI lectures from UB
- TMDB Dataset
- GloVe Embeddings (Stanford)
- PyTorch, Flask, torchvision

---
