import os
import pandas as pd
import numpy as np
import requests
import random
import ast
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import classification_report

# ───────────── Config ─────────────
CSV_PATH = "data/strictly_balanced_top10_cleaned.csv"
SAVE_DIR = "downloaded_posters"
MODEL_PATH = "models/poster_genre_classifier.pth"
SAMPLE_SIZE = 3000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENRE_COLUMNS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
    'Horror', 'Documentary', 'Animation', 'Music', 'Crime'
]

TMDB_BASE_URL = "https://image.tmdb.org/t/p/w342"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
}

# ───────────── Image Transform ─────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ───────────── Load and Prepare Data ─────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["poster_path"])
df['genres'] = df['genres'].apply(ast.literal_eval)
df['filename'] = df['poster_path'].apply(lambda x: os.path.basename(str(x).strip("/")))
sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Create poster folder
os.makedirs(SAVE_DIR, exist_ok=True)

# ───────────── Download Posters ─────────────
print(f"📥 Downloading {SAMPLE_SIZE} posters...")
download_errors = []

for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Downloading posters"):
    poster_path = row['poster_path']
    url = f"{TMDB_BASE_URL}{poster_path}"
    save_path = os.path.join(SAVE_DIR, os.path.basename(poster_path))

    if os.path.exists(save_path):
        continue  # already exists

    try:
        response = requests.get(url, headers=HEADERS, timeout=10, verify=False)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        download_errors.append((url, str(e)))

print(f"✅ Poster download complete. {len(download_errors)} errors.\n")

# ───────────── Encode Labels ─────────────
def encode_labels(genres):
    return [1 if g in genres else 0 for g in GENRE_COLUMNS]

sampled_df['binary'] = sampled_df['genres'].apply(encode_labels)

# ───────────── Load Model ─────────────
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(GENRE_COLUMNS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ───────────── Evaluate ─────────────
y_true, y_pred = [], []

print("🔎 Running inference and evaluating performance...")
for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Evaluating posters"):
    img_path = os.path.join(SAVE_DIR, row['filename'])
    if not os.path.exists(img_path):
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.sigmoid(model(tensor)).cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)

        y_true.append(row['binary'])
        y_pred.append(preds)

    except Exception as e:
        print(f"❌ Error with {img_path}: {e}")

# ───────────── Results ─────────────
if y_true:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n📊 Classification Report (Per Genre):")
    print(classification_report(y_true, y_pred, target_names=GENRE_COLUMNS, zero_division=0))
else:
    print("⚠️ No valid predictions were made. Check image downloads and paths.")
