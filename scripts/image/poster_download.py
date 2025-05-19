import pandas as pd
import os
import requests
from tqdm import tqdm
import urllib3
import random

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load your cleaned dataset
df = pd.read_csv("data/strictly_balanced_top10_cleaned.csv")  # update path if needed

# Keep only rows with valid poster paths
df = df.dropna(subset=["poster_path"])

# Sample 300 random entries
sampled_df = df.sample(n=3000, random_state=42)

# Base URL for poster images
TMDB_BASE_URL = "https://image.tmdb.org/t/p/w342"

# Folder to save images
SAVE_DIR = "downloaded_posters"
os.makedirs(SAVE_DIR, exist_ok=True)

# Download posters
download_errors = []
print("📥 Downloading posters...")

for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
    poster_path = row["poster_path"]
    poster_url = f"{TMDB_BASE_URL}{poster_path}"
    save_path = os.path.join(SAVE_DIR, poster_path.strip("/"))

    try:
        response = requests.get(poster_url, timeout=10, verify=False)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        download_errors.append((poster_url, str(e)))

print(f"✅ Download complete. Saved to: {SAVE_DIR}")
if download_errors:
    print(f"⚠️ {len(download_errors)} errors occurred. First few:")
    for url, err in download_errors[:5]:
        print(f"{url} → {err}")
