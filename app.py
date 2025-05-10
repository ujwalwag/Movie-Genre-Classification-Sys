import os
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pickle
from PIL import Image
from torchvision import models, transforms

# ─────────── Config ───────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DEVICE = torch.device("cpu")
print(f"✅ Using device: {DEVICE}")

app = Flask(__name__, static_folder='static', template_folder='templates')

GENRE_COLUMNS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
    'Horror', 'Documentary', 'Animation', 'Music', 'Crime'
]

IMG_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────── Text Model Definition ───────────
class GenreLSTM(nn.Module):
    def __init__(self, emb, hid=128, drop=0.3):
        super().__init__()
        vocab_size, emb_dim = emb.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(emb, dtype=torch.float32), requires_grad=False)
        self.lstm = nn.LSTM(emb_dim, hid, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hid * 2, len(GENRE_COLUMNS))

    def forward(self, x):
        lstm_out, _ = self.lstm(self.embedding(x))
        pooled = lstm_out.mean(dim=1)
        dropped = self.dropout(pooled)
        return self.fc(dropped)

# ─────────── Routes ───────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json(silent=True) or {}
    plot = (data.get("plot") or data.get("text") or "").strip()
    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    try:
        print("🔍 Loading tokenizer...")
        with open('models/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)

        print("🔍 Loading embedding matrix and model state...")
        embedding_matrix = np.load('models/embedding_matrix.npy')
        txt_state = torch.load('models/genre_classifier.pth', map_location=DEVICE)

        print("🔍 Initializing model...")
        model = GenreLSTM(embedding_matrix).to(DEVICE)
        model.load_state_dict(txt_state)
        model.eval()

        print("🔍 Tokenizing input...")
        seq = tokenizer.texts_to_sequences([plot])[0][:200]
        arr = np.zeros((1, 200), dtype=np.int64)
        arr[0, :len(seq)] = seq
        tensor = torch.tensor(arr, dtype=torch.long, device=DEVICE)

        print("🔍 Predicting...")
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0].cpu().numpy()

        top3 = probs.argsort()[-3:][::-1]
        print("✅ Prediction done.")
        return jsonify({"genres": [GENRE_COLUMNS[i] for i in top3]})

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return jsonify({"error": f"Failed to predict: {str(e)}"}), 500


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if 'poster' not in request.files:
        return jsonify({"error": "No poster uploaded"}), 400

    try:
        img = Image.open(request.files['poster'].stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(GENRE_COLUMNS))
        state = torch.load('models/poster_genre_classifier.pth', map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()

        tensor = IMG_TF(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor))[0].cpu().numpy()
        top3 = probs.argsort()[-3:][::-1]
        return jsonify({"genres": [GENRE_COLUMNS[i] for i in top3]})

    except Exception as e:
        return jsonify({"error": f"Failed to predict image genre: {str(e)}"}), 500

# ─────────── Launch ───────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

