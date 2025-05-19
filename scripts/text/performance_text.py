# 📦 Imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# 🔧 Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENRE_COLUMNS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
    'Horror', 'Documentary', 'Animation', 'Music', 'Crime'
]

# 📶 Load artifacts
embedding_matrix = np.load('models/embedding_matrix.npy')
with open('models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
model_state = torch.load('models/genre_classifier.pth', map_location=DEVICE)

# 🧠 Define model
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

# 🏗️ Instantiate model
model = GenreLSTM(embedding_matrix).to(DEVICE)
model.load_state_dict(model_state)
model.eval()

# 🔠 Tokenizer helper
def texts_to_tensor(texts, max_len=200):
    seqs = tokenizer.texts_to_sequences(texts)
    arr = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, seq in enumerate(seqs):
        arr[i, :min(len(seq), max_len)] = seq[:max_len]
    return torch.tensor(arr, dtype=torch.long)

# 📄 Load test data
df = pd.read_csv('data/strictly_balanced_top10_cleaned.csv')  # expects "overview" and "genres" columns
df.dropna(subset=['overview', 'genres'], inplace=True)

# 🎯 Encode true genres
df['genres'] = df['genres'].apply(eval)  # from string to list if needed
mlb = MultiLabelBinarizer(classes=GENRE_COLUMNS)
y_true = mlb.fit_transform(df['genres'])

# 🧮 Predictions
batch_size = 64
all_preds = []

with torch.no_grad():
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['overview'].iloc[i:i+batch_size].tolist()
        tensor = texts_to_tensor(batch_texts).to(DEVICE)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.append(preds)

y_pred = np.vstack(all_preds)

# 📊 Report
print("📍 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=GENRE_COLUMNS, digits=3))
