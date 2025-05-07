import torch
import torch.nn as nn           # ← add this
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

# ---- load artifacts ----
tok  = pickle.load(open('models/tokenizer.pickle','rb'))
emb  = np.load('models/embedding_matrix.npy')
state= torch.load('models/genre_classifier.pth', map_location='cpu')

# ---- model definition ----
class GenreLSTM(nn.Module):
    def __init__(self, emb):
        super().__init__()
        v, e = emb.shape
        self.embedding = nn.Embedding(v, e)                       # name matches checkpoint
        self.embedding.weight = nn.Parameter(torch.tensor(emb, dtype=torch.float32),
                                             requires_grad=False)
        self.lstm = nn.LSTM(e, 128, batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(256, 10)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out.mean(1))

# ---- load weights & test ----
model = GenreLSTM(emb)
model.load_state_dict(state)           # should load cleanly now
model.eval()

demo = "A detective must solve a brutal murder in a futuristic city."
seq  = tok.texts_to_sequences([demo])
pad  = pad_sequences(seq, maxlen=200)

with torch.no_grad():
    print(torch.sigmoid(model(torch.tensor(pad))).shape)
print("✅ model runs")
