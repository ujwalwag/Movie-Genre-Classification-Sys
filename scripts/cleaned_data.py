import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    text = str(text).lower()                                # Lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)                 # Remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()                # Normalize whitespace
    tokens = nltk.word_tokenize(text)                       # Tokenize
    tokens = [t for t in tokens if t not in stop_words]     # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]      # Lemmatize
    return " ".join(tokens)






import pandas as pd

# Load your dataset
df = pd.read_csv("C:/Users/waghr/Desktop/BOAI_Project\Multi-Modal-Movie-Genre-Classification-System/data/strictly_balanced_top10.csv")

# Clean the overviews
df['clean_overview'] = df['overview'].apply(clean_text_advanced)

# Preview
print(df[['overview', 'clean_overview']].head(3))

df.to_csv("data/strictly_balanced_top10_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'strictly_balanced_top10_cleaned.csv'")

# Tokenization and Padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Limit vocab to top 20,000 words
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_overview'])

# Convert overviews to sequences of tokens
sequences = tokenizer.texts_to_sequences(df['clean_overview'])

# Pad sequences to a max length (e.g., 200 tokens)
X_text = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

print("✅ Tokenized and padded text shape:", X_text.shape)

# prepare label Matrix
# Define your top 10 genre columns
genre_columns = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
                 'Horror', 'Documentary', 'Animation', 'Music', 'Crime']

# Create the label matrix
import numpy as np
y_labels = df[genre_columns].values.astype('float32')

print("✅ Label matrix shape:", y_labels.shape)

# Save the processed data
np.save("X_text.npy", X_text)
np.save("y_labels.npy", y_labels)
print("💾 Saved preprocessed data for modeling.")

#Load GloVe + Create Embedding Matrix
embedding_index = {}
with open("data/glove.6B.100d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

print(f"✅ Loaded {len(embedding_index)} word vectors from GloVe.")

#Create Embedding Matrix for Your Tokenizer
embedding_dim = 100
vocab_size = min(20000, len(tokenizer.word_index) + 1)

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("✅ Embedding matrix shape:", embedding_matrix.shape)

# Create  dataloaders for training and validation
# 1. Imports
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd

# 2. Load your saved arrays
X_text = np.load("X_text.npy")
y_labels = np.load("y_labels.npy")

# 3. Define Dataset class
class MovieGenreDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. Create dataset
dataset = MovieGenreDataset(X_text, y_labels)

# 5. Split into train/validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 6. Create DataLoaders (set num_workers=0 for Jupyter)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"📦 Train set: {len(train_dataset)} samples | Validation set: {len(val_dataset)} samples")

print(f"📦 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# PyTorch LSTM Model
import torch.nn as nn
import torch.nn.functional as F

class GenreLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.3):
        super(GenreLSTM, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # Embedding layer with pretrained GloVe weights
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # freeze embeddings

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to output genre logits
        self.fc = nn.Linear(hidden_dim * 2, 10)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)         # [B, 200, 100]
        lstm_out, _ = self.lstm(embedded)    # [B, 200, 256]
        pooled = torch.mean(lstm_out, dim=1) # average over time steps
        dropped = self.dropout(pooled)
        output = self.fc(dropped)            # [B, 10] (genre logits)
        return output


# Training Loop Setup
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

# Instantiate model
model = GenreLSTM(embedding_matrix)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop with Validation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

# Set device
device = torch.device('cpu')
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store losses and F1 scores
train_losses = []
val_losses = []
train_micro_f1s = []
val_micro_f1s = []

def train_model_collect(model, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        all_y_true = []
        all_y_pred = []

        for batch in train_loader:
            X_batch, y_batch = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().detach().numpy()
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend((preds >= 0.5).astype(int))

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_micro = f1_score(all_y_true, all_y_pred, average='micro')
        train_micro_f1s.append(train_micro)

        # Validation
        val_loss, val_micro = validate_model_collect(model, val_loader)
        val_losses.append(val_loss)
        val_micro_f1s.append(val_micro)

        print(f"📚 Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Micro-F1: {train_micro:.4f} | Val Loss: {val_loss:.4f} | Val Micro-F1: {val_micro:.4f}")

def validate_model_collect(model, val_loader):
    model.eval()
    running_loss = 0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = [b.to(device) for b in batch]
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend((preds >= 0.5).astype(int))

    avg_val_loss = running_loss / len(val_loader)
    val_micro = f1_score(all_y_true, all_y_pred, average='micro')
    return avg_val_loss, val_micro

def final_plot_losses_and_f1s(train_losses, val_losses, train_f1s, val_f1s):
    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.title("📉 Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # F1 plot
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label="Train Micro-F1", marker='o')
    plt.plot(val_f1s, label="Validation Micro-F1", marker='o')
    plt.title("📈 Micro F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Start Training
train_model_collect(model, train_loader, val_loader, epochs=10)
final_plot_losses_and_f1s(train_losses, val_losses, train_micro_f1s, val_micro_f1s)