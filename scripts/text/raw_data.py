import pandas as pd

# Load the dataset
df = pd.read_csv("data/TMDB_movie_dataset_v11.csv")

# Basic shape and structure
print("✅ Dataset loaded successfully!")
print("📦 Shape:", df.shape)
print("\n🧱 Columns:")
print(df.columns.tolist())

# Missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())


# Keep only the essential columns
columns_to_keep = ['id', 'genres', 'overview', 'poster_path', 'keywords']
df = df[columns_to_keep].copy()

# Drop rows where any of these essential columns are missing
df.dropna(subset=columns_to_keep, inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Show final shape and sample
print("✅ Cleaned dataset shape:", df.shape)
print("🔍 Remaining columns:", df.columns.tolist())
print("\n📌 Sample rows:")
print(df.head())

# Missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())


# Drop rows with missing genres or overview or poster_path
df_clean = df.dropna(subset=['genres', 'overview', 'poster_path'])
print("✅ After dropping rows with missing genres, overview, or poster_path:")
print(df_clean.shape)

import ast
from sklearn.preprocessing import MultiLabelBinarizer

# Convert stringified list to actual list
# Safely split genre strings into list of genres
df_clean['genres'] = df_clean['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')] if pd.notnull(x) else [])

# Initialize binarizer
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df_clean['genres'])

# Create DataFrame for genre labels
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
df_clean = df_clean.reset_index(drop=True)
df_clean = pd.concat([df_clean, genre_df], axis=1)

print("🎯 Genres processed successfully. Genre classes:", mlb.classes_)



# Base URL for TMDb images
base_url = "https://image.tmdb.org/t/p/w500"

df_clean['poster_url'] = df_clean['poster_path'].apply(lambda x: f"{base_url}{x}")


df_clean.to_csv("data/cleaned_movie_dataset.csv", index=False)
print("💾 Cleaned dataset saved.")
# Save to CSV
df_clean.to_csv("data/cleaned_movie_dataset.csv", index=False)
print("💾 Dataset saved as 'cleaned_movie_dataset.csv'")


import matplotlib.pyplot as plt

genre_counts = df_clean['genres'].explode().value_counts()
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar')
plt.title("🎬 Genre Distribution")
plt.xlabel("Genres")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Your one-hot genre columns
genre_columns = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
                 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
                 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

# Assuming df_clean already exists
df_clean['genre_count'] = df_clean[genre_columns].sum(axis=1)

# Distribution of genre counts per movie
genre_count_distribution = df_clean['genre_count'].value_counts().sort_index()

print("🎬 Number of movies by number of genres assigned:")
print(genre_count_distribution)

# Optional: Visualize it
plt.figure(figsize=(8, 5))
genre_count_distribution.plot(kind='bar', color='skyblue')
plt.title("🎭 Number of Genres per Movie")
plt.xlabel("Number of Genres")
plt.ylabel("Number of Movies")
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load your cleaned dataset
df_clean = pd.read_csv("data/cleaned_movie_dataset.csv")

# Step 2: Automatically determine top 10 genres by frequency
all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
              'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

# Calculate total appearances per genre
genre_totals = df_clean[all_genres].sum().sort_values(ascending=False)
top_10_genres = genre_totals.head(10).index.tolist()
print("🏆 Top 10 Genres Selected:", top_10_genres)

# Step 3: Set max samples per genre
max_samples_per_genre = 15000

# Step 4: Balance the dataset
balanced_df = pd.DataFrame()
already_added_indices = set()

for genre in top_10_genres:
    genre_subset = df_clean[df_clean[genre] == 1]
    genre_subset = genre_subset[~genre_subset.index.isin(already_added_indices)]

    if len(genre_subset) > max_samples_per_genre:
        genre_subset = genre_subset.sample(n=max_samples_per_genre, random_state=42)

    already_added_indices.update(genre_subset.index)
    balanced_df = pd.concat([balanced_df, genre_subset])

balanced_df = balanced_df.drop_duplicates().reset_index(drop=True)

# Step 5: Save to CSV
balanced_df.to_csv("data/balanced_top10_genres.csv", index=False)
print("✅ Balanced dataset saved as 'balanced_top10_genres.csv'")
print("🎯 New shape:", balanced_df.shape)
print("📊 Genre distribution:\n", balanced_df[top_10_genres].sum())

# Step 6: Plot the genre distribution
genre_counts = balanced_df[top_10_genres].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar', color='cornflowerblue')
plt.title("🎬 Number of Movies per Genre (Top 10, Balanced)", fontsize=14)
plt.ylabel("Number of Movies")
plt.xlabel("Genre")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import pandas as pd

# Load your cleaned dataset
df_clean = pd.read_csv("data/cleaned_movie_dataset.csv")

# Determine top 10 genres
all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
              'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

genre_totals = df_clean[all_genres].sum().sort_values(ascending=False)
top_10_genres = genre_totals.head(10).index.tolist()

# Target per genre
max_per_genre = 20000

# Tracker for how many movies we've kept per genre
genre_counter = {genre: 0 for genre in top_10_genres}
final_rows = []

# Iterate over the dataset row by row
for _, row in df_clean.iterrows():
    genres_in_row = [genre for genre in top_10_genres if row[genre] == 1]

    # Check if adding this row would overflow any genre
    if all(genre_counter[g] < max_per_genre for g in genres_in_row):
        final_rows.append(row)
        for g in genres_in_row:
            genre_counter[g] += 1

    # Stop early if all genre limits are reached
    if all(count >= max_per_genre for count in genre_counter.values()):
        break

# Final DataFrame
balanced_strict_df = pd.DataFrame(final_rows)

# Save
balanced_strict_df.to_csv("data/strictly_balanced_top10.csv", index=False)
print("✅ Saved as 'strictly_balanced_top10.csv'")
print("🎯 Shape:", balanced_strict_df.shape)
print("📊 Genre counts:\n", balanced_strict_df[top_10_genres].sum())

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the strictly balanced dataset
df_strict = pd.read_csv("data/strictly_balanced_top10.csv")

# Step 2: Define the top 10 genres (update if needed)
top_10_genres = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
                 'Horror', 'Documentary', 'Animation', 'Music', 'Crime']

# Step 3: Count the number of times each genre appears
genre_counts = df_strict[top_10_genres].sum().sort_values(ascending=False)

# Step 4: Plot the bar chart
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar', color='mediumseagreen')
plt.title("🎯 Strictly Balanced Genre Distribution (Top 10 Genres)", fontsize=14)
plt.ylabel("Number of Movies")
plt.xlabel("Genre")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import nltk
nltk.download('punkt_tab', download_dir='C:/Users/waghr/nltk_data')
import os
import nltk
nltk.data.path.append('C:/Users/waghr/nltk_data')

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


# Automatically download resources if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')


import re
import nltk
from nltk.corpus import stopwords
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
df = pd.read_csv("data/strictly_balanced_top10.csv")

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
# Create Embedding Matrix for Your Tokenizer
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

# ✅ Save the embedding matrix
np.save("embedding_matrix.npy", embedding_matrix)
print("💾 Saved embedding matrix to 'embedding_matrix.npy'")


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



# Save trained model
torch.save(model.state_dict(), "genre_classifier.pth")
print("✅ Model weights saved as 'genre_classifier.pth'")

# Save the Tokenizer using Pickle
import pickle

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Tokenizer saved as 'tokenizer.pickle'")


# Load the tokenizer
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print("✅ Tokenizer loaded successfully!")


# Unknown text example
unknown_text = "Two brothers are torn apart after they steal a bag of money and are hunted down by a ruthless killer."

# Preprocess
sequence = tokenizer.texts_to_sequences([unknown_text])
padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')

input_tensor = torch.tensor(padded_sequence, dtype=torch.long)

# Predict
with torch.no_grad():
    output_logits = model(input_tensor)
    preds = torch.sigmoid(output_logits).numpy()

# Map predictions to genre names
genre_columns = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
                 'Horror', 'Documentary', 'Animation', 'Music', 'Crime']

import numpy as np

top_N = 3  # Pick top 3 genres
top_indices = np.argsort(preds[0])[-top_N:][::-1]

predicted_genres = [genre_columns[i] for i in top_indices]

print(f"🎬 Top-{top_N} Predicted Genres:", predicted_genres)

# --- 1. Import necessary libraries ---
import torch
import torch.nn as nn
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 2. Load Tokenizer ---
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print("✅ Tokenizer loaded!")

# --- 3. Load Embedding Matrix ---
embedding_matrix = np.load('embedding_matrix.npy')

print("✅ Embedding matrix loaded!")

# --- 4. Define Your Genre Columns (Top 10 Genres) ---
genre_columns = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Action',
                 'Horror', 'Documentary', 'Animation', 'Music', 'Crime']

# --- 5. Define the GenreLSTM Model Class ---
class GenreLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.3):
        super(GenreLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # freeze embeddings

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 10)  # 10 genres

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        dropped = self.dropout(pooled)
        output = self.fc(dropped)
        return output

print("✅ Model architecture ready!")

# --- 6. Load Saved Model Weights ---
model = GenreLSTM(embedding_matrix)
model.load_state_dict(torch.load('genre_classifier.pth', map_location=torch.device('cpu')))
model.eval()

print("✅ Text genre model loaded and ready for prediction!")



