import json
import numpy as np

with open("models/tokenizer_wordindex.json") as f:
    word_index = json.load(f)

embedding_matrix = np.load("models/embedding_matrix.npy")

print("word_index size:", len(word_index))
print("embedding_matrix shape:", embedding_matrix.shape)
