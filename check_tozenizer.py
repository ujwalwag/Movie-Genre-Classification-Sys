import json

with open("models/tokenizer.json", "r", encoding="utf-8") as f:
    full_tokenizer = json.load(f)

word_index = full_tokenizer["config"]["word_index"]

with open("models/tokenizer_wordindex.json", "w", encoding="utf-8") as f:
    json.dump(word_index, f, indent=2)

print("✅ Saved models/tokenizer_wordindex.json")
