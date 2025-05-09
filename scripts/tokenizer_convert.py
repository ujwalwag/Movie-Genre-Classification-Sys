import pickle
import json

# Update this path to your pickle file
PICKLE_PATH = "models/tokenizer.pickle"
OUTPUT_JSON = "models/tokenizer_wordindex.json"

try:
    with open(PICKLE_PATH, "rb") as f:
        obj = pickle.load(f, encoding="latin1")  # Avoid keras load issues

    # Try accessing word_index safely
    if hasattr(obj, "word_index"):
        word_index = obj.word_index
    elif isinstance(obj, dict):
        word_index = obj
    else:
        raise ValueError("❌ Unexpected object type. Cannot extract word_index.")

    # Save as JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(word_index, out, ensure_ascii=False, indent=2)

    print(f"✅ Extracted word_index and saved to: {OUTPUT_JSON}")

except Exception as e:
    print("❌ Failed to process tokenizer:", str(e))
