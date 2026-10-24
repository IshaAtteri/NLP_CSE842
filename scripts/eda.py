import json
import numpy as np

# Load the corpus
with open("data/raw/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Calculate word counts for each article
word_counts = []

for doc in corpus:
    text = doc["body"]
    word_counts.append(len(text.split()))

# Convert to NumPy array
word_counts = np.array(word_counts)

# Summary statistics
print(f"Total documents: {len(corpus)}")
print(f"Minimum words: {np.min(word_counts)}")
print(f"Maximum words: {np.max(word_counts)}")
print(f"Mean words: {np.mean(word_counts):.2f}")
print(f"Median words: {np.median(word_counts):.2f}")
print(f"25th percentile: {np.percentile(word_counts, 25):.2f}")
print(f"75th percentile: {np.percentile(word_counts, 75):.2f}")
print(f"90th percentile: {np.percentile(word_counts, 90):.2f}")

# Check missing values in important fields
fields = ["title", "body", "url", "source"]

for field in fields:
    missing = sum(
        1 for doc in corpus
        if doc.get(field) is None
        or not str(doc.get(field)).strip()
    )

    print(f"{field}: {missing} missing values")

# Check for unusually short documents
short_docs = [
    doc for doc in corpus
    if len((doc.get("body") or "").split()) < 100
]

print(f"\nDocuments with fewer than 100 words: {len(short_docs)}")