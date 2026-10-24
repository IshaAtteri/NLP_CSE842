import json

with open("data/raw/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

documents = []

for i, doc in enumerate(corpus):
    documents.append({
        "doc_id": f"doc_{i:04d}",
        "title": doc["title"],
        "text": doc["body"],
        "url": doc["url"],
        "source": doc["source"]
    })

print(f"Prepared {len(documents)} documents")
print(documents[0]["doc_id"])
print(documents[0]["title"])