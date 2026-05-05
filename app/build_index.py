import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

CSV_PATH = "examples/data/news_dataset.csv"

print("🔄 Loading CSV...")
df = pd.read_csv(CSV_PATH, encoding="utf-8").fillna("")

df["content"] = df["title"] + ". " + df["text"]

def chunk_text(text, size=150):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

data = []

for _, row in df.iterrows():
    chunks = chunk_text(row["content"])
    for chunk in chunks:
        data.append({
            "text": chunk,
            "category": row["category"],
            "source": row["source"],
            "url": row["url"],
            "title": row["title"]
        })

print(f"📊 Total chunks: {len(data)}")

print("🔄 Embedding...")
model = SentenceTransformer("keepitreal/vietnamese-sbert")
texts = [d["text"] for d in data]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

print("🔄 Building FAISS...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

print("💾 Saving...")
faiss.write_index(index, "examples/data/faiss.index")

with open("examples/data/data.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ DONE")