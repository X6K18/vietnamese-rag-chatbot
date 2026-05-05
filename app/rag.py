import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class RAG:
    def __init__(self, index_path, data_path):
        self.index = faiss.read_index(index_path)

        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        # lazy load
        self.model = None

    def encode(self, query):
        if self.model is None:
            self.model = SentenceTransformer("keepitreal/vietnamese-sbert")
        return self.model.encode([query])

    def search(self, query, category=None, k=2):
        q_emb = self.encode(query)
        D, I = self.index.search(np.array(q_emb), k * 5)

        results = []
        for idx in I[0]:
            item = self.data[idx]

            if category and item["category"] != category:
                continue

            results.append(item)

            if len(results) == k:
                break

        return results