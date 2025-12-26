from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# Load KB files
files = [
    "data/platform_kb.txt",
    "data/artist_kb.txt",
    "data/customer_kb.txt"
]

texts = []
for f in files:
    with open(f, "r", encoding="utf-8") as file:
        texts.append(file.read())

documents = "\n".join(texts).split("\n")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(documents, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index + documents
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/index.faiss")

with open("faiss_index/docs.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ Knowledge base ingested successfully")
