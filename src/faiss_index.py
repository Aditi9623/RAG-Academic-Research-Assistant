import faiss
import numpy as np


# Paths
EMBEDDINGS_PATH = "dataset/processed/embeddings.npy"
INDEX_PATH = "dataset/processed/faiss_index.index"


# Load embeddings
embeddings = np.load(EMBEDDINGS_PATH)

print("Embedding shape:", embeddings.shape)


# Get embedding dimension
dimension = embeddings.shape[1]


# Create FAISS index
index = faiss.IndexFlatL2(dimension)


# Add embeddings to index
index.add(embeddings)


# Save FAISS index
faiss.write_index(index, INDEX_PATH)


print("Total vectors in FAISS index:", index.ntotal)
print("FAISS index saved to:", INDEX_PATH)