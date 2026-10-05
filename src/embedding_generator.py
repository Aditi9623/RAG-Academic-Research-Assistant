from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


# Input and output paths
INPUT_PATH = "dataset/processed/chunks.csv"
OUTPUT_PATH = "dataset/processed/embeddings.npy"

# Load processed chunks
df = pd.read_csv(INPUT_PATH)

print("Total chunks:", len(df))

# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings...")

embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

# Convert to NumPy array
embeddings = np.array(embeddings)

# Save embeddings
np.save(OUTPUT_PATH, embeddings)

print("Embedding shape:", embeddings.shape)
print("Embeddings saved to:", OUTPUT_PATH)