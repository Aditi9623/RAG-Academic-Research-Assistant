from chunker import create_chunks


sample_text = """
Retrieval-Augmented Generation combines information retrieval
with large language models. It retrieves relevant information
from an external knowledge source and provides that information
to the language model as context. This can help improve the
grounding of generated responses. Academic research assistants
can use this approach to retrieve relevant research papers
and generate answers based on retrieved evidence.
"""


chunks = create_chunks(sample_text, chunk_size=30, overlap=5)


print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)