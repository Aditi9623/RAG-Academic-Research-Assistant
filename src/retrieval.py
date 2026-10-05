import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

INDEX_PATH = "dataset/processed/faiss_index.index"
CHUNKS_PATH = "dataset/processed/chunks.csv"

TOP_K = 20
DISPLAY_K = 5


print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("FAISS vectors:", index.ntotal)

print("\nLoading processed chunks...")
df = pd.read_csv(CHUNKS_PATH)

print("Total chunks:", len(df))

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    tokenizer_kwargs={"use_fast": True}
)

print("Embedding model loaded.")


def reference_penalty(text):
    """
    Calculate a penalty for chunks that look like
    bibliography/reference-list content.

    Higher penalty = more likely to be a reference chunk.
    """

    text = str(text).strip()
    text_lower = text.lower()

    penalty = 0

    # ---------------------------------------------
    # Citation markers
    # ---------------------------------------------

    bracket_citations = len(
        re.findall(
            r"\[\d{1,3}\]",
            text
        )
    )

    if bracket_citations >= 2:
        penalty += bracket_citations * 2

    # ---------------------------------------------
    # Publication years
    # ---------------------------------------------

    year_count = len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text
        )
    )

    if year_count >= 3:
        penalty += year_count * 2

    # ---------------------------------------------
    # arXiv references
    # ---------------------------------------------

    arxiv_count = len(
        re.findall(
            r"\barxiv\b",
            text_lower
        )
    )

    penalty += arxiv_count * 5

    # ---------------------------------------------
    # DOI patterns
    # ---------------------------------------------

    doi_count = len(
        re.findall(
            r"\bdoi\b|10\.\d{4,9}/",
            text_lower
        )
    )

    penalty += doi_count * 5

    # ---------------------------------------------
    # Proceedings
    # ---------------------------------------------

    proceedings_count = len(
        re.findall(
            r"\bproceedings\b",
            text_lower
        )
    )

    penalty += proceedings_count * 4

    # ---------------------------------------------
    # Bibliography headings
    # ---------------------------------------------

    reference_headers = [
        "references",
        "bibliography",
        "reference list",
        "references and notes"
    ]

    for header in reference_headers:

        if text_lower.startswith(header):
            penalty += 100

    # ---------------------------------------------
    # Strong reference pattern:
    #
    # Author + title + arXiv/venue + year
    # ---------------------------------------------

    if (
        bracket_citations >= 2
        and year_count >= 2
        and (
            arxiv_count >= 1
            or doi_count >= 1
            or proceedings_count >= 1
        )
    ):
        penalty += 50

    return penalty


def retrieve_chunks(question, top_k=TOP_K):

    print("\nCreating query embedding...")

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    print("Retrieving relevant academic chunks...")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    candidate_results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        row = df.iloc[index_id]

        text = str(row["text"])

        penalty = reference_penalty(text)

        # -----------------------------------------
        # Lower distance = better semantic match
        #
        # We add the reference penalty so that
        # bibliography chunks move downward.
        # -----------------------------------------

        adjusted_score = float(distance) + penalty

        candidate_results.append({
            "paper_id": row["paper_id"],
            "page_number": int(row["page_number"]),
            "chunk_id": row["chunk_id"],
            "distance": float(distance),
            "reference_penalty": penalty,
            "adjusted_score": adjusted_score,
            "text": text
        })

    # ---------------------------------------------
    # Sort using adjusted score
    # Lower = better
    # ---------------------------------------------

    candidate_results.sort(
        key=lambda x: x["adjusted_score"]
    )

    return candidate_results[:DISPLAY_K]


if __name__ == "__main__":

    question = input(
        "\nEnter your research question: "
    )

    results = retrieve_chunks(question)

    print("\n--- Retrieved Results ---")

    if not results:

        print(
            "\nNo suitable academic evidence "
            "was found."
        )

    else:

        for number, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult: {number}"
            )

            print(
                f"Paper ID: "
                f"{result['paper_id']}"
            )

            print(
                f"Page: "
                f"{result['page_number']}"
            )

            print(
                f"Chunk ID: "
                f"{result['chunk_id']}"
            )

            print(
                f"Distance: "
                f"{result['distance']}"
            )

            print(
                f"Reference Penalty: "
                f"{result['reference_penalty']}"
            )

            print(
                f"Adjusted Score: "
                f"{result['adjusted_score']}"
            )

            print("Text:")

            print(
                result["text"][:1000]
            )

            print("-" * 70)