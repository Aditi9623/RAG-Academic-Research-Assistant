import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "dataset/processed/chunks.csv"
INDEX_PATH = "dataset/processed/faiss_index.index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 20
FINAL_TOP_K = 5

MAX_CHUNKS_PER_PAPER = 2


# ============================================================
# LOAD FAISS
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print(f"FAISS vectors: {index.ntotal}")


# ============================================================
# LOAD CHUNKS
# ============================================================

print("\nLoading processed chunks...")

df = pd.read_csv(CHUNKS_PATH)

print(f"Total chunks: {len(df)}")

print("\nAvailable columns:")
print(list(df.columns))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# LOAD CROSS ENCODER
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker model loaded.")


# ============================================================
# SAFE COLUMN ACCESS
# ============================================================

def get_column_value(row, possible_columns, default="Not available"):

    for column in possible_columns:

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                return value

    return default


def get_paper_id(row):

    return get_column_value(
        row,
        ["paper_id", "paper", "id"],
        "Unknown"
    )


def get_title(row):

    return get_column_value(
        row,
        ["title", "paper_title"],
        "Title not available"
    )


def get_year(row):

    return get_column_value(
        row,
        ["year", "publication_year"],
        "Year not available"
    )


def get_page(row):

    return get_column_value(
        row,
        [
            "page",
            "page_number",
            "page_num",
            "page_no"
        ],
        "Not available"
    )


def get_chunk_id(row):

    return get_column_value(
        row,
        [
            "chunk_id",
            "chunk",
            "id"
        ],
        "Not available"
    )


# ============================================================
# REFERENCE / BIBLIOGRAPHY DETECTION
# ============================================================

def reference_penalty(text):

    text = str(text)

    text_lower = text.lower()

    penalty = 0.0

    # --------------------------------------------------------
    # Strong reference indicators
    # --------------------------------------------------------

    strong_terms = [

        "references",
        "bibliography",
        "reference list",
        "doi:",
        "https://doi.org/",
        "arxiv preprint",
        "proceedings of the",
        "journal of",
        "vol.",
        "pp."

    ]

    for term in strong_terms:

        if term in text_lower:

            penalty += 4.0


    # --------------------------------------------------------
    # Academic citation patterns
    # --------------------------------------------------------

    citation_count = text_lower.count("[")

    if citation_count >= 5:

        penalty += 3.0

    if citation_count >= 10:

        penalty += 5.0

    if citation_count >= 15:

        penalty += 8.0


    # --------------------------------------------------------
    # Reference-like author/year patterns
    # --------------------------------------------------------

    reference_patterns = [

        " et al.",
        "(2020)",
        "(2021)",
        "(2022)",
        "(2023)",
        "(2024)",
        "(2025)",
        "(2026)",
        "2020).",
        "2021).",
        "2022).",
        "2023).",
        "2024).",
        "2025).",
        "2026)."

    ]

    pattern_count = 0

    for pattern in reference_patterns:

        if pattern in text_lower:

            pattern_count += 1


    if pattern_count >= 4:

        penalty += 5.0


    if pattern_count >= 8:

        penalty += 8.0


    return penalty


# ============================================================
# DETECT REFERENCE CHUNK
# ============================================================

def is_reference_chunk(text):

    text = str(text)

    text_lower = text.lower()

    penalty = reference_penalty(text)


    # Very strong penalty
    if penalty >= 12:

        return True


    # --------------------------------------------------------
    # Reference-list style chunk
    # --------------------------------------------------------

    reference_words = [

        "references",
        "bibliography",
        "arxiv preprint",
        "proceedings of",
        "doi:",
        "https://doi.org/"

    ]

    reference_word_count = 0

    for word in reference_words:

        if word in text_lower:

            reference_word_count += 1


    if reference_word_count >= 2:

        return True


    # --------------------------------------------------------
    # Dense bibliography detection
    # --------------------------------------------------------

    citation_count = text_lower.count("[")

    year_count = sum(
        text_lower.count(f"{year})")
        for year in range(2015, 2027)
    )


    if citation_count >= 10 and year_count >= 4:

        return True


    # --------------------------------------------------------
    # Detect many author/reference patterns
    # --------------------------------------------------------

    et_al_count = text_lower.count(
        " et al."
    )

    if et_al_count >= 3:

        return True


    return False


# ============================================================
# CONTENT QUALITY
# ============================================================

def content_quality_score(text):

    text = str(text)

    text_lower = text.lower()

    score = 0.0


    positive_terms = [

        "is defined as",
        "defined as",
        "refers to",
        "is a",
        "is an",
        "means",
        "consists of",
        "works by",
        "involves",
        "allows",
        "enables",
        "proposes",
        "method",
        "framework",
        "architecture",
        "approach",
        "retrieval",
        "generation",
        "knowledge",
        "external",
        "context"

    ]


    for term in positive_terms:

        if term in text_lower:

            score += 0.75


    word_count = len(
        text.split()
    )


    # Good academic chunk length
    if 80 <= word_count <= 250:

        score += 2.0


    elif 40 <= word_count < 80:

        score += 1.0


    # Too short
    if word_count < 30:

        score -= 2.0


    # Too long
    if word_count > 400:

        score -= 1.0


    return score


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    q = question.lower()


    # LIMITATION FIRST
    if any(
        word in q
        for word in [
            "limitation",
            "limitations",
            "drawback",
            "drawbacks",
            "weakness",
            "weaknesses",
            "problem",
            "problems"
        ]
    ):

        return "LIMITATION"


    if any(
        word in q
        for word in [
            "what is",
            "what are",
            "define",
            "definition",
            "meaning"
        ]
    ):

        return "DEFINITION"


    if any(
        word in q
        for word in [
            "how does",
            "how do",
            "how can",
            "how is",
            "working",
            "works"
        ]
    ):

        return "HOW"


    if any(
        word in q
        for word in [
            "why",
            "purpose",
            "benefit",
            "benefits",
            "advantage",
            "advantages"
        ]
    ):

        return "PURPOSE"


    if any(
        word in q
        for word in [
            "compare",
            "comparison",
            "difference",
            "different"
        ]
    ):

        return "COMPARISON"


    if any(
        word in q
        for word in [
            "dataset",
            "data set"
        ]
    ):

        return "DATASET"


    if any(
        word in q
        for word in [
            "technique",
            "method",
            "algorithm",
            "architecture"
        ]
    ):

        return "TECHNIQUE"


    if any(
        word in q
        for word in [
            "metric",
            "evaluation",
            "accuracy",
            "performance"
        ]
    ):

        return "METRIC"


    return "GENERAL"


# ============================================================
# INTENT SCORE
# ============================================================

def intent_score(text, question):

    text_lower = str(text).lower()

    q_type = question_type(
        question
    )

    score = 0.0


    intent_terms = {

        "DEFINITION": [

            "defined",
            "definition",
            "refers to",
            "is a",
            "is an",
            "means",
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "rag",
            "external knowledge",
            "external database",
            "knowledge base",
            "retrieval",
            "generation"

        ],


        "HOW": [

            "retrieval",
            "generation",
            "query",
            "context",
            "retrieve",
            "generate",
            "pipeline",
            "process"

        ],


        "PURPOSE": [

            "benefit",
            "benefits",
            "improve",
            "accuracy",
            "hallucination",
            "knowledge",
            "external",
            "grounded"

        ],


        "COMPARISON": [

            "compared",
            "difference",
            "whereas",
            "while",
            "versus",
            "better",
            "similar"

        ],


        "LIMITATION": [

            "limitation",
            "limitations",
            "drawback",
            "challenge",
            "problem",
            "weakness",
            "difficulty",
            "lack",
            "however",
            "but"

        ],


        "DATASET": [

            "dataset",
            "data set",
            "corpus",
            "samples",
            "training data",
            "test data"

        ],


        "TECHNIQUE": [

            "method",
            "algorithm",
            "architecture",
            "framework",
            "technique",
            "model",
            "retrieval",
            "reranking"

        ],


        "METRIC": [

            "accuracy",
            "precision",
            "recall",
            "f1",
            "evaluation",
            "metric",
            "performance"

        ],


        "GENERAL": [

            "retrieval",
            "generation",
            "llm",
            "language model",
            "knowledge"

        ]

    }


    for term in intent_terms.get(
        q_type,
        []
    ):

        if term in text_lower:

            score += 1.0


    return score


# ============================================================
# DEFINITION-SPECIFIC SCORE
# ============================================================

def definition_score(text):

    text_lower = str(text).lower()

    score = 0.0


    definition_phrases = [

        "is defined as",
        "defined as",
        "refers to",
        "is a technique",
        "is an approach",
        "is a framework",
        "is a method",
        "combines",
        "incorporates",
        "enhances llms",
        "retrieves relevant",
        "external knowledge",
        "external knowledge base",
        "external database"

    ]


    for phrase in definition_phrases:

        if phrase in text_lower:

            score += 2.0


    # RAG-specific explanation
    if (
        "retrieval-augmented generation"
        in text_lower
    ):

        score += 2.0


    if (
        "retrieval augmented generation"
        in text_lower
    ):

        score += 2.0


    if (
        "external knowledge"
        in text_lower
    ):

        score += 2.0


    return score


# ============================================================
# DUPLICATE TEXT DETECTION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = " ".join(
        text.split()
    )

    return text


def is_duplicate(
    text,
    selected_texts
):

    normalized = normalize_text(
        text
    )


    for selected in selected_texts:

        selected_normalized = normalize_text(
            selected
        )


        # Exact duplicate
        if normalized == selected_normalized:

            return True


        words1 = set(
            normalized.split()
        )

        words2 = set(
            selected_normalized.split()
        )


        if not words1 or not words2:

            continue


        intersection = len(
            words1.intersection(words2)
        )

        union = len(
            words1.union(words2)
        )


        similarity = (
            intersection / union
        )


        if similarity >= 0.85:

            return True


    return False


# ============================================================
# QUESTION
# ============================================================

question = input(
    "\nEnter your research question: "
).strip()


if not question:

    print(
        "Question cannot be empty."
    )

    raise SystemExit


q_type = question_type(
    question
)


# ============================================================
# QUERY EMBEDDING
# ============================================================

print(
    "\nCreating query embedding..."
)

query_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True,
    normalize_embeddings=True
)

query_embedding = query_embedding.astype(
    "float32"
)


# ============================================================
# FAISS RETRIEVAL
# ============================================================

print(
    "Retrieving relevant academic chunks..."
)

distances, indices = index.search(
    query_embedding,
    INITIAL_TOP_K
)


# ============================================================
# BUILD CANDIDATES
# ============================================================

candidates = []


for rank, idx in enumerate(
    indices[0]
):

    if idx < 0 or idx >= len(df):

        continue


    row = df.iloc[idx]

    text = str(
        row["text"]
    )


    # Skip obvious reference chunks immediately
    if is_reference_chunk(text):

        continue


    ref_penalty = reference_penalty(
        text
    )


    quality = content_quality_score(
        text
    )


    intent = intent_score(
        text,
        question
    )


    definition = 0.0


    if q_type == "DEFINITION":

        definition = definition_score(
            text
        )


    candidates.append({

        "row": row,

        "text": text,

        "distance": float(
            distances[0][rank]
        ),

        "reference_penalty": ref_penalty,

        "quality": quality,

        "intent": intent,

        "definition": definition

    })


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

print(
    "\nReranking retrieved evidence..."
)


texts = [
    candidate["text"]
    for candidate in candidates
]


pairs = [
    [question, text]
    for text in texts
]


cross_scores = reranker.predict(
    pairs
)


for candidate, cross_score in zip(
    candidates,
    cross_scores
):

    candidate["cross_score"] = float(
        cross_score
    )


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    candidate["final_score"] = (

        candidate["cross_score"]

        + candidate["quality"]

        + candidate["intent"]

        + candidate["definition"]

        - candidate["reference_penalty"]

    )


# ============================================================
# SORT
# ============================================================

candidates.sort(
    key=lambda x: x["final_score"],
    reverse=True
)


# ============================================================
# SELECT EVIDENCE
# ============================================================

selected = []

selected_texts = []

paper_counts = {}


for candidate in candidates:

    row = candidate["row"]

    text = candidate["text"]

    paper_id = get_paper_id(
        row
    )


    # --------------------------------------------------------
    # Safety reference check
    # --------------------------------------------------------

    if is_reference_chunk(text):

        continue


    # --------------------------------------------------------
    # Duplicate text
    # --------------------------------------------------------

    if is_duplicate(
        text,
        selected_texts
    ):

        continue


    # --------------------------------------------------------
    # Same paper limit
    # --------------------------------------------------------

    current_count = paper_counts.get(
        paper_id,
        0
    )


    if current_count >= MAX_CHUNKS_PER_PAPER:

        continue


    # --------------------------------------------------------
    # Select
    # --------------------------------------------------------

    selected.append(
        candidate
    )

    selected_texts.append(
        text
    )

    paper_counts[paper_id] = (
        current_count + 1
    )


    if len(selected) >= FINAL_TOP_K:

        break


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n--- Selected Evidence ---"
)


if not selected:

    print(
        "No suitable evidence found."
    )

else:

    for i, candidate in enumerate(
        selected,
        start=1
    ):

        row = candidate["row"]


        print(
            f"\nEvidence: {i}"
        )

        print(
            f"Paper ID: "
            f"{get_paper_id(row)}"
        )

        print(
            f"Title: "
            f"{get_title(row)}"
        )

        print(
            f"Year: "
            f"{get_year(row)}"
        )

        print(
            f"Page: "
            f"{get_page(row)}"
        )

        print(
            f"Chunk ID: "
            f"{get_chunk_id(row)}"
        )

        print(
            f"Cross-Encoder Score: "
            f"{candidate['cross_score']}"
        )

        print(
            f"Reference Penalty: "
            f"{candidate['reference_penalty']}"
        )

        print(
            f"Content Quality: "
            f"{candidate['quality']}"
        )

        print(
            f"Intent Score: "
            f"{candidate['intent']}"
        )

        print(
            f"Definition Score: "
            f"{candidate['definition']}"
        )

        print(
            f"Final Score: "
            f"{candidate['final_score']}"
        )

        print("Text:")

        print(
            candidate["text"]
        )

        print(
            "-" * 70
        )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\nRetrieval Summary"
)

print(
    "-----------------"
)

print(
    f"Initial FAISS candidates: "
    f"{INITIAL_TOP_K}"
)

print(
    f"Candidates after reference filtering: "
    f"{len(candidates)}"
)

print(
    f"Final selected evidence: "
    f"{len(selected)}"
)

print(
    f"Unique papers selected: "
    f"{len(set(paper_counts.keys()))}"
)

print(
    f"Question type: "
    f"{q_type}"
)

print(
    "\nReranking completed successfully."
)