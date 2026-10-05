import pandas as pd
import faiss
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from llm_generator import generate_answer
from evidence_answer import build_clean_evidence


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "dataset/processed/chunks.csv"
INDEX_PATH = "dataset/processed/faiss_index.index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 20
FINAL_TOP_K = 5


# ============================================================
# LOAD DATA
# ============================================================

print("Loading chunks...")

chunks_df = pd.read_csv(CHUNKS_PATH)

print(f"Total chunks loaded: {len(chunks_df)}")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print(f"FAISS vectors loaded: {index.ntotal}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# LOAD RERANKER
# ============================================================

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded.")


# ============================================================
# REFERENCE PENALTY
# ============================================================

def reference_penalty(text):

    text = str(text).strip()
    text_lower = text.lower()

    penalty = 0

    citation_count = len(
        re.findall(r"\[\d{1,3}\]", text)
    )

    year_count = len(
        re.findall(r"\b(?:19|20)\d{2}\b", text)
    )

    arxiv_count = len(
        re.findall(r"\barxiv\b", text_lower)
    )

    doi_count = len(
        re.findall(
            r"\bdoi\b|10\.\d{4,9}/",
            text_lower
        )
    )

    proceedings_count = len(
        re.findall(
            r"\bproceedings\b",
            text_lower
        )
    )

    if citation_count >= 2:
        penalty += citation_count * 2

    if year_count >= 3:
        penalty += year_count * 2

    penalty += arxiv_count * 5
    penalty += doi_count * 5
    penalty += proceedings_count * 4

    reference_headers = [
        "references",
        "bibliography",
        "reference list",
        "references and notes"
    ]

    for header in reference_headers:

        if text_lower.startswith(header):
            penalty += 100

    if (
        citation_count >= 2
        and year_count >= 2
        and (
            arxiv_count >= 1
            or doi_count >= 1
            or proceedings_count >= 1
        )
    ):
        penalty += 50

    return penalty


# ============================================================
# BIBLIOGRAPHY / METADATA DETECTION
# ============================================================

def is_reference_chunk(text):

    text = str(text).strip()
    text_lower = text.lower()

    words = text.split()

    citation_count = len(
        re.findall(r"\[\d{1,3}\]", text)
    )

    year_count = len(
        re.findall(r"\b(?:19|20)\d{2}\b", text)
    )

    # Explicit reference headings
    if text_lower.startswith(
        (
            "references",
            "bibliography",
            "reference list",
            "references and notes"
        )
    ):
        return True

    # Very citation-heavy chunk
    if citation_count >= 6:
        return True

    # Many years + citations
    if (
        citation_count >= 3
        and year_count >= 3
    ):
        return True

    # Strong publication-reference signals
    strong_reference_terms = [
        "proceedings of the",
        "advances in neural information processing",
        "association for computational linguistics",
        "journal of",
        "doi:",
        "arxiv.org"
    ]

    strong_reference_matches = sum(
        1
        for term in strong_reference_terms
        if term in text_lower
    )

    if (
        strong_reference_matches >= 2
        and citation_count >= 1
    ):
        return True

    # Short bibliography-like chunks
    if (
        len(words) < 120
        and citation_count >= 2
        and year_count >= 2
    ):
        return True

    return False


# ============================================================
# TITLE / ABSTRACT HEADER DETECTION
# ============================================================

def metadata_penalty(text):

    text = str(text).strip()
    text_lower = text.lower()

    penalty = 0
    words = text.split()

    # Abstract/header
    if (
        len(words) < 180
        and (
            "abstract—" in text_lower
            or "abstract -" in text_lower
            or text_lower.startswith("abstract")
        )
    ):
        penalty += 8

    # Academic affiliation indicators
    affiliation_terms = [
        "university",
        "institute",
        "department",
        "school of",
        "college of"
    ]

    affiliation_count = sum(
        1
        for term in affiliation_terms
        if term in text_lower
    )

    if (
        affiliation_count >= 2
        and len(words) < 180
    ):
        penalty += 5

    # Paper title / author header pattern
    if (
        len(words) < 100
        and (
            "authors" in text_lower
            or "et al." in text_lower
        )
    ):
        penalty += 4

    return penalty


# ============================================================
# CONTENT QUALITY
# ============================================================

def content_quality_score(text):

    text = str(text).strip()

    score = 0
    words = text.split()

    if len(words) >= 80:
        score += 3

    if len(words) >= 150:
        score += 2

    if "." in text:
        score += 1

    if len(words) >= 40:
        score += 1

    if len(words) < 30:
        score -= 3

    explanatory_terms = [
        "refers to",
        "is a",
        "is an",
        "consists of",
        "involves",
        "allows",
        "enables",
        "provides",
        "uses",
        "integrates",
        "incorporates"
    ]

    for term in explanatory_terms:

        if term in text.lower():
            score += 1

    return score


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    q = question.lower()

    if (
        "what is" in q
        or "what are" in q
        or "define" in q
        or "definition" in q
    ):
        return "DEFINITION"

    if (
        "how does" in q
        or "how do" in q
        or "how is" in q
        or "how are" in q
        or "how can" in q
    ):
        return "HOW"

    if (
        "why" in q
        or "purpose" in q
        or "benefit" in q
        or "advantage" in q
    ):
        return "PURPOSE"

    if (
        "compare" in q
        or "comparison" in q
        or "difference" in q
        or "different" in q
    ):
        return "COMPARISON"

    if (
        "limitation" in q
        or "limitations" in q
        or "challenge" in q
        or "challenges" in q
        or "drawback" in q
        or "drawbacks" in q
        or "future work" in q
    ):
        return "LIMITATION"

    if (
        "dataset" in q
        or "datasets" in q
        or "data used" in q
        or "training data" in q
        or "testing data" in q
    ):
        return "DATASET"

    if (
        "technique" in q
        or "techniques" in q
        or "method" in q
        or "methods" in q
        or "algorithm" in q
        or "algorithms" in q
        or "model" in q
        or "models" in q
        or "approach" in q
        or "approaches" in q
    ):
        return "TECHNIQUE"

    if (
        "metric" in q
        or "metrics" in q
        or "accuracy" in q
        or "precision" in q
        or "recall" in q
        or "f1" in q
        or "auc" in q
        or "auroc" in q
        or "performance" in q
        or "evaluation" in q
    ):
        return "METRIC"

    return "GENERAL"


# ============================================================
# QUESTION INTENT SCORE
# ============================================================

def intent_score(question, text):

    q_type = question_type(question)
    text_lower = str(text).lower()

    score = 0

    keyword_map = {

        "DEFINITION": [
            "definition",
            "defined as",
            "refers to",
            "is a",
            "is an",
            "consists of",
            "involves",
            "integrates",
            "incorporates",
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "rag"
        ],

        "HOW": [
            "method",
            "process",
            "retrieval",
            "generation",
            "architecture",
            "workflow",
            "approach",
            "pipeline",
            "steps"
        ],

        "PURPOSE": [
            "purpose",
            "benefit",
            "advantage",
            "improve",
            "accuracy",
            "knowledge",
            "hallucination",
            "reliability"
        ],

        "COMPARISON": [
            "compared",
            "comparison",
            "difference",
            "versus",
            "while",
            "whereas",
            "similar"
        ],

        "LIMITATION": [
            "limitation",
            "challenge",
            "challenges",
            "problem",
            "drawback",
            "future work",
            "future research",
            "difficulty",
            "difficulties",
            "issue",
            "issues",
            "inefficient",
            "inefficiency",
            "noise",
            "noisy",
            "generalization"
        ],

        "DATASET": [
            "dataset",
            "datasets",
            "data",
            "training",
            "testing",
            "benchmark"
        ],

        "TECHNIQUE": [
            "method",
            "methodology",
            "technique",
            "algorithm",
            "model",
            "approach",
            "architecture"
        ],

        "METRIC": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "auroc",
            "metric",
            "evaluation",
            "performance"
        ]
    }

    keywords = keyword_map.get(
        q_type,
        []
    )

    multiplier = (
        2
        if q_type in [
            "DATASET",
            "TECHNIQUE",
            "METRIC"
        ]
        else 1
    )

    for keyword in keywords:

        if keyword in text_lower:
            score += multiplier

    return score


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates(question):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        INITIAL_TOP_K
    )

    candidates = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        row = chunks_df.iloc[
            int(index_id)
        ]

        candidates.append(
            {
                "index_id": int(index_id),
                "distance": float(distance),
                "paper_id": row["paper_id"],
                "page_number": int(
                    row["page_number"]
                ),
                "chunk_id": row["chunk_id"],
                "text": str(row["text"])
            }
        )

    return candidates


# ============================================================
# RERANK
# ============================================================

def rerank_candidates(
    question,
    candidates
):

    if not candidates:
        return []

    pairs = [
        (
            question,
            candidate["text"]
        )
        for candidate in candidates
    ]

    cross_scores = reranker.predict(
        pairs
    )

    ranked = []

    for candidate, cross_score in zip(
        candidates,
        cross_scores
    ):

        text = candidate["text"]

        # Hard remove bibliography
        if is_reference_chunk(text):
            continue

        ref_penalty = reference_penalty(
            text
        )

        meta_penalty = metadata_penalty(
            text
        )

        quality = content_quality_score(
            text
        )

        intent = intent_score(
            question,
            text
        )

        final_score = (
            float(cross_score)
            - float(ref_penalty)
            - float(meta_penalty)
            + float(quality)
            + float(intent)
        )

        candidate["cross_score"] = float(
            cross_score
        )

        candidate["reference_penalty"] = (
            ref_penalty
        )

        candidate["metadata_penalty"] = (
            meta_penalty
        )

        candidate["quality_score"] = (
            quality
        )

        candidate["intent_score"] = (
            intent
        )

        candidate["final_score"] = (
            final_score
        )

        ranked.append(
            candidate
        )

    ranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return ranked


# ============================================================
# FINAL EVIDENCE SELECTION
# ============================================================

def select_evidence(
    question,
    ranked_candidates,
    top_k=FINAL_TOP_K
):

    evidence = []
    used_chunks = set()

    q_type = question_type(
        question
    )

    for candidate in ranked_candidates:

        chunk_id = candidate[
            "chunk_id"
        ]

        text = candidate[
            "text"
        ]

        if chunk_id in used_chunks:
            continue

        if is_reference_chunk(text):
            continue

        # Definition-specific protection
        if q_type == "DEFINITION":

            meta_penalty = metadata_penalty(
                text
            )

            if meta_penalty >= 8:
                continue

        evidence.append(
            {
                "paper_id": candidate[
                    "paper_id"
                ],

                "page_number": candidate[
                    "page_number"
                ],

                "chunk_id": chunk_id,

                "reranking_score": candidate[
                    "final_score"
                ],

                "reference_penalty": candidate[
                    "reference_penalty"
                ],

                "text": text
            }
        )

        used_chunks.add(
            chunk_id
        )

        if len(evidence) >= top_k:
            break

    return evidence


# ============================================================
# DISPLAY RAW EVIDENCE
# ============================================================

def display_evidence(evidence):

    print(
        "\n" + "-" * 70
    )

    print(
        "SELECTED RAW EVIDENCE"
    )

    print(
        "-" * 70
    )

    for i, item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"\nEvidence {i}"
        )

        print(
            f"Paper: {item['paper_id']}"
        )

        print(
            f"Page: {item['page_number']}"
        )

        print(
            f"Score: "
            f"{item['reranking_score']:.4f}"
        )

        print(
            f"Reference Penalty: "
            f"{item['reference_penalty']}"
        )

        print(
            f"Text: "
            f"{item['text'][:600]}"
        )


# ============================================================
# DISPLAY CLEAN EVIDENCE
# ============================================================

def display_clean_evidence(
    clean_evidence
):

    print(
        "\n" + "-" * 70
    )

    print(
        "CLEAN EVIDENCE SENTENCES"
    )

    print(
        "-" * 70
    )

    for i, item in enumerate(
        clean_evidence,
        start=1
    ):

        print(
            f"\nClean Evidence {i}"
        )

        print(
            f"Paper: {item['paper_id']}"
        )

        print(
            f"Page: {item['page_number']}"
        )

        print(
            f"Relevance: "
            f"{item['relevance_score']}"
        )

        print(
            f"Text: {item['text']}"
        )


# ============================================================
# CONTROLLED LIMITATION ANSWER
# ============================================================

def build_controlled_limitation_answer(
    clean_evidence
):

    answers = []

    for item in clean_evidence:

        text = str(
            item.get("text", "")
        ).strip()

        if not text:
            continue

        # Remove trailing academic citation markers
        text = re.sub(
            r"\s\[\d{1,3}\]\s*$",
            "",
            text
        ).strip()

        # Remove accidental PDF line-break hyphenation
        text = re.sub(
            r"(\w)-\s+(\w)",
            r"\1\2",
            text
        )

        # Clean excessive whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if not text:
            continue

        answers.append(
            text
        )

    if not answers:

        return (
            "The retrieved academic evidence does not "
            "contain enough relevant information to "
            "answer this question reliably."
        )

    output = []

    for i, text in enumerate(
        answers,
        start=1
    ):

        output.append(
            f"**Limitation {i}:** {text}"
        )

    return "\n\n".join(
        output
    )


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def answer_question(question):

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        "\nQuestion:"
    )

    print(
        question
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\n[1/5] Retrieving relevant academic chunks..."
    )

    candidates = retrieve_candidates(
        question
    )

    print(
        f"Retrieved {len(candidates)} candidates."
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\n[2/5] Reranking and filtering evidence..."
    )

    ranked_candidates = rerank_candidates(
        question,
        candidates
    )

    print(
        "Remaining after filtering: "
        f"{len(ranked_candidates)}"
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\n[3/5] Selecting final evidence..."
    )

    evidence = select_evidence(
        question,
        ranked_candidates,
        FINAL_TOP_K
    )

    print(
        f"Selected {len(evidence)} evidence chunks."
    )

    display_evidence(
        evidence
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\n[4/5] Extracting relevant evidence sentences..."
    )

    clean_evidence = build_clean_evidence(
        evidence=evidence,
        question=question,
        max_items=FINAL_TOP_K
    )

    print(
        f"Clean evidence items: "
        f"{len(clean_evidence)}"
    )

    display_clean_evidence(
        clean_evidence
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print(
        "\n[5/5] Generating evidence-grounded answer..."
    )

    if not clean_evidence:

        answer = (
            "The retrieved academic evidence does not "
            "contain enough relevant information to "
            "answer this question reliably."
        )

    else:

        q_type = question_type(
            question
        )

        # ----------------------------------------------------
        # CONTROLLED PATH FOR LIMITATION QUESTIONS
        # ----------------------------------------------------

        if q_type == "LIMITATION":

            print(
                "Using controlled evidence-grounded "
                "limitation generation."
            )

            answer = build_controlled_limitation_answer(
                clean_evidence
            )

        # ----------------------------------------------------
        # LLM PATH FOR OTHER QUESTION TYPES
        # ----------------------------------------------------

        else:

            answer = generate_answer(
                question,
                clean_evidence
            )

    return {
        "question": question,
        "answer": answer,
        "evidence": clean_evidence
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your research question: "
    ).strip()

    if not question:

        print(
            "Please enter a research question."
        )

    else:

        result = answer_question(
            question
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            result["answer"]
        )