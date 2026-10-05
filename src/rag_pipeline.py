import os
import re
import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer, CrossEncoder

from llm_generator import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "chunks.csv"
)

FAISS_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "faiss_index.index"
)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)

INITIAL_TOP_K = 100
FINAL_TOP_K = 5


# ============================================================
# LOAD DATA
# ============================================================

print("Loading chunks...")

chunks_df = pd.read_csv(
    CHUNKS_PATH
)

print("Loading FAISS index...")

index = faiss.read_index(
    FAISS_PATH
)

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print(
    f"Total chunks available: "
    f"{len(chunks_df)}"
)

print(
    f"FAISS vectors: "
    f"{index.ntotal}"
)


# ============================================================
# COLUMN DETECTION
# ============================================================

def get_column(
    df,
    possible_names,
    default=None
):

    for name in possible_names:

        if name in df.columns:
            return name

    return default


PAPER_ID_COL = get_column(
    chunks_df,
    [
        "paper_id",
        "paperId",
        "id"
    ]
)

TITLE_COL = get_column(
    chunks_df,
    [
        "title",
        "paper_title"
    ]
)

PAGE_COL = get_column(
    chunks_df,
    [
        "page",
        "page_number",
        "page_no"
    ]
)

TEXT_COL = get_column(
    chunks_df,
    [
        "text",
        "chunk",
        "content"
    ]
)

YEAR_COL = get_column(
    chunks_df,
    [
        "year",
        "publication_year"
    ]
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_text(text):

    text = safe_text(text)

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_page(row):

    if PAGE_COL is None:
        return ""

    return safe_text(
        row.get(
            PAGE_COL,
            ""
        )
    )


def get_paper_id(row):

    if PAPER_ID_COL is None:
        return ""

    return safe_text(
        row.get(
            PAPER_ID_COL,
            ""
        )
    )


def get_title(row):

    if TITLE_COL is None:
        return ""

    return safe_text(
        row.get(
            TITLE_COL,
            ""
        )
    )


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    q = question.lower().strip()

    if (
        q.startswith("what is")
        or q.startswith("what are")
        or "define " in q
        or "definition of" in q
        or "meaning of" in q
    ):
        return "DEFINITION"

    if (
        q.startswith("how ")
        or "how does" in q
        or "how can" in q
        or "how is" in q
    ):
        return "HOW"

    if (
        "compare" in q
        or "comparison" in q
        or "difference between" in q
        or "differentiate" in q
        or "versus" in q
        or " vs " in q
    ):
        return "COMPARISON"

    if (
        "limitation" in q
        or "limitations" in q
        or "drawback" in q
        or "disadvantage" in q
        or "challenge" in q
    ):
        return "LIMITATION"

    if (
        "dataset" in q
        or "data set" in q
    ):
        return "DATASET"

    if (
        "method" in q
        or "technique" in q
        or "algorithm" in q
    ):
        return "TECHNIQUE"

    if (
        "metric" in q
        or "evaluation" in q
        or "performance" in q
    ):
        return "METRIC"

    return "GENERAL"


# ============================================================
# REFERENCE DETECTION
# ============================================================

def is_reference_chunk(text):

    text = safe_text(text)

    lower = text.lower()

    # Strong bibliography indicators
    strong_terms = [
        "references",
        "bibliography",
        "doi:",
        "arxiv:",
        "isbn",
        "issn",
        "proceedings"
    ]

    strong_count = 0

    for term in strong_terms:

        if term in lower:
            strong_count += 1

    if strong_count >= 1:

        # DOI / arXiv / bibliography are especially strong
        if (
            "doi:" in lower
            or "arxiv:" in lower
            or "bibliography" in lower
            or "references" in lower
        ):
            return True

    # URLs in academic reference-like chunks
    if "https://doi.org/" in lower:
        return True

    if "http://doi.org/" in lower:
        return True

    # Typical numbered references
    reference_pattern = re.compile(
        r"\[\d+\]\s+"
        r"[A-Z][A-Za-z\-]+.*"
        r"\b(19|20)\d{2}\b"
    )

    if reference_pattern.search(text):
        return True

    # Citation-heavy text
    citation_matches = re.findall(
        r"\[\d+\]",
        text
    )

    if len(citation_matches) >= 4:
        return True

    # Bibliographic phrases
    bibliography_phrases = [
        "arxiv preprint",
        "in the proceedings",
        "vol.",
        "pp.",
        "unpublished manuscript",
        "accessed:",
        "available:",
        "retrieved from"
    ]

    phrase_count = 0

    for phrase in bibliography_phrases:

        if phrase in lower:
            phrase_count += 1

    if phrase_count >= 1:
        return True

    return False


# ============================================================
# METADATA DETECTION
# ============================================================

def is_metadata_chunk(text):

    text = safe_text(text)

    lower = text.lower()

    metadata_terms = [
        "index terms",
        "keywords:",
        "key words:",
        "received:",
        "accepted:",
        "published:",
        "corresponding author",
        "email:",
        "copyright",
        "all rights reserved"
    ]

    for term in metadata_terms:

        if term in lower:
            return True

    return False


# ============================================================
# NAVIGATION / PROCEDURAL SENTENCE DETECTION
# ============================================================

def is_navigation_sentence(sentence):

    s = normalize_text(
        sentence
    )

    patterns = [

        "this article delineates",
        "this article presents",
        "this article discusses",

        "this paper discusses",
        "this paper presents",
        "this paper describes",
        "this paper introduces",

        "this study presents",
        "this study discusses",

        "we will present",
        "we present",
        "we summarize",
        "we introduce",
        "we discuss",
        "we describe",
        "we review",
        "we examine",
        "we investigate",

        "as follows",

        "following sections",
        "the following section",

        "the paper is organized",
        "this paper is organized",

        "existing analysis work",
        "existing research work",

        "future research directions",
        "prospective avenues for research",

        "points out prospective avenues",

        "applications of"
    ]

    for pattern in patterns:

        if pattern in s:
            return True

    if s.startswith("table "):
        return True

    if s.startswith("figure "):
        return True

    if s.startswith("fig."):
        return True

    return False


# ============================================================
# DEFINITION EXCLUSION
# ============================================================

def is_definition_excluded(sentence):

    s = normalize_text(
        sentence
    )

    excluded_patterns = [

        "we will present",
        "we present",
        "we summarize",
        "we introduce",
        "we discuss",
        "we describe",
        "we review",
        "we examine",
        "we investigate",

        "as follows",

        "following sections",
        "the following section",

        "this article delineates",
        "this article presents",
        "this article discusses",

        "this paper discusses",
        "this paper presents",
        "this paper describes",
        "this paper introduces",

        "this study presents",
        "this study discusses",

        "existing analysis work",
        "existing research work",

        "future research directions",
        "prospective avenues for research",

        "applications of",

        "table ",
        "figure ",
        "fig."
    ]

    for pattern in excluded_patterns:

        if pattern in s:
            return True

    return False


# ============================================================
# BAD SENTENCE DETECTION
# ============================================================

def is_bad_sentence(sentence):

    sentence = safe_text(
        sentence
    )

    if not sentence:
        return True

    if len(sentence) < 30:
        return True

    if is_navigation_sentence(
        sentence
    ):
        return True

    if is_reference_chunk(
        sentence
    ):
        return True

    if is_metadata_chunk(
        sentence
    ):
        return True

    lower = sentence.lower()

    # URLs
    if "https://" in lower:
        return True

    if "http://" in lower:
        return True

    # DOI
    if "doi.org/" in lower:
        return True

    # Strong bibliography indicators
    if "arxiv preprint" in lower:
        return True

    if "proceedings" in lower:
        return True

    # Multiple citation markers
    citations = re.findall(
        r"\[\d+\]",
        sentence
    )

    if len(citations) >= 2:
        return True

    # Author/bibliographic pattern
    if re.search(
        r'^[A-Z][A-Za-z\-]+,\s+["“]',
        sentence
    ):
        return True

    # Typical bibliography year pattern
    if re.search(
        r"\b(19|20)\d{2}\b.*"
        r"(arxiv|preprint|doi|proceedings)",
        lower
    ):
        return True

    return False


# ============================================================
# CONTENT QUALITY
# ============================================================

def content_quality(text):

    text = safe_text(
        text
    )

    if not text:
        return 0

    lower = text.lower()

    score = 0

    useful_terms = [

        "retrieval",
        "generation",
        "knowledge",
        "language model",
        "large language model",
        "external knowledge",
        "documents",
        "context",
        "semantic",
        "embedding",
        "vector",
        "query"
    ]

    for term in useful_terms:

        if term in lower:
            score += 2

    if len(text) > 300:
        score += 2

    if len(text) > 600:
        score += 1

    if is_reference_chunk(
        text
    ):
        score -= 20

    if is_metadata_chunk(
        text
    ):
        score -= 20

    if is_navigation_sentence(
        text
    ):
        score -= 15

    return score


# ============================================================
# DEFINITION SCORE
# ============================================================

def definition_score(text):

    text = safe_text(
        text
    )

    lower = text.lower()

    score = 0

    definition_patterns = [

        "is a technique",
        "is a method",
        "is an approach",
        "is a framework",
        "is a paradigm",

        "refers to",
        "defined as",
        "can be defined as",

        "is designed to",

        "combines",
        "integrates",
        "consists of",

        "enhances",
        "augments",

        "retrieves relevant",
        "external knowledge",
        "knowledge base",
        "retrieved documents",

        "explanatory",
        "generative model",
        "language model"
    ]

    for pattern in definition_patterns:

        if pattern in lower:
            score += 2

    if (
        "retrieval-augmented generation"
        in lower
    ):
        score += 5

    if (
        "retrieval augmented generation"
        in lower
    ):
        score += 5

    if "external knowledge" in lower:
        score += 5

    if "knowledge base" in lower:
        score += 4

    if "retrieved documents" in lower:
        score += 4

    if "document fragments" in lower:
        score += 4

    if (
        "generative model"
        in lower
    ):
        score += 3

    if is_navigation_sentence(
        text
    ):
        score -= 20

    if is_reference_chunk(
        text
    ):
        score -= 30

    return score


# ============================================================
# DEFINITION BOOST
# ============================================================

def definition_boost(text):

    text = safe_text(
        text
    )

    lower = text.lower()

    boost = 0

    if "retrieval augmentation" in lower:
        boost += 20

    if "retrieval-augmented generation" in lower:
        boost += 20

    if "retrieval augmented generation" in lower:
        boost += 20

    if "external knowledge" in lower:
        boost += 12

    if "knowledge base" in lower:
        boost += 10

    if "retrieved documents" in lower:
        boost += 10

    if "document fragments" in lower:
        boost += 10

    if "language model" in lower:
        boost += 4

    if is_navigation_sentence(
        text
    ):
        boost -= 30

    if is_reference_chunk(
        text
    ):
        boost -= 40

    return boost


# ============================================================
# INTENT SCORE
# ============================================================

def intent_score(
    text,
    qtype
):

    text = safe_text(
        text
    )

    lower = text.lower()

    score = 0

    if qtype == "DEFINITION":

        keywords = [

            "retrieval",
            "generation",
            "external knowledge",
            "knowledge base",
            "retrieved documents",
            "document fragments",
            "language model",
            "generative model",
            "ground"
        ]

    elif qtype == "HOW":

        keywords = [

            "step",
            "process",
            "retrieve",
            "query",
            "embedding",
            "rerank",
            "generate",
            "pipeline"
        ]

    elif qtype == "COMPARISON":

        keywords = [

            "difference",
            "compared",
            "versus",
            "whereas",
            "while",
            "however"
        ]

    elif qtype == "LIMITATION":

        keywords = [

            "limitation",
            "challenge",
            "drawback",
            "problem",
            "hallucination",
            "cost",
            "latency"
        ]

    else:

        keywords = [

            "retrieval",
            "generation",
            "knowledge",
            "language model",
            "documents",
            "context"
        ]

    for keyword in keywords:

        if keyword in lower:
            score += 2

    return score


# ============================================================
# REFERENCE PENALTY
# ============================================================

def reference_penalty(text):

    text = safe_text(
        text
    )

    lower = text.lower()

    penalty = 0

    if is_reference_chunk(
        text
    ):
        penalty += 40

    if is_metadata_chunk(
        text
    ):
        penalty += 30

    if is_navigation_sentence(
        text
    ):
        penalty += 25

    if "doi:" in lower:
        penalty += 10

    if "arxiv:" in lower:
        penalty += 10

    if "arxiv preprint" in lower:
        penalty += 15

    if "http://" in lower:
        penalty += 8

    if "https://" in lower:
        penalty += 8

    return penalty


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates(
    question,
    top_k=INITIAL_TOP_K
):

    print(
        f"Retrieving {top_k} candidates..."
    )

    query_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True
        )
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    candidates = []

    for rank, idx in enumerate(
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(chunks_df):
            continue

        row = chunks_df.iloc[
            idx
        ]

        text = safe_text(
            row.get(
                TEXT_COL,
                ""
            )
        )

        if not text:
            continue

        candidate = row.to_dict()

        candidate["_faiss_rank"] = (
            rank + 1
        )

        candidate["_faiss_score"] = (
            float(
                distances[0][rank]
            )
        )

        candidates.append(
            candidate
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

    qtype = question_type(
        question
    )

    texts = []

    valid_candidates = []

    for item in candidates:

        text = safe_text(
            item.get(
                TEXT_COL,
                ""
            )
        )

        if not text:
            continue

        texts.append(
            text
        )

        valid_candidates.append(
            item
        )

    if not texts:
        return []

    pairs = [
        [
            question,
            text
        ]
        for text in texts
    ]

    scores = reranker.predict(
        pairs
    )

    results = []

    for item, cross_score in zip(
        valid_candidates,
        scores
    ):

        text = safe_text(
            item.get(
                TEXT_COL,
                ""
            )
        )

        # Hard removal
        if is_reference_chunk(
            text
        ):
            continue

        if is_metadata_chunk(
            text
        ):
            continue

        d_score = 0
        d_boost = 0

        if qtype == "DEFINITION":

            d_score = definition_score(
                text
            )

            d_boost = definition_boost(
                text
            )

        quality = content_quality(
            text
        )

        intent = intent_score(
            text,
            qtype
        )

        ref_penalty = reference_penalty(
            text
        )

        final_score = (
            float(cross_score)
            + quality
            + intent
            + d_score
            + d_boost
            - ref_penalty
        )

        result = dict(
            item
        )

        result["_cross_score"] = (
            float(cross_score)
        )

        result["_definition_score"] = (
            d_score
        )

        result["_definition_boost"] = (
            d_boost
        )

        result["_quality_score"] = (
            quality
        )

        result["_intent_score"] = (
            intent
        )

        result["_reference_penalty"] = (
            ref_penalty
        )

        result["_final_score"] = (
            final_score
        )

        results.append(
            result
        )

    results.sort(
        key=lambda x:
        x["_final_score"],
        reverse=True
    )

    return results


# ============================================================
# SELECT EVIDENCE
# ============================================================

def select_evidence(
    candidates,
    question,
    final_k=FINAL_TOP_K
):

    qtype = question_type(
        question
    )

    selected = []

    seen_text = set()

    paper_counts = {}

    for candidate in candidates:

        text = safe_text(
            candidate.get(
                TEXT_COL,
                ""
            )
        )

        if not text:
            continue

        if is_reference_chunk(
            text
        ):
            continue

        if is_metadata_chunk(
            text
        ):
            continue

        normalized = normalize_text(
            text
        )

        if normalized in seen_text:
            continue

        if qtype == "DEFINITION":

            d_score = candidate.get(
                "_definition_score",
                0
            )

            d_boost = candidate.get(
                "_definition_boost",
                0
            )

            if (
                d_score < 5
                and d_boost < 10
            ):
                continue

        paper_id = get_paper_id(
            candidate
        )

        current_count = (
            paper_counts.get(
                paper_id,
                0
            )
        )

        if current_count >= 2:
            continue

        selected.append(
            candidate
        )

        seen_text.add(
            normalized
        )

        paper_counts[
            paper_id
        ] = current_count + 1

        if len(selected) >= final_k:
            break

    return selected


# ============================================================
# CLEAN SENTENCE
# ============================================================

def clean_sentence(sentence):

    sentence = safe_text(
        sentence
    )

    sentence = re.sub(
        r"\s+",
        " ",
        sentence
    )

    sentence = re.sub(
        r"^[•●▪◦\-]+\s*",
        "",
        sentence
    )

    # Remove simple numeric citations
    sentence = re.sub(
        r"\[\d+(?:\s*,\s*\d+)*\]",
        "",
        sentence
    )

    # Fix hyphenation created by PDF extraction
    sentence = re.sub(
        r"(\w)-\s+(\w)",
        r"\1\2",
        sentence
    )

    sentence = re.sub(
        r"\s+",
        " ",
        sentence
    )

    return sentence.strip()


# ============================================================
# SENTENCE RELEVANCE
# ============================================================

def sentence_relevance(
    sentence,
    qtype
):

    sentence = clean_sentence(
        sentence
    )

    lower = sentence.lower()

    score = 0

    if qtype == "DEFINITION":

        positive_terms = [

            "retrieval augmentation",

            "retrieval-augmented generation",

            "retrieval augmented generation",

            "external knowledge",

            "knowledge base",

            "retrieved documents",

            "document fragments",

            "retrieve relevant",

            "ground the model",

            "generative model",

            "language model",

            "large language model",

            "external database"
        ]

        for term in positive_terms:

            if term in lower:
                score += 3

        if (
            "external knowledge"
            in lower
            and "model" in lower
        ):
            score += 5

        if (
            "knowledge base"
            in lower
            and (
                "generative model"
                in lower
                or
                "language model"
                in lower
            )
        ):
            score += 5

        if (
            "retrieved documents"
            in lower
            and "language model"
            in lower
        ):
            score += 5

        if (
            "document fragments"
            in lower
        ):
            score += 5

        if (
            "retrieval augmentation"
            in lower
            and "external knowledge"
            in lower
        ):
            score += 6

    elif qtype == "HOW":

        positive_terms = [

            "retrieve",
            "query",
            "embedding",
            "vector",
            "rerank",
            "generate",
            "context",
            "pipeline"
        ]

        for term in positive_terms:

            if term in lower:
                score += 2

    elif qtype == "COMPARISON":

        positive_terms = [

            "difference",
            "compared",
            "whereas",
            "while",
            "however",
            "versus"
        ]

        for term in positive_terms:

            if term in lower:
                score += 2

    else:

        positive_terms = [

            "retrieval",
            "generation",
            "knowledge",
            "language model",
            "documents",
            "context"
        ]

        for term in positive_terms:

            if term in lower:
                score += 2

    negative_terms = [

        "we will present",
        "we present",
        "we summarize",
        "we introduce",
        "we discuss",
        "we describe",
        "we review",
        "we examine",
        "we investigate",

        "as follows",

        "following sections",
        "the following section",

        "this article delineates",
        "this article presents",
        "this article discusses",

        "this paper discusses",
        "this paper presents",
        "this paper describes",
        "this paper introduces",

        "this study presents",
        "this study discusses",

        "existing analysis work",
        "existing research work",

        "future research directions",
        "prospective avenues for research",

        "applications of",

        "table ",
        "figure ",
        "fig."
    ]

    for term in negative_terms:

        if term in lower:
            score -= 20

    return score


# ============================================================
# EXTRACT CLEAN EVIDENCE
# ============================================================

def extract_clean_evidence(
    selected,
    question
):

    qtype = question_type(
        question
    )

    evidence = []

    seen_sentences = set()

    for candidate in selected:

        text = safe_text(
            candidate.get(
                TEXT_COL,
                ""
            )
        )

        if not text:
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # Remove complete bibliography chunks BEFORE
        # sentence splitting.
        # ----------------------------------------------------

        if is_reference_chunk(
            text
        ):
            continue

        if is_metadata_chunk(
            text
        ):
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence = clean_sentence(
                sentence
            )

            if not sentence:
                continue

            if is_bad_sentence(
                sentence
            ):
                continue

            if is_navigation_sentence(
                sentence
            ):
                continue

            if (
                qtype == "DEFINITION"
                and is_definition_excluded(
                    sentence
                )
            ):
                continue

            normalized = normalize_text(
                sentence
            )

            if normalized in seen_sentences:
                continue

            score = sentence_relevance(
                sentence,
                qtype
            )

            # ------------------------------------------------
            # DEFINITION QUESTIONS
            # ------------------------------------------------

            if qtype == "DEFINITION":

                substantive_signals = [

                    "external knowledge",

                    "knowledge base",

                    "retrieved documents",

                    "document fragments",

                    "retrieval augmentation",

                    "retrieval-augmented generation",

                    "retrieval augmented generation",

                    "retrieve relevant",

                    "ground the model",

                    "generative model"
                ]

                has_substantive_signal = any(
                    signal in normalized
                    for signal in substantive_signals
                )

                if not has_substantive_signal:
                    continue

                if score < 5:
                    continue

            else:

                if score < 2:
                    continue

            evidence.append(
                {
                    "paper_id":
                        get_paper_id(
                            candidate
                        ),

                    "title":
                        get_title(
                            candidate
                        ),

                    "page":
                        get_page(
                            candidate
                        ),

                    "text":
                        sentence,

                    "score":
                        score
                }
            )

            seen_sentences.add(
                normalized
            )

    # Strongest evidence first
    evidence.sort(
        key=lambda x:
        x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Remove near-duplicate evidence
    # --------------------------------------------------------

    final_evidence = []

    for item in evidence:

        current = normalize_text(
            item["text"]
        )

        duplicate = False

        for previous in final_evidence:

            previous_text = normalize_text(
                previous["text"]
            )

            # Basic overlap check
            current_words = set(
                current.split()
            )

            previous_words = set(
                previous_text.split()
            )

            if not current_words:
                continue

            overlap = (
                len(
                    current_words
                    & previous_words
                )
                /
                len(current_words)
            )

            if overlap > 0.75:

                duplicate = True
                break

        if not duplicate:

            final_evidence.append(
                item
            )

    # Definition answers should remain concise.
    if qtype == "DEFINITION":

        final_evidence = (
            final_evidence[:3]
        )

    else:

        final_evidence = (
            final_evidence[:5]
        )

    return final_evidence


# ============================================================
# DISPLAY CLEAN EVIDENCE
# ============================================================

def display_clean_evidence(
    evidence
):

    print(
        f"\nClean Evidence items: "
        f"{len(evidence)}"
    )

    for i, item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"\nClean Evidence {i} "
            f"{item['paper_id']} "
            f"Page {item['page']}:"
        )

        print(
            f"\"{item['text']}\""
        )


# ============================================================
# DISPLAY SELECTED EVIDENCE
# ============================================================

def display_evidence(
    evidence
):

    for i, item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"\nEvidence {i} "
            f"{get_paper_id(item)} "
            f"Page {get_page(item)}:"
        )

        print(
            safe_text(
                item.get(
                    TEXT_COL,
                    ""
                )
            )[:700]
        )


# ============================================================
# MAIN ANSWER PIPELINE
# ============================================================

def answer_question(
    question
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG ACADEMIC RESEARCH ASSISTANT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestion:\n{question}"
    )

    # ========================================================
    # STEP 1
    # RETRIEVAL
    # ========================================================

    print(
        "\n[1/5] "
        "Retrieving relevant academic chunks..."
    )

    candidates = retrieve_candidates(
        question,
        INITIAL_TOP_K
    )

    print(
        f"Retrieved "
        f"{len(candidates)} candidates."
    )

    # ========================================================
    # STEP 2
    # RERANKING
    # ========================================================

    print(
        "\n[2/5] "
        "Reranking and filtering evidence..."
    )

    ranked_candidates = rerank_candidates(
        question,
        candidates
    )

    print(
        "Remaining after filtering: "
        f"{len(ranked_candidates)}"
    )

    qtype = question_type(
        question
    )

    print(
        "\nTOP RERANKED CANDIDATES:"
    )

    for i, item in enumerate(
        ranked_candidates[:10],
        start=1
    ):

        print(
            f"\n#{i} "
            f"{get_paper_id(item)} "
            f"Page {get_page(item)} "
            f"Score "
            f"{item['_final_score']:.4f}"
        )

        print(
            f"Definition Score: "
            f"{item['_definition_score']}"
        )

        print(
            f"Definition Boost: "
            f"{item['_definition_boost']}"
        )

        print(
            "Text: "
            +
            safe_text(
                item.get(
                    TEXT_COL,
                    ""
                )
            )[:700]
        )

    # ========================================================
    # STEP 3
    # SELECT EVIDENCE
    # ========================================================

    print(
        "\n[3/5] "
        "Selecting final evidence..."
    )

    evidence = select_evidence(
        ranked_candidates,
        question,
        FINAL_TOP_K
    )

    print(
        f"Selected "
        f"{len(evidence)} evidence chunks."
    )

    display_evidence(
        evidence
    )

    # ========================================================
    # STEP 4
    # CLEAN EVIDENCE
    # ========================================================

    print(
        "\n[4/5] "
        "Extracting relevant evidence sentences..."
    )

    clean_evidence = extract_clean_evidence(
        evidence,
        question
    )

    display_clean_evidence(
        clean_evidence
    )

    # ========================================================
    # STEP 5
    # GENERATE ANSWER
    # ========================================================

    print(
        "\n[5/5] "
        "Generating evidence-grounded answer..."
    )

    print(
        f"Detected question type: "
        f"{qtype}"
    )

    if not clean_evidence:

        answer = (
            "The retrieved academic evidence "
            "does not contain enough relevant "
            "information to answer this question "
            "reliably."
        )

    else:

        # IMPORTANT:
        # Your llm_generator.py expects:
        #
        # generate_answer(question, evidence)
        #
        # NOT:
        # generate_answer(question, context=...)
        #
        answer = generate_answer(
            question,
            clean_evidence
        )

    return {
        "question":
            question,

        "answer":
            answer,

        "evidence":
            clean_evidence
    }


# ============================================================
# PROGRAM ENTRY POINT
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
            "\n"
            + "=" * 70
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

        print(
            "=" * 70
        )