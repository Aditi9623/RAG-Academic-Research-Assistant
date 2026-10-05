import pandas as pd
import re


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "dataset/processed/chunks.csv"

TOP_K = 3

df = pd.read_csv(CHUNKS_PATH)


# ============================================================
# REFERENCE / BIBLIOGRAPHY DETECTION
# ============================================================

def reference_penalty(text):
    """
    Detect bibliography/reference-heavy chunks.

    Higher value = more likely to be reference content.
    """

    text = str(text).strip()
    text_lower = text.lower()

    penalty = 0

    # --------------------------------------------------------
    # Citation numbers such as [42], [105]
    # --------------------------------------------------------

    citation_count = len(
        re.findall(
            r"\[\s*\d{1,3}\s*\]",
            text
        )
    )

    # --------------------------------------------------------
    # Publication years
    # --------------------------------------------------------

    year_count = len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text
        )
    )

    # --------------------------------------------------------
    # Academic reference indicators
    # --------------------------------------------------------

    arxiv_count = len(
        re.findall(
            r"\barxiv\b",
            text_lower
        )
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

    # --------------------------------------------------------
    # Strong reference indicators
    # --------------------------------------------------------

    if citation_count >= 2:
        penalty += citation_count * 2

    if year_count >= 3:
        penalty += year_count * 2

    penalty += arxiv_count * 5
    penalty += doi_count * 5
    penalty += proceedings_count * 4

    # --------------------------------------------------------
    # Reference-list headings
    # --------------------------------------------------------

    reference_headers = [
        "references",
        "bibliography",
        "reference list",
        "references and notes"
    ]

    for header in reference_headers:

        if text_lower.startswith(header):

            penalty += 100

    # --------------------------------------------------------
    # Strong combined bibliography pattern
    # --------------------------------------------------------

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
# REFERENCE CHUNK CHECK
# ============================================================

def is_reference_chunk(text):

    penalty = reference_penalty(text)

    return penalty >= 10


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    # Remove citation numbers
    text = re.sub(
        r"\[\s*\d{1,3}\s*\]",
        "",
        text
    )

    # Remove hyphenation caused by PDF extraction
    text = re.sub(
        r"-\s+",
        "",
        text
    )

    # Keep only alphanumeric characters
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TEXT SIMILARITY
# ============================================================

def text_similarity(text_a, text_b):

    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a or not b:
        return 0.0

    words_a = set(a.split())
    words_b = set(b.split())

    if not words_a or not words_b:
        return 0.0

    intersection = len(
        words_a & words_b
    )

    union = len(
        words_a | words_b
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# EXACT / NEAR DUPLICATE CHECK
# ============================================================

def is_duplicate_text(
    text,
    previous_texts,
    threshold=0.85
):

    normalized = normalize_text(
        text
    )

    if not normalized:
        return True

    for previous in previous_texts:

        previous_normalized = (
            normalize_text(previous)
        )

        # Exact duplicate
        if normalized == previous_normalized:
            return True

        # Near duplicate
        similarity = text_similarity(
            normalized,
            previous_normalized
        )

        if similarity >= threshold:
            return True

    return False


# ============================================================
# METADATA / NON-CONTENT DETECTION
# ============================================================

def is_metadata_chunk(text):

    text = str(text).strip()

    lower = text.lower()


    # --------------------------------------------------------
    # Index terms
    # --------------------------------------------------------

    if re.search(
        r"\bindex\s+terms?\b",
        lower
    ):

        return True


    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    if lower.startswith(
        "keywords:"
    ):

        return True


    # --------------------------------------------------------
    # Copyright / publication metadata
    # --------------------------------------------------------

    metadata_patterns = [

        "© ",
        "copyright",
        "all rights reserved",
        "received:",
        "accepted:",
        "published:",
        "available online",
        "issn",
        "isbn"

    ]


    for pattern in metadata_patterns:

        if pattern in lower:

            return True


    return False


# ============================================================
# NAVIGATION / PAPER-STRUCTURE DETECTION
# ============================================================

def is_navigation_chunk(text):

    text = str(text).strip()

    lower = text.lower()


    navigation_patterns = [

        "this paper is organized",
        "the paper is organized",
        "the remainder of this paper",
        "the rest of this paper",
        "this article is organized",
        "the following sections",
        "in the next section",
        "section ii",
        "section iii",
        "section iv",
        "section v"

    ]


    for pattern in navigation_patterns:

        if pattern in lower:

            return True


    return False


# ============================================================
# DEFINITION RELEVANCE
# ============================================================

def definition_relevance(
    text,
    question
):

    text_lower = str(text).lower()

    question_lower = str(
        question
    ).lower()


    score = 0


    # --------------------------------------------------------
    # Strong definition terms
    # --------------------------------------------------------

    definition_terms = [

        "retrieval-augmented generation",
        "retrieval augmented generation",
        "rag is",
        "rag refers",
        "rag enhances",
        "rag retrieves",
        "retrieving relevant document",
        "retrieves relevant document",
        "external knowledge",
        "external database",
        "external databases",
        "external knowledge base",
        "knowledge base",
        "document chunks",
        "semantic similarity"

    ]


    for term in definition_terms:

        if term in text_lower:

            score += 2


    # --------------------------------------------------------
    # Strong penalty for unrelated limitation content
    # --------------------------------------------------------

    limitation_terms = [

        "inefficien",
        "insufficient",
        "multi-step reasoning",
        "limited scope",
        "drawback",
        "limitation",
        "challenge",
        "future work",
        "future research",
        "iterative retrieval"

    ]


    for term in limitation_terms:

        if term in text_lower:

            score -= 2


    # --------------------------------------------------------
    # Remove index terms from consideration
    # --------------------------------------------------------

    if "index terms" in text_lower:

        score -= 20


    return score


# ============================================================
# GENERAL QUESTION RELEVANCE
# ============================================================

def question_relevance(
    text,
    question
):

    text_lower = str(text).lower()

    question_lower = str(
        question
    ).lower()


    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question_lower
        )
    )


    text_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            text_lower
        )
    )


    if not question_words or not text_words:

        return 0


    overlap = len(
        question_words
        & text_words
    )


    return overlap


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    q = str(
        question
    ).lower()


    if any(
        word in q
        for word in [
            "what is",
            "define",
            "definition",
            "meaning"
        ]
    ):

        return "DEFINITION"


    if any(
        word in q
        for word in [
            "limitation",
            "limitations",
            "drawback",
            "drawbacks",
            "challenge",
            "challenges"
        ]
    ):

        return "LIMITATION"


    if any(
        word in q
        for word in [
            "compare",
            "comparison",
            "difference",
            "differences"
        ]
    ):

        return "COMPARISON"


    if any(
        word in q
        for word in [
            "how does",
            "how do",
            "how can",
            "how is",
            "how are",
            "how to"
        ]
    ):

        return "HOW"


    return "GENERAL"


# ============================================================
# SELECT EVIDENCE
# ============================================================

def select_evidence(
    ranked_results,
    top_k=TOP_K,
    question=""
):

    evidence = []

    used_chunks = set()

    used_papers = set()

    previous_texts = []


    qtype = question_type(
        question
    )


    # ========================================================
    # FIRST PASS
    # ========================================================

    candidates = []


    for score, index_id in ranked_results:

        score = float(score)

        row = df.iloc[index_id]

        chunk_id = row["chunk_id"]

        text = str(
            row["text"]
        ).strip()


        if not text:

            continue


        # ----------------------------------------------------
        # Same chunk ID
        # ----------------------------------------------------

        if chunk_id in used_chunks:

            continue


        # ----------------------------------------------------
        # Reference chunks
        # ----------------------------------------------------

        if is_reference_chunk(text):

            continue


        # ----------------------------------------------------
        # Metadata chunks
        # ----------------------------------------------------

        if is_metadata_chunk(text):

            continue


        # ----------------------------------------------------
        # Navigation chunks
        # ----------------------------------------------------

        if is_navigation_chunk(text):

            continue


        # ----------------------------------------------------
        # Duplicate content across papers
        # ----------------------------------------------------

        if is_duplicate_text(
            text,
            previous_texts,
            threshold=0.85
        ):

            continue


        # ----------------------------------------------------
        # Question-specific relevance
        # ----------------------------------------------------

        relevance = question_relevance(
            text,
            question
        )


        if qtype == "DEFINITION":

            relevance += (
                definition_relevance(
                    text,
                    question
                )
            )


        candidates.append(
            {
                "score": score,
                "index_id": index_id,
                "row": row,
                "text": text,
                "relevance": relevance
            }
        )


    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda x: (
            x["relevance"],
            x["score"]
        ),
        reverse=True
    )


    # ========================================================
    # SECOND PASS
    # ========================================================

    for candidate in candidates:

        score = candidate["score"]

        row = candidate["row"]

        text = candidate["text"]

        chunk_id = row["chunk_id"]

        paper_id = row["paper_id"]


        # ----------------------------------------------------
        # Duplicate safety
        # ----------------------------------------------------

        if chunk_id in used_chunks:

            continue


        if is_duplicate_text(
            text,
            previous_texts,
            threshold=0.85
        ):

            continue


        # ----------------------------------------------------
        # For definitions:
        # avoid filling all slots with limitation chunks
        # ----------------------------------------------------

        if qtype == "DEFINITION":

            definition_score = (
                definition_relevance(
                    text,
                    question
                )
            )


            if definition_score < 0:

                continue


        # ----------------------------------------------------
        # Keep evidence
        # ----------------------------------------------------

        evidence_item = {

            "paper_id": paper_id,

            "page_number": int(
                row["page_number"]
            ),

            "chunk_id": chunk_id,

            "reranking_score": score,

            "reference_penalty": reference_penalty(
                text
            ),

            "text": text

        }


        evidence.append(
            evidence_item
        )


        used_chunks.add(
            chunk_id
        )


        previous_texts.append(
            text
        )


        used_papers.add(
            paper_id
        )


        if len(evidence) >= top_k:

            break


    return evidence