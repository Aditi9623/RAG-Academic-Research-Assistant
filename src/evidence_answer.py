import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "dataset/processed/chunks.csv"

MAX_SENTENCES_PER_EVIDENCE = 3
MAX_EVIDENCE_ITEMS = 5


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    return pd.read_csv(CHUNKS_PATH)


# ============================================================
# REFERENCE DETECTION
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

    if text_lower.startswith(
        (
            "references",
            "bibliography",
            "reference list",
            "references and notes"
        )
    ):
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

    if text_lower.startswith(
        (
            "references",
            "bibliography",
            "reference list",
            "references and notes"
        )
    ):
        return True

    if citation_count >= 6:
        return True

    if (
        citation_count >= 3
        and year_count >= 3
    ):
        return True

    strong_reference_terms = [
        "proceedings of the",
        "advances in neural information processing",
        "association for computational linguistics",
        "journal of",
        "doi:",
        "arxiv.org"
    ]

    matches = sum(
        1
        for term in strong_reference_terms
        if term in text_lower
    )

    if matches >= 2 and citation_count >= 1:
        return True

    if (
        len(words) < 120
        and citation_count >= 2
        and year_count >= 2
    ):
        return True

    return False


# ============================================================
# METADATA DETECTION
# ============================================================

def metadata_penalty(text):

    text = str(text).strip()
    text_lower = text.lower()

    penalty = 0
    words = text.split()

    if (
        len(words) < 180
        and (
            "abstract—" in text_lower
            or "abstract -" in text_lower
            or text_lower.startswith("abstract")
        )
    ):
        penalty += 8

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

    if affiliation_count >= 2 and len(words) < 180:
        penalty += 5

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
# SENTENCE METADATA FILTER
# ============================================================

def is_metadata_sentence(sentence):

    text = str(sentence).strip()
    text_lower = text.lower()

    words = text.split()

    if len(words) < 6:
        return True

    metadata_terms = [
        "abstract—",
        "abstract -",
        "authors",
        "corresponding author",
        "department of",
        "university",
        "institute",
        "school of",
        "college of",
        "email:",
        "retrieval-augmented generation for large language models: a survey"
    ]

    matches = sum(
        1
        for term in metadata_terms
        if term in text_lower
    )

    if matches >= 2:
        return True

    if (
        len(words) < 25
        and (
            "survey" in text_lower
            or "review" in text_lower
        )
        and not any(
            verb in text_lower
            for verb in [
                "is",
                "are",
                "involves",
                "uses",
                "provides",
                "allows",
                "enables",
                "causes",
                "creates",
                "limits",
                "restricts"
            ]
        )
    ):
        return True

    return False


# ============================================================
# PAPER NAVIGATION FILTER
# ============================================================

def is_navigation_sentence(sentence):

    """
    Detect sentences that describe the paper itself instead
    of providing substantive research evidence.

    Example rejected sentence:

    "At the end, this article delineates the challenges
    currently faced and points out prospective avenues
    for research and development."

    This is paper-navigation text, not a specific limitation.
    """

    text = str(sentence).strip()
    lower = text.lower()

    if not lower:
        return True

    # --------------------------------------------------------
    # Direct paper-navigation patterns
    # --------------------------------------------------------

    navigation_patterns = [

        r"\bthis article delineates\b",
        r"\bthis article discusses\b",
        r"\bthis article presents\b",
        r"\bthis article reviews\b",
        r"\bthis article examines\b",
        r"\bthis article identifies\b",

        r"\bthis paper delineates\b",
        r"\bthis paper discusses\b",
        r"\bthis paper presents\b",
        r"\bthis paper reviews\b",
        r"\bthis paper examines\b",
        r"\bthis paper identifies\b",

        r"\bthis study delineates\b",
        r"\bthis study discusses\b",
        r"\bthis study presents\b",
        r"\bthis study reviews\b",
        r"\bthis study examines\b",
        r"\bthis study identifies\b",

        r"\bthe article delineates\b",
        r"\bthe article discusses\b",
        r"\bthe article presents\b",
        r"\bthe article reviews\b",
        r"\bthe article examines\b",
        r"\bthe article identifies\b",

        r"\bthe paper delineates\b",
        r"\bthe paper discusses\b",
        r"\bthe paper presents\b",
        r"\bthe paper reviews\b",
        r"\bthe paper examines\b",
        r"\bthe paper identifies\b",

        r"\bthe study delineates\b",
        r"\bthe study discusses\b",
        r"\bthe study presents\b",
        r"\bthe study reviews\b",
        r"\bthe study examines\b",
        r"\bthe study identifies\b",

        r"^at the end,?\s+(?:this|the)\s+(?:article|paper|study)\b",

        r"\bpoints out prospective avenues\b",
        r"\bprospective avenues for research\b",
        r"\bfuture research directions\b",
        r"\bfuture work is discussed\b",
        r"\bfuture work includes\b",

        r"\bthe following sections\b",
        r"\bthe remainder of this paper\b",
        r"\bthe rest of this paper\b",

        r"\bthe paper is organized as follows\b",
        r"\bthe article is organized as follows\b",
        r"\bthis paper is organized as follows\b",

        r"^in this paper, we (?:discuss|present|review|examine|survey|identify)\b",
        r"^in this article, we (?:discuss|present|review|examine|survey|identify)\b",
        r"^in this study, we (?:discuss|present|review|examine|survey|identify)\b"
    ]

    for pattern in navigation_patterns:

        if re.search(pattern, lower):
            return True

    # --------------------------------------------------------
    # Generic paper-description detection
    # --------------------------------------------------------

    paper_terms = [
        "this article",
        "this paper",
        "this study",
        "the article",
        "the paper",
        "the study"
    ]

    navigation_verbs = [
        "delineates",
        "discusses",
        "presents",
        "reviews",
        "examines",
        "summarizes",
        "surveys",
        "identifies",
        "points out"
    ]

    if (
        any(term in lower for term in paper_terms)
        and any(term in lower for term in navigation_verbs)
    ):

        # If the sentence only says that the paper
        # discusses challenges, reject it.
        concrete_limitation_terms = [
            "insufficient",
            "inefficient",
            "limitation",
            "limited",
            "constrain",
            "constrained",
            "restrict",
            "restricted",
            "difficulty",
            "difficult",
            "challenging",
            "noise",
            "noisy",
            "redundant",
            "costly",
            "lack of",
            "brittle",
            "fails",
            "unable"
        ]

        if not any(
            term in lower
            for term in concrete_limitation_terms
        ):
            return True

    # --------------------------------------------------------
    # Research-direction/navigation phrases
    # --------------------------------------------------------

    navigation_only_phrases = [
        "challenges currently faced",
        "avenues for research",
        "research and development",
        "future research",
        "research directions",
        "prospective avenues"
    ]

    if any(
        phrase in lower
        for phrase in navigation_only_phrases
    ):

        concrete_limitation_terms = [
            "insufficient",
            "inefficient",
            "limitation",
            "limited",
            "constrain",
            "constrained",
            "restrict",
            "restricted",
            "difficulty",
            "difficult",
            "challenging",
            "noise",
            "noisy",
            "redundant",
            "costly",
            "lack of",
            "brittle",
            "fails",
            "unable"
        ]

        if not any(
            term in lower
            for term in concrete_limitation_terms
        ):
            return True

    return False


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        )

        if len(sentence.split()) < 5:
            continue

        if is_metadata_sentence(sentence):
            continue

        if is_navigation_sentence(sentence):
            continue

        cleaned.append(sentence)

    return cleaned


# ============================================================
# QUESTION KEYWORDS
# ============================================================

def question_keywords(question):

    stop_words = {
        "what",
        "are",
        "is",
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "and",
        "or",
        "with",
        "from",
        "using",
        "used",
        "does",
        "do",
        "how",
        "can",
        "could",
        "would",
        "should",
        "main",
        "major",
        "important",
        "about",
        "please",
        "explain",
        "give",
        "list",
        "their",
        "its",
        "this",
        "that"
    }

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
        question.lower()
    )

    return [
        word
        for word in words
        if word not in stop_words
    ]


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    q = question.lower()

    if any(
        term in q
        for term in [
            "limitation",
            "limitations",
            "challenge",
            "challenges",
            "drawback",
            "drawbacks"
        ]
    ):
        return "LIMITATION"

    if (
        "dataset" in q
        or "datasets" in q
        or "training data" in q
        or "testing data" in q
    ):
        return "DATASET"

    if any(
        term in q
        for term in [
            "technique",
            "techniques",
            "method",
            "methods",
            "algorithm",
            "algorithms",
            "model",
            "models",
            "approach",
            "approaches"
        ]
    ):
        return "TECHNIQUE"

    if any(
        term in q
        for term in [
            "metric",
            "metrics",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "auroc",
            "performance",
            "evaluation"
        ]
    ):
        return "METRIC"

    if (
        "compare" in q
        or "comparison" in q
        or "difference" in q
    ):
        return "COMPARISON"

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

    return "GENERAL"


# ============================================================
# LIMITATION-SPECIFIC SCORING
# ============================================================

def limitation_score(sentence):

    text = sentence.lower()

    score = 0

    strong_terms = [
        "insufficient",
        "inefficien",
        "inefficiency",
        "limited scope",
        "limitation",
        "limitations",
        "challenge",
        "challenges",
        "drawback",
        "drawbacks",
        "difficulty",
        "difficulties",
        "problem",
        "problems",
        "restrict",
        "restricted",
        "constrain",
        "constrained",
        "shortcoming",
        "shortcomings",
        "brittle",
        "costly",
        "time-consuming",
        "time consuming",
        "noisy",
        "noise",
        "redundant",
        "generalize",
        "generalization",
        "hallucination",
        "hard to",
        "difficult to",
        "unable to",
        "fails to",
        "lack of"
    ]

    for term in strong_terms:

        if term in text:
            score += 3

    limitation_patterns = [
        "can lead to",
        "may lead to",
        "leads to",
        "poses a challenge",
        "pose challenges",
        "is insufficient",
        "are insufficient",
        "is difficult",
        "are difficult",
        "is costly",
        "are costly",
        "is restricted",
        "are restricted",
        "is constrained",
        "are constrained",
        "making it difficult",
        "making it challenging",
        "thereby constraining",
        "thereby limiting",
        "which limits",
        "which restricts",
        "which constrains"
    ]

    for pattern in limitation_patterns:

        if pattern in text:
            score += 4

    return score


# ============================================================
# NON-LIMITATION PENALTY
# ============================================================

def non_limitation_penalty(sentence):

    text = sentence.lower()

    penalty = 0

    definition_patterns = [
        "is a technique",
        "is an approach",
        "is a method",
        "refers to",
        "consists of",
        "involves retrieving",
        "retrieves relevant",
        "enhances llms by",
        "enhances large language models",
        "provides relevant",
        "allows llms",
        "enables llms"
    ]

    for pattern in definition_patterns:

        if pattern in text:
            penalty += 7

    solution_patterns = [
        "promising solution",
        "potential solution",
        "can improve",
        "can be used to",
        "leveraging",
        "researchers use",
        "researchers have optimized",
        "optimized the retrieval",
        "offers a solution",
        "aid in developing",
        "helps to",
        "helps improve"
    ]

    for pattern in solution_patterns:

        if pattern in text:
            penalty += 6

    positive_patterns = [
        "remarkable success",
        "excellent capability",
        "strong capability",
        "powerful capability",
        "has achieved",
        "have achieved",
        "successfully"
    ]

    for pattern in positive_patterns:

        if pattern in text:
            penalty += 4

    return penalty


# ============================================================
# GENERAL SENTENCE RELEVANCE
# ============================================================

def sentence_relevance(sentence, question):

    sentence_lower = sentence.lower()

    keywords = question_keywords(question)

    score = 0

    for keyword in keywords:

        if keyword in sentence_lower:

            score += 1

            if len(keyword) >= 7:
                score += 1

    q_type = question_type(question)

    if q_type == "LIMITATION":

        score += limitation_score(
            sentence
        )

        score -= non_limitation_penalty(
            sentence
        )

        if any(
            term in sentence_lower
            for term in [
                "limitation",
                "challenge",
                "difficulty",
                "insufficient",
                "inefficient",
                "constrain",
                "restrict",
                "costly",
                "noisy",
                "brittle",
                "lack of"
            ]
        ):
            score += 3

    elif q_type == "DATASET":

        for term in [
            "dataset",
            "datasets",
            "training",
            "testing",
            "benchmark",
            "corpus"
        ]:

            if term in sentence_lower:
                score += 2

    elif q_type == "TECHNIQUE":

        for term in [
            "method",
            "technique",
            "algorithm",
            "model",
            "approach",
            "architecture",
            "retrieval",
            "reranking",
            "embedding",
            "encoder",
            "generator"
        ]:

            if term in sentence_lower:
                score += 2

    elif q_type == "METRIC":

        for term in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "auroc",
            "metric",
            "evaluation",
            "performance"
        ]:

            if term in sentence_lower:
                score += 2

    return score


# ============================================================
# CLEAN SENTENCE
# ============================================================

def clean_sentence(sentence):

    sentence = str(sentence).strip()

    sentence = re.sub(
        r"\s+",
        " ",
        sentence
    )

    sentence = re.sub(
        r"^\s*\[\d{1,3}\]\s*",
        "",
        sentence
    )

    sentence = re.sub(
        r"(\w)-\s+(\w)",
        r"\1\2",
        sentence
    )

    return sentence.strip()


# ============================================================
# DUPLICATE NORMALIZATION
# ============================================================

def normalize_for_duplicate(text):

    text = str(text).lower()

    text = re.sub(
        r"\[\d{1,3}\]",
        "",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    words = text.split()

    filler = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "that",
        "this",
        "is",
        "are",
        "can",
        "may"
    }

    return set(
        word
        for word in words
        if word not in filler
    )


def sentence_similarity(
    text_a,
    text_b
):

    set_a = normalize_for_duplicate(
        text_a
    )

    set_b = normalize_for_duplicate(
        text_b
    )

    if not set_a or not set_b:
        return 0.0

    intersection = len(
        set_a.intersection(set_b)
    )

    union = len(
        set_a.union(set_b)
    )

    return intersection / union


# ============================================================
# EXTRACT RELEVANT SENTENCES
# ============================================================

def extract_relevant_sentences(
    text,
    question,
    max_sentences=MAX_SENTENCES_PER_EVIDENCE
):

    if not text:
        return []

    if is_reference_chunk(text):
        return []

    sentences = split_sentences(text)

    scored_sentences = []

    for sentence in sentences:

        cleaned = clean_sentence(
            sentence
        )

        if not cleaned:
            continue

        if is_metadata_sentence(cleaned):
            continue

        if is_navigation_sentence(cleaned):
            continue

        score = sentence_relevance(
            cleaned,
            question
        )

        if (
            question_type(question)
            == "LIMITATION"
            and non_limitation_penalty(
                cleaned
            ) >= 7
            and limitation_score(
                cleaned
            ) < 4
        ):
            continue

        if score > 0:

            scored_sentences.append(
                (
                    score,
                    cleaned
                )
            )

    scored_sentences.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []

    for score, sentence in scored_sentences:

        duplicate = False

        for existing in selected:

            if sentence_similarity(
                sentence,
                existing
            ) >= 0.60:

                duplicate = True
                break

        if duplicate:
            continue

        selected.append(
            sentence
        )

        if len(selected) >= max_sentences:
            break

    return selected


# ============================================================
# CONCEPT GROUPS
# ============================================================

def concept_keywords(text):

    text_lower = str(text).lower()

    concepts = []

    concept_groups = {

        "retrieval_reasoning": [
            "single retrieval",
            "singular retrieval",
            "inefficien",
            "multi-step reasoning",
            "insufficient",
            "limited scope"
        ],

        "document_noise": [
            "lengthy content",
            "lengthy document",
            "long document",
            "substantial noise",
            "noisy information",
            "redundant",
            "extracting relevant information"
        ],

        "retriever_architecture": [
            "model architecture",
            "architecture",
            "bert",
            "retriever",
            "retrieval model"
        ],

        "generalization": [
            "generalize",
            "generalization",
            "application domains",
            "different domains"
        ],

        "evaluation": [
            "brittle evaluation",
            "brittle evaluations",
            "benchmark",
            "evaluation protocol",
            "difficult to assess",
            "assess performance"
        ],

        "knowledge_update": [
            "knowledge update",
            "updated knowledge",
            "update its knowledge",
            "knowledge changes"
        ],

        "training_data": [
            "training data",
            "lack of training data",
            "annotated relevance",
            "relevance labels",
            "data augmentation"
        ],

        "search_intent": [
            "search intent",
            "user intent",
            "query intent"
        ]
    }

    for concept, terms in concept_groups.items():

        if any(
            term in text_lower
            for term in terms
        ):
            concepts.append(
                concept
            )

    return set(concepts)


# ============================================================
# BUILD CLEAN EVIDENCE
# ============================================================

def build_clean_evidence(
    evidence,
    question,
    max_items=MAX_EVIDENCE_ITEMS
):

    clean_evidence = []

    if not evidence:
        return clean_evidence

    sentence_candidates = []

    # --------------------------------------------------------
    # Collect candidate sentences
    # --------------------------------------------------------

    for item in evidence:

        text = str(
            item.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        if is_reference_chunk(text):
            continue

        sentences = extract_relevant_sentences(
            text,
            question,
            MAX_SENTENCES_PER_EVIDENCE
        )

        for sentence in sentences:

            sentence_candidates.append(
                {
                    "paper_id": item.get(
                        "paper_id",
                        "Unknown"
                    ),

                    "page_number": item.get(
                        "page_number",
                        "Unknown"
                    ),

                    "chunk_id": item.get(
                        "chunk_id",
                        "Unknown"
                    ),

                    "sentence": sentence,

                    "score": sentence_relevance(
                        sentence,
                        question
                    ),

                    "concepts": concept_keywords(
                        sentence
                    )
                }
            )

    sentence_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    used_text = []
    used_concepts = set()

    # --------------------------------------------------------
    # Select diverse evidence
    # --------------------------------------------------------

    for candidate in sentence_candidates:

        sentence = candidate[
            "sentence"
        ]

        concepts = candidate[
            "concepts"
        ]

        duplicate = False

        for existing in used_text:

            if sentence_similarity(
                sentence,
                existing
            ) >= 0.60:

                duplicate = True
                break

        if duplicate:
            continue

        if (
            concepts
            and concepts.issubset(
                used_concepts
            )
        ):
            continue

        clean_evidence.append(
            {
                "paper_id": candidate[
                    "paper_id"
                ],

                "page_number": candidate[
                    "page_number"
                ],

                "chunk_id": candidate[
                    "chunk_id"
                ],

                "relevance_score": candidate[
                    "score"
                ],

                "text": sentence
            }
        )

        used_text.append(
            sentence
        )

        used_concepts.update(
            concepts
        )

        if len(clean_evidence) >= max_items:
            break

    return clean_evidence


# ============================================================
# DISPLAY
# ============================================================

def print_clean_evidence(
    clean_evidence
):

    if not clean_evidence:

        print(
            "\nNo sufficiently relevant evidence found."
        )

        return

    print(
        "\n" + "=" * 70
    )

    print(
        "CLEAN EVIDENCE"
    )

    print(
        "=" * 70
    )

    for number, item in enumerate(
        clean_evidence,
        start=1
    ):

        print(
            f"\nEvidence {number}"
        )

        print(
            f"Paper ID   : {item['paper_id']}"
        )

        print(
            f"Page       : {item['page_number']}"
        )

        print(
            f"Relevance  : {item['relevance_score']}"
        )

        print(
            f"Text       : {item['text']}"
        )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# TEST
# ============================================================

def test_evidence_answer(
    question,
    evidence
):

    print(
        "\nQuestion:"
    )

    print(
        question
    )

    clean_evidence = build_clean_evidence(
        evidence=evidence,
        question=question
    )

    print_clean_evidence(
        clean_evidence
    )

    return clean_evidence


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Evidence Answer Layer"
    )

    print(
        "Module loaded successfully."
    )

    print(
        "Limitation-aware evidence scoring enabled."
    )

    print(
        "Paper-navigation filtering enabled."
    )