import os
import re
import requests
import pandas as pd

OPENALEX_URL = "https://api.openalex.org/works"

OUTPUT_PATH = "dataset/processed/papers_metadata.csv"

# Number of papers requested from OpenAlex before filtering/ranking
SEARCH_RESULTS = 30

# Number of final papers saved
FINAL_RESULTS = 15


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate detection and relevance scoring.
    """
    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def get_title_key(title):
    """
    Creates a normalized title key for duplicate detection.
    """
    return normalize_text(title)


def remove_duplicate_papers(papers):
    """
    Removes duplicate papers based primarily on normalized title.
    """

    unique_papers = []
    seen_titles = set()
    seen_dois = set()

    for paper in papers:

        title = paper.get("title") or ""
        doi = paper.get("doi") or ""

        title_key = get_title_key(title)

        doi_key = normalize_text(doi)

        # Skip duplicate title
        if title_key and title_key in seen_titles:
            continue

        # Skip duplicate DOI
        if doi_key and doi_key in seen_dois:
            continue

        if title_key:
            seen_titles.add(title_key)

        if doi_key:
            seen_dois.add(doi_key)

        unique_papers.append(paper)

    return unique_papers


# ============================================================
# TOPIC RELEVANCE
# ============================================================

def calculate_relevance_score(title, topic):
    """
    Calculates a transparent keyword-based relevance score.

    Higher score = stronger relationship to the research topic.
    """

    title_text = normalize_text(title)
    topic_text = normalize_text(topic)

    score = 0

    # --------------------------------------------------------
    # General topic words
    # --------------------------------------------------------

    topic_terms = [
        "rag",
        "retrieval augmented generation",
        "retrieval augmented",
        "large language model",
        "large language models",
        "llm",
        "academic",
        "research assistant",
        "research",
        "literature",
        "literature review",
        "research paper",
        "research papers"
    ]

    for term in topic_terms:

        term_normalized = normalize_text(term)

        if term_normalized in title_text:
            score += 5

    # --------------------------------------------------------
    # Highly important concepts
    # --------------------------------------------------------

    high_priority_terms = [
        "academic research",
        "research assistant",
        "academic assistant",
        "literature review",
        "related work",
        "research task",
        "research tasks",
        "research paper",
        "academic papers"
    ]

    for term in high_priority_terms:

        if normalize_text(term) in title_text:
            score += 12

    # --------------------------------------------------------
    # RAG-specific terms
    # --------------------------------------------------------

    rag_terms = [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "retrieval augmented",
        "retrieval-augmented"
    ]

    for term in rag_terms:

        if normalize_text(term) in title_text:
            score += 8

    # --------------------------------------------------------
    # LLM-specific terms
    # --------------------------------------------------------

    llm_terms = [
        "large language model",
        "large language models",
        "llm",
        "llms"
    ]

    for term in llm_terms:

        if normalize_text(term) in title_text:
            score += 6

    # --------------------------------------------------------
    # Related research-assistant functionality
    # --------------------------------------------------------

    functionality_terms = [
        "assistant",
        "research",
        "literature",
        "retrieval",
        "question answering",
        "question-answering",
        "knowledge",
        "citation",
        "academic"
    ]

    for term in functionality_terms:

        if normalize_text(term) in title_text:
            score += 3

    # --------------------------------------------------------
    # Topic word overlap
    # --------------------------------------------------------

    topic_words = set(
        word
        for word in topic_text.split()
        if len(word) > 2
    )

    title_words = set(
        word
        for word in title_text.split()
        if len(word) > 2
    )

    overlap = topic_words.intersection(
        title_words
    )

    score += len(overlap) * 2

    return score


# ============================================================
# RELEVANCE CATEGORY
# ============================================================

def relevance_category(score):

    if score >= 30:
        return "Highly Relevant"

    if score >= 18:
        return "Relevant"

    if score >= 10:
        return "Supporting"

    return "Low Relevance"


# ============================================================
# SEARCH OPENALEX
# ============================================================

def search_academic_papers(
    topic,
    max_results=SEARCH_RESULTS
):

    params = {

        "search": topic,

        "per-page": max_results,

        "sort": "relevance_score:desc"
    }

    response = requests.get(
        OPENALEX_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    papers = []

    for work in data.get(
        "results",
        []
    ):

        title = work.get(
            "display_name"
        )

        if not title:
            continue

        publication_year = work.get(
            "publication_year"
        )

        doi = work.get(
            "doi"
        )

        primary_location = (
            work.get(
                "primary_location"
            ) or {}
        )

        landing_page = (
            primary_location.get(
                "landing_page_url"
            )
        )

        pdf_url = (
            primary_location.get(
                "pdf_url"
            )
        )

        open_access_info = (
            work.get(
                "open_access"
            ) or {}
        )

        is_open_access = (
            open_access_info.get(
                "is_oa",
                False
            )
        )

        # ----------------------------------------------------
        # Try best OA location if PDF is unavailable
        # ----------------------------------------------------

        if not pdf_url:

            best_oa_location = (
                work.get(
                    "best_oa_location"
                ) or {}
            )

            pdf_url = (
                best_oa_location.get(
                    "pdf_url"
                )
            )

            if not landing_page:

                landing_page = (
                    best_oa_location.get(
                        "landing_page_url"
                    )
                )

        # ----------------------------------------------------
        # Authors
        # ----------------------------------------------------

        authors = []

        for authorship in work.get(
            "authorships",
            []
        ):

            author = (
                authorship.get(
                    "author"
                ) or {}
            )

            author_name = (
                author.get(
                    "display_name"
                )
            )

            if author_name:

                authors.append(
                    author_name
                )

        # ----------------------------------------------------
        # Paper URL
        # ----------------------------------------------------

        paper_url = (
            landing_page
            or doi
            or work.get("id")
        )

        # ----------------------------------------------------
        # Relevance score
        # ----------------------------------------------------

        relevance_score = (
            calculate_relevance_score(
                title,
                topic
            )
        )

        papers.append({

            "title": title,

            "authors": ", ".join(
                authors
            ),

            "year": publication_year,

            "doi": doi,

            "url": paper_url,

            "open_access": is_open_access,

            "pdf_url": pdf_url,

            "relevance_score":
                relevance_score,

            "relevance_category":
                relevance_category(
                    relevance_score
                )

        })

    return papers


# ============================================================
# SORT AND FILTER
# ============================================================

def rank_papers(
    papers,
    final_results=FINAL_RESULTS
):

    # --------------------------------------------------------
    # Remove duplicates first
    # --------------------------------------------------------

    unique_papers = (
        remove_duplicate_papers(
            papers
        )
    )

    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    unique_papers.sort(
        key=lambda paper:
            (
                paper.get(
                    "relevance_score",
                    0
                ),

                # Prefer newer papers when
                # relevance is similar
                paper.get(
                    "year",
                    0
                ) or 0
            ),

        reverse=True
    )

    # --------------------------------------------------------
    # Keep final number
    # --------------------------------------------------------

    return unique_papers[
        :final_results
    ]


# ============================================================
# SAVE METADATA
# ============================================================

def save_paper_metadata(
    papers
):

    os.makedirs(
        "dataset/processed",
        exist_ok=True
    )

    metadata_records = []

    for number, paper in enumerate(
        papers,
        start=1
    ):

        metadata_records.append({

            "paper_number":
                number,

            "title":
                paper["title"],

            "authors":
                paper["authors"],

            "year":
                paper["year"],

            "doi":
                paper["doi"],

            "paper_url":
                paper["url"],

            "open_access":
                paper["open_access"],

            "pdf_url":
                paper["pdf_url"],

            "relevance_score":
                paper[
                    "relevance_score"
                ],

            "relevance_category":
                paper[
                    "relevance_category"
                ]

        })

    df = pd.DataFrame(
        metadata_records
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print(
        "\nPaper metadata saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "Total papers saved:",
        len(df)
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    papers
):

    print(
        "\n======================================"
    )

    print(
        "ACADEMIC PAPER DISCOVERY"
    )

    print(
        "======================================"
    )

    for number, paper in enumerate(
        papers,
        start=1
    ):

        print(
            f"\n{number}. "
            f"{paper['title']}"
        )

        print(
            f"   Relevance Score: "
            f"{paper['relevance_score']}"
        )

        print(
            f"   Relevance: "
            f"{paper['relevance_category']}"
        )

        print(
            f"   Authors: "
            f"{paper['authors']}"
        )

        print(
            f"   Year: "
            f"{paper['year']}"
        )

        print(
            f"   DOI: "
            f"{paper['doi']}"
        )

        print(
            f"   Open Access: "
            f"{paper['open_access']}"
        )

        print(
            f"   PDF URL: "
            f"{paper['pdf_url']}"
        )

        print(
            f"   Paper Link: "
            f"{paper['url']}"
        )

        print(
            "-" * 70
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter research topic: "
    ).strip()

    if not topic:

        print(
            "\nResearch topic cannot be empty."
        )

        raise SystemExit

    print(
        "\nSearching academic literature..."
    )

    # --------------------------------------------------------
    # Search more papers initially
    # --------------------------------------------------------

    papers = search_academic_papers(
        topic,
        max_results=SEARCH_RESULTS
    )

    print(
        f"\nOpenAlex returned "
        f"{len(papers)} papers."
    )

    # --------------------------------------------------------
    # Remove duplicates + rank
    # --------------------------------------------------------

    ranked_papers = rank_papers(
        papers,
        final_results=FINAL_RESULTS
    )

    print(
        f"After duplicate removal: "
        f"{len(ranked_papers)} papers retained."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_paper_metadata(
        ranked_papers
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        ranked_papers
    )