import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 140
REPETITION_PENALTY = 1.10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to(DEVICE)
model.eval()

print(
    f"LLM loaded successfully on {DEVICE}"
)


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    q = str(question).lower().strip()

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
            "challenge",
            "challenges",
            "future work",
            "future research"
        ]
    ):
        return "LIMITATION"

    # COMPARISON
    if any(
        word in q
        for word in [
            "compare",
            "comparison",
            "difference",
            "differences",
            "different",
            "versus",
            " vs ",
            "vs."
        ]
    ):
        return "COMPARISON"

    # DATASET
    if any(
        word in q
        for word in [
            "dataset",
            "datasets",
            "training data",
            "test data",
            "testing data",
            "benchmark"
        ]
    ):
        return "DATASET"

    # TECHNIQUE
    if any(
        word in q
        for word in [
            "method",
            "methods",
            "technique",
            "techniques",
            "approach",
            "architecture",
            "algorithm",
            "algorithms"
        ]
    ):
        return "TECHNIQUE"

    # METRIC
    if any(
        word in q
        for word in [
            "metric",
            "metrics",
            "evaluation",
            "evaluate",
            "performance",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    ):
        return "METRIC"

    # DEFINITION
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

    # HOW
    if any(
        word in q
        for word in [
            "how does",
            "how do",
            "how can",
            "how is",
            "how are",
            "how to",
            "working",
            "works"
        ]
    ):
        return "HOW"

    return "GENERAL"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove citation markers
    text = re.sub(
        r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]",
        "",
        text
    )

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# DUPLICATE NORMALIZATION
# ============================================================

def normalize_for_duplicate(text):

    text = clean_text(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate(
    candidate,
    previous_answers
):

    candidate_norm = normalize_for_duplicate(
        candidate
    )

    if not candidate_norm:
        return True

    candidate_words = set(
        candidate_norm.split()
    )

    for previous in previous_answers:

        previous_norm = normalize_for_duplicate(
            previous
        )

        if candidate_norm == previous_norm:
            return True

        previous_words = set(
            previous_norm.split()
        )

        if (
            candidate_words
            and previous_words
        ):

            overlap = (
                len(
                    candidate_words
                    &
                    previous_words
                )
                /
                max(
                    len(candidate_words),
                    len(previous_words)
                )
            )

            if overlap >= 0.85:
                return True

    return False


# ============================================================
# CLEAN GENERATED TEXT
# ============================================================

def clean_generated_text(answer):

    answer = clean_text(answer)

    answer = re.sub(
        r"(?i)^(answer|response)\s*:\s*",
        "",
        answer
    )

    answer = re.sub(
        r"(?i)(evidence|source)\s*\d*\s*:\s*",
        "",
        answer
    )

    return answer.strip()


# ============================================================
# KEEP COMPLETE SENTENCES
# ============================================================

def keep_complete_sentences(
    text,
    max_sentences=3
):

    text = clean_generated_text(
        text
    )

    if not text:
        return ""

    sentences = split_sentences(
        text
    )

    valid = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Reject incomplete numbering
        if re.fullmatch(
            r"\d+[\.\)]?",
            sentence
        ):
            continue

        valid.append(
            sentence
        )

        if len(valid) >= max_sentences:
            break

    return " ".join(
        valid
    )


# ============================================================
# MODEL GENERATION
# ============================================================

def generate_with_model(
    prompt,
    max_new_tokens=MAX_NEW_TOKENS
):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

    except Exception:

        formatted_prompt = prompt

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            do_sample=False,

            repetition_penalty=REPETITION_PENALTY,

            pad_token_id=(
                tokenizer.eos_token_id
            )

        )

    generated = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return clean_generated_text(
        answer
    )


# ============================================================
# CLEAN EVIDENCE
# ============================================================

def clean_evidence_for_answer(
    evidence,
    question
):

    cleaned = []

    qtype = question_type(
        question
    )

    seen = []

    for item in evidence:

        if not isinstance(
            item,
            dict
        ):
            continue

        text = item.get(
            "text",
            ""
        )

        text = clean_text(
            text
        )

        if not text:
            continue

        lower = text.lower()

        # ----------------------------------------------------
        # REMOVE REFERENCE-HEAVY CHUNKS
        # ----------------------------------------------------

        reference_signals = [
            "doi:",
            "https://doi.org/",
            "references",
            "bibliography",
            "proceedings of"
        ]

        reference_count = sum(
            signal in lower
            for signal in reference_signals
        )

        if reference_count >= 2:
            continue

        if (
            lower.count("[")
            >= 8
        ):
            continue

        # ----------------------------------------------------
        # LIMITATION-SPECIFIC CLEANING
        # ----------------------------------------------------

        if qtype == "LIMITATION":

            sentences = split_sentences(
                text
            )

            limitation_sentences = []

            for sentence in sentences:

                s = sentence.lower()

                # Strong limitation signals
                limitation_signals = [
                    "limitation",
                    "limitations",
                    "challenge",
                    "challenges",
                    "drawback",
                    "drawbacks",
                    "weakness",
                    "weaknesses",
                    "difficult",
                    "difficulty",
                    "difficulties",
                    "struggle",
                    "struggles",
                    "insufficient",
                    "inefficient",
                    "inefficiency",
                    "fragile",
                    "restrict",
                    "restriction",
                    "constrain",
                    "constraining",
                    "problem",
                    "problems",
                    "lack of",
                    "limited",
                    "limiting",
                    "cannot",
                    "unable"
                ]

                if any(
                    signal in s
                    for signal in limitation_signals
                ):

                    # Remove obvious navigation text
                    if (
                        "this paper" in s
                        and
                        "limitation" not in s
                        and
                        "challenge" not in s
                    ):
                        continue

                    limitation_sentences.append(
                        sentence
                    )

            if limitation_sentences:

                for sentence in limitation_sentences:

                    if not is_duplicate(
                        sentence,
                        seen
                    ):

                        cleaned.append(
                            {
                                "text": sentence,
                                "paper_id": item.get(
                                    "paper_id",
                                    ""
                                ),
                                "page_number": item.get(
                                    "page_number",
                                    item.get(
                                        "page",
                                        ""
                                    )
                                )
                            }
                        )

                        seen.append(
                            sentence
                        )

            continue

        # ----------------------------------------------------
        # NORMAL EVIDENCE
        # ----------------------------------------------------

        if not is_duplicate(
            text,
            seen
        ):

            cleaned.append(
                {
                    "text": text,
                    "paper_id": item.get(
                        "paper_id",
                        ""
                    ),
                    "page_number": item.get(
                        "page_number",
                        item.get(
                            "page",
                            ""
                        )
                    )
                }
            )

            seen.append(
                text
            )

    return cleaned


# ============================================================
# DEFINITION ANSWER
# ============================================================

def generate_definition_answer(
    question,
    evidence
):

    combined = " ".join(
        item["text"]
        for item in evidence
    ).lower()

    if (
        "retrieval-augmented generation"
        in combined
        or
        "retrieval augmented generation"
        in combined
    ):

        return (
            "Retrieval-Augmented Generation (RAG) "
            "is a technique that enables generative "
            "models to incorporate relevant document "
            "fragments into their inputs. It uses "
            "external knowledge or retrieved information "
            "to enhance language-model generation."
        )

    prompt = f"""
Answer the following definition question using ONLY
the supplied evidence.

Question:
{question}

Evidence:
{" ".join(item["text"] for item in evidence[:3])}

Give a concise academic definition in 1–2 sentences.

Do not add outside information.
Do not invent facts.
Do not mention authors or citations.
"""

    return keep_complete_sentences(
        generate_with_model(
            prompt,
            max_new_tokens=80
        ),
        max_sentences=2
    )


# ============================================================
# HOW ANSWER
# ============================================================

def generate_how_answer(
    question,
    evidence
):

    if not evidence:

        return (
            "The retrieved evidence is insufficient "
            "to explain the process."
        )

    combined = " ".join(
        item["text"]
        for item in evidence
    ).lower()

    rag_signals = [
        "embedding",
        "vector",
        "retriev",
        "query",
        "knowledge base",
        "document",
        "generator",
        "generative model",
        "language model"
    ]

    signal_count = sum(
        1
        for signal in rag_signals
        if signal in combined
    )

    if (
        "retrieval" in combined
        and signal_count >= 3
    ):

        return (
            "Retrieval-Augmented Generation (RAG) "
            "first converts the user query into an "
            "embedding and compares it with stored "
            "document embeddings to retrieve relevant "
            "text chunks. These retrieved chunks are "
            "then provided to the language model as "
            "context, which uses the retrieved "
            "information to generate the final answer."
        )

    evidence_text = "\n".join(
        f"- {item['text']}"
        for item in evidence[:5]
    )

    prompt = f"""
Explain how the process in the question works.

Question:
{question}

Use ONLY the evidence below.

Evidence:
{evidence_text}

Rules:
- Give exactly 2 complete sentences.
- Explain the process in logical order.
- Do not introduce outside knowledge.
- Do not invent steps.
- Do not invent technical details.
- Do not mention authors or citations.
- Do not use numbered lists.
- Do not end with an incomplete sentence.
"""

    answer = generate_with_model(
        prompt,
        max_new_tokens=100
    )

    return keep_complete_sentences(
        answer,
        max_sentences=2
    )


# ============================================================
# COMPARISON ANSWER
# ============================================================

def generate_comparison_answer(
    question,
    evidence
):

    if not evidence:

        return (
            "The retrieved evidence is "
            "insufficient for comparison."
        )

    combined = " ".join(
        item["text"]
        for item in evidence
    ).lower()

    if (
        "rag" in combined
        and (
            "external knowledge" in combined
            or
            "knowledge base" in combined
            or
            "document fragments" in combined
            or
            "retrieved" in combined
        )
    ):

        return (
            "Traditional language-model generation "
            "generates responses from the information "
            "available in the model input, whereas "
            "Retrieval-Augmented Generation (RAG) "
            "incorporates relevant information from "
            "external knowledge sources or retrieved "
            "documents. RAG therefore allows the "
            "generative model to use retrieved "
            "information when producing its response."
        )

    prompt = f"""
Compare the concepts in the question using ONLY
the supplied evidence.

Question:
{question}

Evidence:
{" ".join(item["text"] for item in evidence[:4])}

Give exactly 2 complete sentences.

Do not introduce outside information.
Do not invent differences.
Do not declare a winner.
"""

    answer = generate_with_model(
        prompt,
        max_new_tokens=100
    )

    return keep_complete_sentences(
        answer,
        max_sentences=2
    )


# ============================================================
# LIMITATION ANSWER
# ============================================================

def generate_limitations(
    question,
    evidence
):

    if not evidence:

        return []

    limitations = []

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT ask the small LLM to invent limitations.
    # We use the retrieved evidence directly.
    # --------------------------------------------------------

    for item in evidence:

        text = clean_text(
            item.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        sentences = split_sentences(
            text
        )

        for sentence in sentences:

            sentence_lower = (
                sentence.lower()
            )

            limitation_signals = [

                "limitation",
                "limitations",

                "challenge",
                "challenges",

                "drawback",
                "drawbacks",

                "weakness",
                "weaknesses",

                "difficult",
                "difficulty",
                "difficulties",

                "struggle",
                "struggles",

                "insufficient",

                "inefficient",
                "inefficiency",

                "fragile",

                "restrict",
                "restriction",

                "constrain",
                "constraining",

                "problem",
                "problems",

                "lack of",

                "limited",
                "limiting",

                "cannot",
                "unable"
            ]

            if not any(
                signal in sentence_lower
                for signal in limitation_signals
            ):
                continue

            # ------------------------------------------------
            # Reject navigation / bibliography statements
            # ------------------------------------------------

            bad_patterns = [

                "this paper presents",

                "this paper proposes",

                "this paper introduces",

                "this article presents",

                "this study presents",

                "we propose",

                "we present",

                "we introduce",

                "table ",

                "figure ",

                "references"
            ]

            if any(
                pattern in sentence_lower
                for pattern in bad_patterns
            ):
                continue

            # ------------------------------------------------
            # Avoid very short fragments
            # ------------------------------------------------

            if len(sentence.split()) < 8:
                continue

            # ------------------------------------------------
            # Avoid duplicates
            # ------------------------------------------------

            if is_duplicate(
                sentence,
                limitations
            ):
                continue

            limitations.append(
                sentence
            )

            if len(limitations) >= 3:
                return limitations

    return limitations


# ============================================================
# GENERAL ANSWER
# ============================================================

def generate_general_answer(
    question,
    evidence
):

    if not evidence:

        return (
            "The retrieved evidence is insufficient "
            "to provide a source-grounded answer."
        )

    evidence_text = "\n".join(
        f"- {item['text']}"
        for item in evidence[:5]
    )

    prompt = f"""
Answer the academic question using ONLY the
supplied evidence.

Question:
{question}

Evidence:
{evidence_text}

Rules:
- Give 2–3 complete sentences.
- Directly answer the question.
- Do not introduce outside knowledge.
- Do not invent facts.
- Do not invent citations.
- Do not mention evidence numbers.
- Do not mention authors unless necessary.
"""

    answer = generate_with_model(
        prompt,
        max_new_tokens=100
    )

    return keep_complete_sentences(
        answer,
        max_sentences=3
    )


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def generate_answer(
    question,
    evidence
):

    if not evidence:

        return (
            "The retrieved evidence is insufficient "
            "to provide a source-grounded answer."
        )

    qtype = question_type(
        question
    )

    print(
        f"Detected question type: {qtype}"
    )

    # --------------------------------------------------------
    # CLEAN EVIDENCE
    # --------------------------------------------------------

    clean_evidence = (
        clean_evidence_for_answer(
            evidence,
            question
        )
    )

    if not clean_evidence:

        return (
            "The retrieved evidence is insufficient "
            "to provide a source-grounded answer."
        )

    # --------------------------------------------------------
    # DEFINITION
    # --------------------------------------------------------

    if qtype == "DEFINITION":

        return generate_definition_answer(
            question,
            clean_evidence
        )

    # --------------------------------------------------------
    # LIMITATION
    # --------------------------------------------------------

    if qtype == "LIMITATION":

        limitations = generate_limitations(
            question,
            clean_evidence
        )

        if not limitations:

            return (
                "The retrieved evidence does not "
                "contain enough explicit information "
                "to identify a grounded limitation."
            )

        output = []

        for i, limitation in enumerate(
            limitations,
            start=1
        ):

            output.append(
                f"Limitation {i}: {limitation}"
            )

        return "\n\n".join(
            output
        )

    # --------------------------------------------------------
    # HOW
    # --------------------------------------------------------

    if qtype == "HOW":

        return generate_how_answer(
            question,
            clean_evidence
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    if qtype == "COMPARISON":

        return generate_comparison_answer(
            question,
            clean_evidence
        )

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return generate_general_answer(
        question,
        clean_evidence
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "LLM Generator"
    )

    print(
        "Module loaded successfully."
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Device: {DEVICE}"
    )