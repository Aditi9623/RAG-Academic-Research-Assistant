from evidence_answer import test_evidence_answer


question = "What are the main challenges and limitations of Retrieval-Augmented Generation?"


evidence = [
    {
        "paper_id": "P001",
        "page_number": 10,
        "chunk_id": "test_001",
        "text": """
        A single retrieval step may be insufficient for complex multi-step
        reasoning tasks. Iterative retrieval can improve the ability to obtain
        additional information, but repeated retrieval may increase the
        computational cost and introduce irrelevant information.
        """
    },
    {
        "paper_id": "P005",
        "page_number": 8,
        "chunk_id": "test_002",
        "text": """
        RAG systems face challenges in understanding user search intent.
        Long and noisy documents can make retrieval difficult. Obtaining
        high-quality human-annotated relevance labels can also be costly.
        Generalization across different domains remains a challenge.
        """
    },
    {
        "paper_id": "P008",
        "page_number": 27,
        "chunk_id": "test_003",
        "text": """
        Knowledge updates can create challenges for systems that depend on
        retrieved information. Evaluation can also be brittle because
        benchmark capabilities may vary across different tasks.
        """
    }
]


test_evidence_answer(
    question,
    evidence
)