import streamlit as st
import sys
import os
import time
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

# --------------------------------------------------
# Add src folder to Python path
# --------------------------------------------------

SRC_PATH = os.path.join(os.path.dirname(__file__), "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from evidence_selector import select_evidence
from llm_generator import generate_answer


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .evidence-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

INDEX_PATH = "dataset/processed/faiss_index.index"
CHUNKS_PATH = "dataset/processed/chunks.csv"


# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    index = faiss.read_index(INDEX_PATH)

    df = pd.read_csv(CHUNKS_PATH)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        tokenizer_kwargs={"use_fast": True}
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return index, df, embedding_model, reranker


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 RAG-Based Academic Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Evidence-grounded research assistance using Retrieval-Augmented Generation'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ System Information")

    st.write("**Embedding Model**")
    st.write("all-MiniLM-L6-v2")

    st.write("**Vector Database**")
    st.write("FAISS")

    st.write("**Reranker**")
    st.write("MS MARCO MiniLM")

    st.write("**LLM**")
    st.write("Qwen2.5-0.5B-Instruct")

    st.divider()

    st.write("**Current Dataset**")

    try:
        dataset_preview = pd.read_csv(CHUNKS_PATH)

        st.metric(
            "Document Chunks",
            len(dataset_preview)
        )

        st.metric(
            "Papers",
            dataset_preview["paper_id"].nunique()
        )

    except Exception:
        st.warning("Dataset could not be loaded.")


# --------------------------------------------------
# Load System
# --------------------------------------------------

with st.spinner("Loading research assistant..."):

    try:
        index, df, embedding_model, reranker = load_models()

    except Exception as e:

        st.error("System could not be loaded.")

        st.exception(e)

        st.stop()


# --------------------------------------------------
# Main Question Input
# --------------------------------------------------

st.subheader("🔎 Ask a Research Question")

query = st.text_area(
    "Enter your academic research question:",
    placeholder="Example: What is Retrieval-Augmented Generation?",
    height=100
)


# --------------------------------------------------
# Generate Answer Button
# --------------------------------------------------

generate_button = st.button(
    "🚀 Generate Research Answer",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# RAG Pipeline
# --------------------------------------------------

if generate_button:

    if not query.strip():

        st.warning("Please enter a research question.")

        st.stop()

    start_time = time.time()

    # ----------------------------------------------
    # Step 1: Query Embedding
    # ----------------------------------------------

    with st.spinner("🔍 Retrieving relevant research evidence..."):

        query_embedding = embedding_model.encode([query])

        distances, indices = index.search(
            query_embedding,
            10
        )

    # ----------------------------------------------
    # Step 2: Prepare Documents
    # ----------------------------------------------

    documents = []

    for index_id in indices[0]:

        row = df.iloc[index_id]

        documents.append(
            row["text"]
        )

    # ----------------------------------------------
    # Step 3: Reranking
    # ----------------------------------------------

    with st.spinner("📊 Reranking retrieved evidence..."):

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = reranker.predict(pairs)

        ranked_results = sorted(
            zip(scores, indices[0]),
            key=lambda x: x[0],
            reverse=True
        )

    # ----------------------------------------------
    # Step 4: Evidence Selection
    # ----------------------------------------------

    evidence = select_evidence(
        ranked_results,
        top_k=3
    )

    # ----------------------------------------------
    # Step 5: LLM Generation
    # ----------------------------------------------

    with st.spinner("🤖 Generating grounded answer..."):

        answer = generate_answer(
            query,
            evidence
        )

    elapsed_time = time.time() - start_time

    # --------------------------------------------------
    # Answer
    # --------------------------------------------------

    st.divider()

    st.subheader("💡 Grounded Research Answer")

    st.write(answer)

    # --------------------------------------------------
    # Evidence & Citations
    # --------------------------------------------------

    st.subheader("📑 Evidence & Citations")

    if evidence:

        for rank, item in enumerate(evidence, start=1):

            with st.expander(
                f"Evidence {rank} — "
                f"{item['paper_id']} | Page {item['page_number']}"
            ):

                st.write(
                    f"**Paper ID:** {item['paper_id']}"
                )

                st.write(
                    f"**Page:** {item['page_number']}"
                )

                st.write(
                    f"**Chunk ID:** {item['chunk_id']}"
                )

                st.write(
                    f"**Reranking Score:** "
                    f"{item['reranking_score']:.4f}"
                )

                st.markdown("**Retrieved Evidence:**")

                st.write(item["text"])

    else:

        st.warning(
            "No suitable evidence was found for this question."
        )

    # --------------------------------------------------
    # Retrieval Statistics
    # --------------------------------------------------

    st.subheader("📊 Retrieval Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Initial Retrieved",
            len(indices[0])
        )

    with col2:
        st.metric(
            "Evidence Selected",
            len(evidence)
        )

    with col3:
        st.metric(
            "Response Time",
            f"{elapsed_time:.2f} sec"
        )

    # --------------------------------------------------
    # Research Pipeline
    # --------------------------------------------------

    st.subheader("🔄 RAG Pipeline")

    st.info(
        "Research Question → "
        "Semantic Retrieval → "
        "Reranking → "
        "Evidence Selection → "
        "LLM Generation → "
        "Grounded Answer"
    )