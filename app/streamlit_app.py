"""
Steam Reviews RAG Chatbot — Streamlit Application

Interactive chatbot that answers questions about Steam games
based on real user reviews using RAG (Retrieval-Augmented Generation).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from config import (
    REVIEWS_RAW, GAMES_DESC, GROQ_API_KEY,
    GROQ_MODEL, GROQ_API_URL, GROQ_MAX_TOKENS,
    FAISS_INDEX_PATH, FAISS_SENTENCES_PATH,
    SENTENCE_MODEL_NAME, RAG_TOP_K, RAG_MIN_SIMILARITY
)
from src.rag.retriever import ReviewRetriever
from src.rag.generator import ReviewGenerator


# --- Page config ---
st.set_page_config(
    page_title="Steam Reviews Chatbot",
    page_icon="🎮",
    layout="wide"
)


# --- Initialize components (cached so they load once) ---
@st.cache_resource
def load_retriever():
    """Load or build FAISS index."""
    retriever = ReviewRetriever(model_name=SENTENCE_MODEL_NAME)

    if FAISS_INDEX_PATH.exists() and FAISS_SENTENCES_PATH.exists():
        retriever.load(FAISS_INDEX_PATH, FAISS_SENTENCES_PATH)
    else:
        df = pd.read_csv(REVIEWS_RAW)
        df = df.dropna(subset=["review"])

        if GAMES_DESC.exists():
            games = pd.read_csv(GAMES_DESC)
            df = df.merge(games, left_on="game_name", right_on="name", how="left")

        retriever.build_index(df)
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        retriever.save(FAISS_INDEX_PATH, FAISS_SENTENCES_PATH)

    return retriever


def get_generator(temperature):
    """Create generator with current settings."""
    return ReviewGenerator(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        api_url=GROQ_API_URL,
        max_tokens=GROQ_MAX_TOKENS,
        temperature=temperature
    )


def process_query(prompt, retriever, temperature, top_k):
    """Retrieve reviews and generate answer for a query."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve
    retrieved = retriever.search(prompt, top_k=top_k, min_similarity=RAG_MIN_SIMILARITY)

    # Generate
    generator = get_generator(temperature)
    generator.conversation_history = st.session_state.history
    answer = generator.generate(prompt, retrieved)
    st.session_state.history = generator.get_history()

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "reviews": retrieved
    })


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.1, max_value=1.0, value=0.7, step=0.1,
        help="Low = safe and precise, High = creative but less reliable"
    )

    top_k = st.slider(
        "Number of reviews to retrieve",
        min_value=1, max_value=10, value=RAG_TOP_K,
        help="How many reviews to use as context for the answer"
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.divider()

    # Conversation history in sidebar
    if "messages" in st.session_state and st.session_state.messages:
        st.header("💬 Chat History")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content'][:50]}...")
        st.divider()

    st.header("📊 Dataset Info")
    st.markdown("""
    - **Source:** Steam Game Reviews
    - **Reviews:** ~10,000
    - **Games:** 240
    - **Model:** Groq (gpt-oss-20b)
    - **Retrieval:** FAISS + MiniLM-L6-v2
    """)

    st.divider()

    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses **RAG** (Retrieval-Augmented Generation)
    to answer questions about Steam games based on real user reviews.

    Built as part of an NLP portfolio project.
    """)


# --- Main UI ---
st.title("🎮 Steam Reviews Chatbot")
st.caption("Ask questions about Steam games — answers powered by real user reviews")

# Load retriever
retriever = load_retriever()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# Suggested questions (only when chat is empty)
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    suggestions = [
        "What are the best horror games?",
        "Is Stardew Valley worth buying?",
        "What do players complain about in Fallout 76?",
        "Which games are good for playing with friends?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, key=f"sug_{i}"):
            process_query(suggestion, retriever, temperature, top_k)
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "reviews" in message:
            with st.expander("📄 Retrieved Reviews"):
                for review, score in message["reviews"]:
                    st.markdown(f"**Relevance: {score:.2f}**")
                    st.markdown(f"_{review}_")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about any Steam game..."):
    process_query(prompt, retriever, temperature, top_k)
    st.rerun()