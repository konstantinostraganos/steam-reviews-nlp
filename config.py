"""
Central configuration for the Steam Reviews NLP project.

All paths, hyperparameters, and settings live here so that
every notebook and module pulls from a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (API keys, secrets)
load_dotenv()

# ---------------------------------------------------------------------------
# Environment detection: Colab vs local
# ---------------------------------------------------------------------------
ON_COLAB = os.path.exists("/content")

if ON_COLAB:
    ROOT = Path("/content/steam-reviews-nlp")
else:
    ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Raw data files
# ---------------------------------------------------------------------------
REVIEWS_RAW = DATA_RAW / "steam_game_reviews_10_000_stratified.csv"
GAMES_DESC = DATA_RAW / "games_description.csv"

# ---------------------------------------------------------------------------
# Processed data files (created by preprocessing notebook)
# ---------------------------------------------------------------------------
TRAIN_CSV = DATA_PROCESSED / "train.csv"
VAL_CSV = DATA_PROCESSED / "val.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Data split ratios
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the remaining data after test split

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL = "recommendation"
TEXT_COL = "review"
POSITIVE_LABEL = "Recommended"
NEGATIVE_LABEL = "Not Recommended"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MIN_REVIEW_LENGTH = 10  # minimum characters after cleaning

# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# ---------------------------------------------------------------------------
# Word2Vec
# ---------------------------------------------------------------------------
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 3
W2V_EPOCHS = 10

# ---------------------------------------------------------------------------
# Sentence Transformer (used for embeddings, FAISS, and RAG)
# ---------------------------------------------------------------------------
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------
FAISS_INDEX_PATH = MODELS_DIR / "reviews.index"
FAISS_SENTENCES_PATH = MODELS_DIR / "sentences.pkl"

# ---------------------------------------------------------------------------
# DistilBERT fine-tuning
# ---------------------------------------------------------------------------
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_EPOCHS = 3
DISTILBERT_BATCH_SIZE = 16
DISTILBERT_LR = 2e-5
DISTILBERT_SAVE_PATH = MODELS_DIR / "distilbert_sentiment"

# ---------------------------------------------------------------------------
# Groq API (LLM for RAG)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MAX_TOKENS = 512
GROQ_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------
RAG_TOP_K = 5
RAG_MIN_SIMILARITY = 0.35