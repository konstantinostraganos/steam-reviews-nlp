# 🎮 Steam Reviews NLP — Sentiment Analysis & RAG Chatbot

[![CI](https://github.com/konstantinostraganos/steam-reviews-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/konstantinostraganos/steam-reviews-nlp/actions/workflows/ci.yml)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steam-reviews-nlp-3ygfbqh2dg2b8354vpgn9d.streamlit.app/)

An end-to-end NLP project that analyzes Steam game reviews for sentiment classification
and provides an interactive RAG-powered chatbot for game recommendations.

**[▶ Try the Live Demo](https://steam-reviews-nlp-3ygfbqh2dg2b8354vpgn9d.streamlit.app/)**

## Project Overview

This project demonstrates the full lifecycle of an NLP/ML system:

1. **Data Preprocessing** — Cleaning and preparing 10,000 Steam game reviews
2. **Text Embeddings** — TF-IDF, Word2Vec, and Sentence Transformer representations
3. **Topic Modeling** — LDA and BERTopic for discovering hidden themes
4. **Sentiment Classification** — Logistic Regression, XGBoost, and fine-tuned DistilBERT
5. **Explainability** — SHAP analysis for model interpretability
6. **RAG System** — FAISS-based retrieval with Groq LLM for answer generation
7. **Interactive App** — Streamlit chatbot interface

## Results

### Sentiment Classification

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 81.0% | 0.874 |
| XGBoost | 84.9% | 0.907 |
| **DistilBERT (fine-tuned)** | **89.3%** | **0.933** |

### RAG Evaluation

| Metric | Score |
|--------|-------|
| Retrieval Relevance | 3.00 / 3.00 |
| Answer Quality | 4.88 / 5.00 |
| Faithfulness | 1.00 / 1.00 |

## Project Structure

    steam-reviews-nlp/
    ├── app/
    │   └── streamlit_app.py          # Streamlit chatbot UI
    ├── data/
    │   ├── raw/                      # Original datasets
    │   └── processed/                # Train/val/test splits
    ├── models/                       # Saved models (not in Git)
    ├── notebooks/
    │   ├── 01_eda.ipynb              # Exploratory Data Analysis
    │   ├── 02_preprocessing.ipynb    # Preprocessing & data splitting
    │   ├── 03_embeddings.ipynb       # TF-IDF, Word2Vec, Sentence Transformers
    │   ├── 04_topic_modeling.ipynb   # LDA & BERTopic
    │   ├── 05_classification.ipynb   # Model training & comparison
    │   ├── 06_xai.ipynb              # SHAP explainability
    │   └── 07_rag_evaluation.ipynb   # RAG system evaluation
    ├── src/
    │   ├── preprocessing.py          # Text cleaning pipeline
    │   ├── embeddings.py             # Embedding utilities
    │   ├── training.py               # Model training functions
    │   ├── evaluation.py             # Metrics & comparison
    │   └── rag/
    │       ├── retriever.py          # FAISS vector retrieval
    │       └── generator.py          # Groq LLM generation
    ├── config.py                     # Central configuration
    ├── requirements.txt              # Dependencies
    └── .env.example                  # API key template


## Tech Stack

- **ML/NLP:** scikit-learn, XGBoost, Hugging Face Transformers, NLTK, Gensim
- **Deep Learning:** PyTorch, DistilBERT
- **Embeddings:** TF-IDF, Word2Vec, Sentence Transformers (MiniLM-L6-v2)
- **Vector Store:** FAISS
- **LLM:** Groq API (gpt-oss-20b)
- **Explainability:** SHAP
- **Topic Modeling:** LDA, BERTopic
- **App:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Plotly

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/konstantinostraganos/steam-reviews-nlp.git
cd steam-reviews-nlp
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements-dev.txt
```

### 3. Set up API key
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 4. Run notebooks
Run notebooks in order (01 → 07) to reproduce the full pipeline.
DistilBERT training (notebook 05) requires GPU — use Google Colab.

### 5. Launch the chatbot
```bash
streamlit run app/streamlit_app.py
```

## Key Design Decisions

- **Two preprocessing modes:** Light (for DistilBERT) and heavy (for TF-IDF/traditional ML) to match each model's needs
- **No data leakage:** TF-IDF fitted on training set only, consistent train/val/test splits across all models
- **FAISS over brute-force:** Vector index for scalable similarity search in the RAG system
- **Direct API calls:** No LangChain — transparent RAG implementation for full understanding and control
- **Conversation history:** Multi-turn chat support in the RAG chatbot

## Author

**Konstantinos Traganos**
MSc in Data Science — Deree, The American College of Greece
