"""
Text embedding utilities for the Steam Reviews NLP project.

Provides wrappers for:
- TF-IDF vectorization (fit on train only to avoid data leakage)
- Word2Vec training and document vectors
- Sentence Transformer encoding
"""

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


def build_tfidf(train_texts, max_features=5000, ngram_range=(1, 2)):
    """Fit TF-IDF on training data only and return vectorizer + matrix.
    
    IMPORTANT: Only call fit on train set to prevent data leakage.
    Use vectorizer.transform() for val/test sets.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    X_train = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train

def train_word2vec(tokenized_texts, vector_size=100, window=5, min_count=3, epochs=10):
    """Train Word2Vec model on tokenized texts.
    
    Args:
        tokenized_texts: list of lists of tokens (e.g. [["great", "game"], ["boring"]])
    """
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=1,  # deterministic
        seed=42
    )
    return model


def get_document_vector(tokens, w2v_model):
    """Get a single document vector by averaging its word vectors.
    
    Words not in vocabulary are skipped.
    Returns zero vector if no words are found.
    """
    vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(w2v_model.vector_size)


def get_document_vectors(tokenized_texts, w2v_model):
    """Get document vectors for a list of tokenized texts."""
    return np.array([get_document_vector(tokens, w2v_model) for tokens in tokenized_texts])


def encode_sentences(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
    """Encode texts using a Sentence Transformer model.
    
    Returns numpy array of shape (n_texts, embedding_dim).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        show_progress_bar=True,
        batch_size=batch_size
    )
    return model, np.array(embeddings)