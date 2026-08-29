"""
FAISS-based retriever for the Steam Reviews RAG system.

Builds a vector index from review embeddings and retrieves
the most similar reviews for a given query.
"""

import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


class ReviewRetriever:
    """Retrieves relevant reviews using FAISS similarity search."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.sentences = None

    def build_index(self, df, text_col="review", game_col="game_name",
                    rec_col="recommendation", genre_col="genres"):
        """Build FAISS index from a DataFrame of reviews.

        Each review is stored as a structured sentence with metadata
        to give the LLM richer context during generation.
        """
        # Build structured sentences
        self.sentences = []
        for _, row in df.iterrows():
            parts = [f'Game: "{row[game_col]}"']
            if genre_col in df.columns and str(row.get(genre_col, "")) != "nan":
                parts.append(f"Genres: {row[genre_col]}")
            parts.append(f"Recommendation: {row[rec_col]}")
            parts.append(f"Review: {row[text_col]}")
            self.sentences.append(" | ".join(parts))

        # Encode and build FAISS index
        embeddings = self.encoder.encode(self.sentences, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Built FAISS index: {self.index.ntotal} vectors, {dimension}d")

    def save(self, index_path, sentences_path):
        """Save FAISS index and sentences to disk."""
        faiss.write_index(self.index, str(index_path))
        with open(sentences_path, "wb") as f:
            pickle.dump(self.sentences, f)
        print(f"Saved index to {index_path}")

    def load(self, index_path, sentences_path):
        """Load FAISS index and sentences from disk."""
        self.index = faiss.read_index(str(index_path))
        with open(sentences_path, "rb") as f:
            self.sentences = pickle.load(f)
        print(f"Loaded index: {self.index.ntotal} vectors")

    def search(self, query, top_k=5, min_similarity=0.35):
        """Search for the most relevant reviews given a query.

        Returns list of (sentence, score) tuples.
        """
        query_vec = self.encoder.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_similarity:
                results.append((self.sentences[idx], float(score)))

        return results