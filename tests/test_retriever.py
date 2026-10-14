"""Tests for the FAISS retriever.

The real sentence-transformer is replaced by a tiny keyword encoder, so the
tests run in milliseconds, need no model download, and have predictable scores.
"""

import numpy as np
import pandas as pd
import pytest

import src.rag.retriever as retriever_module
from src.rag.retriever import ReviewRetriever

VOCABULARY = ["horror", "scary", "farming", "relaxing", "crash", "bugs", "shooter"]


class KeywordEncoder:
    """Stand-in for SentenceTransformer: one dimension per vocabulary word."""

    def __init__(self, model_name=None):
        self.model_name = model_name

    def encode(self, texts, show_progress_bar=False):
        vectors = np.zeros((len(texts), len(VOCABULARY)), dtype="float32")
        for i, text in enumerate(texts):
            words = text.lower().split()
            for j, term in enumerate(VOCABULARY):
                vectors[i, j] = sum(word.strip('".,|:') == term for word in words)
        return vectors


@pytest.fixture
def reviews():
    return pd.DataFrame(
        {
            "game_name": ["Dread Manor", "Green Acres", "Glitch Racer"],
            "genres": ["Horror", "Simulation", np.nan],
            "recommendation": ["Recommended", "Recommended", "Not Recommended"],
            "review": [
                "Truly scary horror atmosphere",
                "Relaxing farming after work",
                "Constant crash and bugs everywhere",
            ],
        }
    )


@pytest.fixture
def retriever(monkeypatch, reviews):
    monkeypatch.setattr(retriever_module, "SentenceTransformer", KeywordEncoder)
    instance = ReviewRetriever()
    instance.build_index(reviews)
    return instance


def test_index_contains_one_vector_per_review(retriever, reviews):
    assert retriever.index.ntotal == len(reviews)


def test_sentences_carry_metadata_for_the_llm(retriever):
    first = retriever.sentences[0]
    assert 'Game: "Dread Manor"' in first
    assert "Genres: Horror" in first
    assert "Recommendation: Recommended" in first
    assert "Review: Truly scary horror atmosphere" in first


def test_missing_genre_is_omitted_not_written_as_nan(retriever):
    third = retriever.sentences[2]
    assert "Genres" not in third
    assert "nan" not in third


def test_most_relevant_review_ranks_first(retriever):
    results = retriever.search("scary horror games", top_k=3, min_similarity=0.0)
    assert "Dread Manor" in results[0][0]


def test_scores_are_sorted_descending(retriever):
    results = retriever.search("relaxing farming with no bugs", top_k=3, min_similarity=0.0)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_number_of_results(retriever):
    assert len(retriever.search("horror farming crash", top_k=2, min_similarity=0.0)) == 2


def test_min_similarity_filters_irrelevant_reviews(retriever):
    # No review mentions shooters, so every similarity is zero
    assert retriever.search("shooter", top_k=3, min_similarity=0.35) == []


def test_save_and_load_round_trip(retriever, tmp_path, monkeypatch):
    index_path = tmp_path / "reviews.index"
    sentences_path = tmp_path / "sentences.pkl"
    retriever.save(index_path, sentences_path)

    monkeypatch.setattr(retriever_module, "SentenceTransformer", KeywordEncoder)
    reloaded = ReviewRetriever()
    reloaded.load(index_path, sentences_path)

    query = "scary horror"
    assert reloaded.search(query, min_similarity=0.0) == retriever.search(
        query, min_similarity=0.0
    )
