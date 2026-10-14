"""Tests for the text preprocessing pipeline."""

import pandas as pd
import pytest

from src.preprocessing import clean_heavy, clean_light, preprocess_dataframe


class TestCleanLight:
    """Light cleaning feeds the transformer model, so it must keep the text readable."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2021 Great game!", "great game"),
            ("Early Access Review Fun so far", "fun so far"),
            ("Check https://example.com now", "check now"),
            ("This is ♥♥♥♥ good", "this is good"),
            ("Too    many   spaces", "too many spaces"),
        ],
    )
    def test_removes_steam_specific_noise(self, raw, expected):
        assert clean_light(raw) == expected

    def test_expands_contractions(self):
        assert clean_light("I don't like it") == "i do not like it"

    def test_keeps_stopwords(self):
        # Transformers need function words for context, so they must survive
        assert clean_light("This is not the best") == "this is not the best"

    @pytest.mark.parametrize("raw", [None, "", "   ", 42])
    def test_invalid_input_returns_empty_string(self, raw):
        assert clean_light(raw) == ""

    @pytest.mark.xfail(
        strict=True,
        reason="Known bug: curly apostrophes (’) are not normalised, so contractions "
        "typed on phones are not expanded (affects ~1.6% of reviews)",
    )
    def test_expands_contractions_with_curly_apostrophe(self):
        assert clean_light("I don’t like it") == "i do not like it"


class TestCleanHeavy:
    """Heavy cleaning feeds TF-IDF models: stopwords removed, words lemmatised."""

    def test_removes_stopwords_and_lemmatises(self):
        assert clean_heavy("The games were AMAZING") == "game amazing"

    @pytest.mark.parametrize("raw", [None, "", "2021"])
    def test_empty_after_cleaning_returns_empty_string(self, raw):
        assert clean_heavy(raw) == ""

    @pytest.mark.xfail(
        strict=True,
        reason="Known bug: NLTK's stopword list contains 'not', 'no' and 'nor', so "
        "negations are deleted and 'not good' becomes 'good' (affects ~34% of reviews)",
    )
    def test_keeps_negations(self):
        assert "not" in clean_heavy("This game is not good").split()


class TestPreprocessDataframe:
    def test_adds_both_cleaned_columns(self):
        df = pd.DataFrame({"review": ["A genuinely great game to play"]})
        result = preprocess_dataframe(df)
        assert {"review_clean", "review_heavy"} <= set(result.columns)

    def test_drops_missing_and_short_reviews(self):
        df = pd.DataFrame({"review": ["A genuinely great game to play", None, "ok", "♥♥♥"]})
        result = preprocess_dataframe(df, min_length=10)
        assert result["review"].tolist() == ["A genuinely great game to play"]

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"review": ["A genuinely great game to play", "ok"]})
        preprocess_dataframe(df)
        assert list(df.columns) == ["review"]
        assert len(df) == 2

    def test_resets_index(self):
        df = pd.DataFrame({"review": ["ok", "A genuinely great game to play"]})
        result = preprocess_dataframe(df)
        assert result.index.tolist() == [0]
