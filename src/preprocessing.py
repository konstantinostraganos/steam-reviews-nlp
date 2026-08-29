"""
Text preprocessing pipeline for Steam game reviews.

Two preprocessing modes:
- 'light': minimal cleaning, keeps text close to original (for transformer models)
- 'heavy': aggressive cleaning with lemmatization (for TF-IDF / traditional ML)
"""

import re
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# --- Contractions dictionary ---
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am", "let's": "let us",
    "it's": "it is", "i'm": "i am", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "wasn't": "was not",
    "weren't": "were not", "isn't": "is not", "aren't": "are not",
    "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not",
}

# --- Individual cleaning functions ---

def remove_year_prefix(text):
    """Remove leading year from review (e.g. '2021 Great game' -> 'Great game')."""
    return re.sub(r"^\d{4}\s+", "", text)


def remove_early_access(text):
    """Remove 'Early Access Review' prefix."""
    return re.sub(r"(?i)^early access review\s*", "", text)


def expand_contractions(text):
    """Expand contractions (e.g. don't -> do not)."""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_hearts(text):
    """Remove Steam's censored profanity symbols (♥)."""
    return re.sub(r"♥+", "", text)


def remove_non_latin(text):
    """Remove non-Latin characters (keeps basic ASCII + common punctuation)."""
    return re.sub(r"[^\x00-\x7F]+", " ", text)


def clean_whitespace(text):
    """Normalize whitespace — collapse multiple spaces, strip edges."""
    return re.sub(r"\s+", " ", text).strip()

# --- Pipeline functions ---

def clean_light(text):
    """Light preprocessing — for transformer models (DistilBERT).
    
    Keeps stopwords and original word forms intact.
    Only removes noise that would confuse any model.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = remove_year_prefix(text)
    text = remove_early_access(text)
    text = text.lower()
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_hearts(text)
    text = remove_non_latin(text)
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters and spaces
    text = clean_whitespace(text)
    return text


def clean_heavy(text):
    """Heavy preprocessing — for TF-IDF / traditional ML models.
    
    Applies stopword removal and lemmatization on top of light cleaning.
    """
    text = clean_light(text)
    if not text:
        return ""
    
    # Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df, text_col="review", min_length=10):
    """Apply full preprocessing pipeline to a DataFrame.
    
    Returns a new DataFrame with:
    - 'review_clean': light-cleaned text (for transformers)
    - 'review_heavy': heavy-cleaned text (for TF-IDF / traditional ML)
    - rows with empty/short reviews removed
    """
    df = df.copy()
    
    # Drop missing reviews
    df = df.dropna(subset=[text_col])
    
    # Apply both cleaning modes
    df["review_clean"] = df[text_col].apply(clean_light)
    df["review_heavy"] = df[text_col].apply(clean_heavy)
    
    # Remove reviews that are too short after cleaning
    df = df[df["review_clean"].str.len() >= min_length]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df    