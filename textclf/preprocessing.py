"""Text cleaning helpers: lowercasing, punctuation removal, lemmatization, stopwords."""
import re

import nltk

PUNCTUATION_SIGNS = list("?:!.,;")

_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "words": "corpora/words",
    "punkt_tab": "tokenizers/punkt_tab",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}
_stop_word_patterns = None


def ensure_nltk_data(*names):
    """Download the named NLTK resources if they are not installed yet."""
    for name in names or _NLTK_RESOURCES:
        try:
            nltk.data.find(_NLTK_RESOURCES[name])
        except LookupError:
            nltk.download(name, quiet=True)


def clean_text(text):
    text = text.replace("\r", " ").replace("\n", " ").replace("    ", " ")
    text = text.lower()
    for punct_sign in PUNCTUATION_SIGNS:
        text = text.replace(punct_sign, "")
    return text.replace("'s", "")


def lemmatize_text(text):
    ensure_nltk_data("wordnet")
    lemmatizer = nltk.WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word, pos="v") for word in text.split(" "))


def remove_stopwords(text):
    global _stop_word_patterns
    if _stop_word_patterns is None:
        ensure_nltk_data("stopwords")
        _stop_word_patterns = [re.compile(r"\b" + re.escape(word) + r"\b")
                               for word in nltk.corpus.stopwords.words("english")]
    for pattern in _stop_word_patterns:
        text = pattern.sub("", text)
    return text


def parse_words(text):
    """Full cleaning pipeline for one document."""
    return remove_stopwords(lemmatize_text(clean_text(text)))


def parse_text_column(df):
    df["Text"] = df["Text"].apply(parse_words)
    return df
