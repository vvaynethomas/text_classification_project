"""Classical text classification: document extraction, TF-IDF features and model comparison."""
from .classifier import MODELS, TextClassifier
from .documents import Corpus, Document, corpify
from .selection import select_best

__all__ = ["MODELS", "TextClassifier", "Corpus", "Document", "corpify", "select_best"]
