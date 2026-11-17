"""Classical text classification: document extraction, TF-IDF features and model comparison."""
from .classifier import MODELS, TextClassifier
from .documents import Corpus, Document, corpify

__all__ = ["MODELS", "TextClassifier", "Corpus", "Document", "corpify"]
