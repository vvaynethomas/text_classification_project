"""Document and Corpus objects, plus text extraction from DOCX, PDF, TXT and CSV sources."""
import logging
import os

import docx
import nltk
import pandas as pd
from pypdf import PdfReader

from .preprocessing import ensure_nltk_data

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".docx", ".pdf", ".txt")
CONTRACTIONS = ["I'm", "I'd", "i'm", "i'd", "I’d", "I’m", "i", "I"]
_english_vocab = None


def extract_string_from_docx(doc_path):
    try:
        doc = docx.Document(doc_path)
    except Exception:
        logger.warning("Could not read %s", doc_path)
        return ""
    return " ".join(run.text for paragraph in doc.paragraphs for run in paragraph.runs)


def extract_string_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            return ""
        # The last page is skipped, as are pages stamped by journal download services.
        pages = [page.extract_text() for page in reader.pages[:-1]]
    except Exception:
        logger.warning("Could not parse %s", pdf_path)
        return ""
    return " ".join(p for p in pages if "content downloaded from" not in p)


def extract_string_from_txt(txt_path):
    with open(txt_path, mode="r", encoding="utf-8", errors="replace") as file:
        return " ".join(file.readlines())


def get_document_paths(directory):
    """All supported documents under `directory`, skipping Office lock files."""
    paths = []
    for root, _dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(SUPPORTED_EXTENSIONS) and not filename.startswith("~$"):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def df_to_corpus(df, text_column_name, label_column_name, title_column_name=None, name=None):
    logger.info("Building corpus from dataframe")
    corpus = Corpus(name=name)
    for _, row in df.iterrows():
        title = row[title_column_name] if title_column_name else None
        corpus.add_document(Document(text=row[text_column_name], label=row[label_column_name], name=title))
    logger.info("Corpus holds %d documents", len(corpus.documents))
    return corpus


def doc_to_df(document):
    return pd.DataFrame({"Name": [document.name], "Text": [document.text], "Label": [document.label]})


def csv_to_corpus(csv_path, text_column_name, label_column_name, title_column_name=None, delimiter=None):
    df = pd.read_csv(csv_path, delimiter=delimiter)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    return df_to_corpus(df, text_column_name, label_column_name, title_column_name, name=name)


def corpify(source, text_column_name=None, label_column_name=None, title_column_name=None):
    """Turn a Document, DataFrame, CSV path, TXT path or directory into a Corpus."""
    if isinstance(source, Document):
        return Corpus(documents=[source])
    if isinstance(source, pd.DataFrame):
        return df_to_corpus(source, text_column_name, label_column_name, title_column_name)
    if isinstance(source, (str, os.PathLike)):
        source = str(source)
        if source.endswith(".csv"):
            return csv_to_corpus(source, text_column_name, label_column_name, title_column_name)
        if os.path.isdir(source):
            return Corpus(directory=source)
        if source.endswith(SUPPORTED_EXTENSIONS):
            return Corpus(documents=[Document(path=source)])
    raise TypeError(f"Cannot build a corpus from {type(source).__name__}: {source!r}")


class Document:
    """A single text with a name and a label. File-based documents take the parent folder as label."""

    def __init__(self, path=None, name=None, label=None, text=None):
        if path:
            self.path = str(path)
            self.label = os.path.basename(os.path.dirname(self.path))
            self.name = os.path.basename(self.path).split(".")[0]
            self.text = self._extract_text()
        else:
            self.path = ""
            self.label = str(label).strip()
            self.text = text
            self.name = name
        self._words = None

    def __repr__(self):
        return f"{self.label} : {self.text}"

    def __str__(self):
        return f"Document object named {self.name} and labeled as {self.label}"

    def __add__(self, other):
        if isinstance(other, Document):
            return Corpus(documents=[self, other])
        return NotImplemented

    def _extract_text(self):
        if self.path.endswith(".docx"):
            return extract_string_from_docx(self.path)
        if self.path.endswith(".pdf"):
            return extract_string_from_pdf(self.path)
        if self.path.endswith(".txt"):
            return extract_string_from_txt(self.path)
        return ""

    @property
    def words(self):
        """Tokens with surrounding punctuation stripped; short non-dictionary tokens are dropped."""
        if self._words is None:
            self._words = self._get_word_list()
        return self._words

    def _get_word_list(self):
        global _english_vocab
        ensure_nltk_data("punkt_tab", "words")
        if _english_vocab is None:
            _english_vocab = set(w.lower() for w in nltk.corpus.words.words())
        final_words = []
        for word in nltk.tokenize.word_tokenize(self.text):
            word = word.strip("".join(c for c in word if not c.isalnum()))
            if len(word) <= 3 and word not in CONTRACTIONS and word.lower() not in _english_vocab:
                word = ""
            if len(word) == 1 and word not in ["a", "i", "I"]:
                word = ""
            if word:
                final_words.append(word)
        return final_words

    def word_count(self):
        return len(self.words)

    def noun_count(self):
        ensure_nltk_data("averaged_perceptron_tagger_eng")
        return len([word for word, tag in nltk.pos_tag(self.words) if "NN" in tag])

    def get_unique(self):
        return sorted(set(word.lower() for word in self.words))

    def count_unique(self):
        return len(self.get_unique())


class Corpus:
    """A collection of Documents, viewable as a dataframe with Name, Text and Label columns."""

    def __init__(self, directory=None, name=None, documents=None):
        self.documents = list(documents or [])
        self.name = name or "corpus"
        if directory:
            for path in get_document_paths(directory):
                self.add_document(Document(path))
            if not name:
                self.name = os.path.basename(os.path.normpath(directory))

    def __add__(self, other):
        if isinstance(other, Corpus):
            return Corpus(name=self.name, documents=self.documents + other.documents)
        if isinstance(other, Document):
            return Corpus(name=self.name, documents=self.documents + [other])
        return NotImplemented

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    def add_document(self, document):
        self.documents.append(document)

    @property
    def df(self):
        return pd.DataFrame([{"Name": d.name, "Text": d.text, "Label": d.label} for d in self.documents],
                            columns=["Name", "Text", "Label"])
