"""Fast end-to-end check on a small synthetic corpus (no downloads, no real data)."""
import random

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from textclf import Corpus, TextClassifier, corpify  # noqa: E402
from textclf.preprocessing import clean_text  # noqa: E402

VOCAB = {
    "Author A": "ship sea captain storm harbor sail wave anchor",
    "Author B": "robot planet laser orbit rocket alien star engine",
    "Author C": "garden rose cottage tea village parlour aunt letter",
}


def synthetic_stories(n_per_author=10, seed=0):
    rng = random.Random(seed)
    rows = []
    for author, words in VOCAB.items():
        words = words.split()
        for i in range(n_per_author):
            text = " ".join(rng.choice(words) for _ in range(60))
            rows.append({"content": text, "Author": author, "Title": f"{author} story {i}"})
    return pd.DataFrame(rows)


def test_classifier_runs_and_separates_authors(tmp_path):
    corpus = corpify(synthetic_stories(), "content", "Author", "Title")
    clf = TextClassifier(inp=corpus, model=["MNB", "SVM", "MLR"], project="smoke", output_dir=tmp_path)
    assert set(clf.model_comparison["Model"]) == {"MNB", "SVM", "MLR"}
    assert clf.model_comparison["Test Set Accuracy"].max() == 1.0
    assert (tmp_path / "smoke.p").exists()
    assert (tmp_path / "model_comparison_smoke.csv").exists()
    clf.best_mod_key, clf.best_mod = "MNB", clf.models["MNB"]
    assert clf.mod_predict("rocket orbit alien laser star") == "Author B"


def test_csv_and_directory_inputs(tmp_path):
    df = synthetic_stories(n_per_author=5)
    csv_path = tmp_path / "stories.csv"
    df.to_csv(csv_path, index=False)
    assert len(corpify(str(csv_path), "content", "Author", "Title")) == 15

    for _, row in df.iterrows():
        folder = tmp_path / "docs" / row["Author"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{row['Title']}.txt").write_text(row["content"])
    corpus = Corpus(directory=tmp_path / "docs")
    assert len(corpus) == 15
    assert set(corpus.df["Label"]) == set(VOCAB)


def test_clean_text():
    assert clean_text("Hello, World!\nIt's here.") == "hello world it here"
