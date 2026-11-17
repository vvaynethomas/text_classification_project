"""Train and compare classifiers on a labeled text collection.

    python classify.py                                  # data/story.data.csv, author labels, all models
    python classify.py --models SVM random_forest KNN MNB MLR --plots
    python classify.py --data path/to/folder            # one sub-folder per label (.txt/.docx/.pdf)
"""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from textclf import MODELS, TextClassifier, corpify  # noqa: E402
from textclf.config import DEFAULT_DATA, OUTPUT_DIR  # noqa: E402

FAST_MODELS = ["random_forest", "SVM", "KNN", "MNB", "MLR"]


def configure_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "run.log", "w")
    file_handler.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in (file_handler, console):
        handler.setFormatter(fmt)
    root = logging.getLogger("textclf")
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV file or folder of documents")
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--label-column", default="Author")
    parser.add_argument("--title-column", default="Title")
    parser.add_argument("--models", nargs="+", default=FAST_MODELS, choices=["best", *MODELS],
                        help="models to fit (default: all but GBM); 'best' fits all six. The best model is "
                             "the one with the highest 5-fold cross-validated accuracy on the training "
                             "set (ties go to the simpler model); test accuracy is only reported")
    parser.add_argument("--threshold", type=int, default=3,
                        help="drop labels with this many examples or fewer (also the TF-IDF min_df)")
    parser.add_argument("--project", help="name used for output files (default: data file or folder name)")
    parser.add_argument("--clean-texts", action="store_true",
                        help="lowercase, strip punctuation, lemmatize and remove stopwords before TF-IDF")
    parser.add_argument("--plots", action="store_true", help="save confusion matrices and PCA/t-SNE plots")
    parser.add_argument("--resume", action="store_true", help="reuse outputs/<project>.p if present")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(args.output_dir)
    if args.data.suffix == ".csv":
        corpus = corpify(str(args.data), args.text_column, args.label_column, args.title_column)
    else:
        corpus = corpify(str(args.data))
    model = "best" if args.models == ["best"] else args.models
    classifier = TextClassifier(inp=corpus, model=model, project=args.project or args.data.stem, threshold=args.threshold,
                                clean_texts=args.clean_texts, plot_bool=args.plots, load_pickle=args.resume,
                                output_dir=args.output_dir)
    if args.plots:
        classifier.dim_red()
    if classifier.best_mod_key:
        print(f"Best model (highest CV accuracy on the training set): {classifier.best_mod_key}")
    return classifier


if __name__ == "__main__":
    main()
