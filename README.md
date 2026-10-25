# Text classification project

This repository contains several related, but not interchangeable, text-classification prototypes and an extraction module. They are preserved as historical variants; file numbering does not establish which one is authoritative. No variant has been verified here as a reproducible, end-to-end entry point.

## Files

| File | Role |
| --- | --- |
| `text_extraction.py` | Defines `Document` and `Corpus`, reads document text, and converts some CSV/dataframe inputs. `text_classifier.py` imports these classes. |
| `text_classifier.py` | Classifier implementation that uses `text_extraction.py`. Includes TF-IDF preprocessing, model training/evaluation, prediction helpers, and plotting. |
| `text_classifier2.py` | A separate classifier variant with its own implementation and attempted CSV, directory, corpus, and pickle inputs. It also imports `text_extraction.py`. Its command-line example is commented out. |
| `text_classification.py` | A larger, mostly self-contained variant that duplicates extraction and classification logic. Its `main()` refers to a hard-coded `story.data.csv`; it is not a verified command-line workflow. |
| `chatbotstuff.py` | Imports VADER's `SentimentIntensityAnalyzer`; it does not connect that analyzer to the classifiers. |

The classifier variants contain TF-IDF text features and implementations for Random Forest, support-vector, k-nearest-neighbor, multinomial Naive Bayes, logistic-regression, and gradient-boosting models. They also contain model-comparison, prediction, and dimensionality-reduction/plotting code. These are code capabilities, not claims that each path currently runs or that any particular model performed well.

## Inputs and outputs in the code

The directory-based paths expect labeled documents organized under directories whose names are used as labels. `text_extraction.py` can extract DOCX, PDF, and TXT text when given a `Document` path, but its directory collector selects DOCX, PDF, and legacy DOC files; legacy DOC extraction is not implemented, and TXT files are not selected by that collector.

`text_classifier2.py` also contains paths for CSV input with text and label columns, a `Corpus` object, and a pickled model. CSV column selection can prompt interactively, and the conversion path is not verified; these should not be treated as stable interfaces. The other classifier variants have different constructors and data-handling code.

Training code splits labeled data into training and test sets and prints model-comparison results. Prediction helpers return a label and may print a conditional probability. Plotting methods write image files, and some variants write pickle files. These outputs depend on the path and method used; there is no shared, documented output format or included example result.

## Environment and reproduction status

There is no dependency manifest, setup script, sample corpus, expected output, or test suite in the repository. Imports across the variants require packages including NumPy, pandas, NLTK, scikit-learn, matplotlib, and seaborn; document extraction also uses PyPDF2 and python-docx, and `chatbotstuff.py` imports `vaderSentiment`. Exact compatible versions are not recorded.

The classifier modules load or download NLTK resources such as `words`, `stopwords`, `punkt`, and `wordnet` at import or initialization time; some document-analysis methods also use part-of-speech tagging. A clean run can therefore require network access and additional NLTK data. The repository does not establish a verified Python version, dependency set, or runnable starting command. Choose none of the variants as a verified entry point until a small, permitted sample dataset and clean-environment smoke test are available.

No data or Colorism-specific label scheme is included, and no experimental results are reproduced or reported here. Before applying this prototype to a Colorism research question, a separate study plan should define the task and labels, document data provenance and permission, preserve a reviewable sample/split, and specify baselines and evaluation measures. That is proposed future work, not a result or capability established by this repository.

## Generated files and data handling

The tracked IDE settings, bytecode caches, and log files were local/generated workspace artifacts, not source inputs. The removed logs contained corpus-derived content; their contents are intentionally not reproduced here. The scripts can create debug logs and model/plot outputs in the working directory, so review generated files for sensitive text before sharing them.
