# Text classification project

Explore **classical machine learning for labeled documents**, from extracting text and building TF-IDF features to comparing classifiers and inspecting predictions.

The project's prototypes connect document handling with model exploration. A concrete example is **author classification for short stories**: `main()` in `text_classification.py` sketches reading story content, author labels, and titles from a local `story.data.csv`, then passing the corpus to a classifier.

## What it explores

- **Document representations:** extraction helpers for DOCX, PDF, and TXT, `Document` and `Corpus` objects, and labeled pandas dataframes.
- **Text features:** lowercasing, punctuation cleanup, lemmatization, stopword removal, and TF-IDF with unigram/bigram settings.
- **Model comparison:** Random Forest, support-vector machines, k-nearest neighbors, multinomial Naive Bayes, logistic regression, and gradient boosting, including grid/randomized hyperparameter searches.
- **Evaluation and interpretation:** train/test accuracy, classification reports, confusion matrices, chi-squared feature inspection, and PCA/t-SNE visualization methods.

## Find your way around

| File | What to look for |
| --- | --- |
| [text_extraction.py](text_extraction.py) | Shared document/corpus data model and extraction and CSV/dataframe helpers. |
| [text_classifier.py](text_classifier.py) | Classifier variant built around the shared extraction module. |
| [text_classifier2.py](text_classifier2.py) | Input-handling experiments, rare-label filtering, TF-IDF configuration, and model comparison. |
| [text_classification.py](text_classification.py) | Combined extraction/classification variant and the short-story example. |
| [chatbotstuff.py](chatbotstuff.py) | Separate VADER sentiment-analyzer import. |

The classifier files are alternative implementations with different constructors. For a useful reading path:

1. Start with `Document` and `Corpus` in `text_extraction.py` to follow text, names, and labels.
2. Read `TextClassifier.__init__` in `text_classifier2.py` for label encoding, train/test splitting, and `TfidfVectorizer`.
3. Follow a model method such as `MNB` or `SVM`, then explore `model_comparison`, `mod_predict`, and `dim_red`.

## Exploring with your own texts

The shared corpus dataframe uses `Name`, `Text`, and `Label`; file-based documents take their label from the parent directory. The short-story sketch maps CSV columns `content`, `Author`, and `Title` to those roles. Supply your own labeled texts; the example CSV is not bundled.

For local experiments, use an isolated Python environment. The main dependencies are NumPy, pandas, scikit-learn, NLTK, matplotlib, seaborn, PyPDF2, and python-docx. Compatible versions are unpinned, and older APIs such as `DataFrame.append` and `PdfFileReader` require compatibility work before running a chosen variant.

Imports can load/download NLTK resources and create logs; classifier construction can start training and parameter searches. Begin with a small dataset, inspect extracted text, and review logs/model outputs before sharing them.

The model-comparison and evaluation methods provide a starting point for investigating how feature choices and classifier families affect predictions on your corpus.
