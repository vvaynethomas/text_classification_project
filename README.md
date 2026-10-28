# Text classification project

An exploration of **classical machine learning for labeled documents**: extract text, represent it with TF-IDF features, and compare classifiers that predict a document's category.

The code connects document handling with the modeling process, from `Document` and `Corpus` objects to training/test splits, model comparison, and prediction helpers. A short-story example in `text_classification.py` uses story content to predict author labels, illustrating one possible task rather than a bundled dataset or reported experiment.

This is an experimental source-code collection, not a packaged application. Start with the reading guide below to explore the ideas; the variants need repairs and a compatible environment before an end-to-end run can be relied on.

## What it explores

- **Document ingestion:** extracting strings from DOCX, PDF, and TXT files, grouping documents into corpora, and working with labeled pandas dataframes.
- **Text features:** lowercasing, punctuation cleanup, lemmatization, stopword removal, and TF-IDF vectorization. `text_classifier2.py` includes unigram/bigram settings and a feature-count limit.
- **Model comparison:** methods for Random Forest, support-vector machines, k-nearest neighbors, multinomial Naive Bayes, multinomial logistic regression, and gradient boosting, with grid/randomized hyperparameter searches in several methods.
- **Evaluation and interpretation:** training/test accuracy, classification reports, confusion matrices, chi-squared feature inspection, and PCA/t-SNE visualization code.

These are techniques represented in the source, not a claim that every implementation currently works.

## Find your way around

| File | What to look for |
| --- | --- |
| [text_extraction.py](text_extraction.py) | Shared extraction functions, `Document` and `Corpus`, and CSV/dataframe conversion helpers. Start here for the data model. |
| [text_classifier.py](text_classifier.py) | Classifier variant importing the shared extraction module, with preprocessing, model methods, evaluation, and prediction helpers. |
| [text_classifier2.py](text_classifier2.py) | Another variant exploring CSV, directory, `Corpus`, and pickle inputs, plus rare-label filtering and TF-IDF configuration. Its command-line example is commented out. |
| [text_classification.py](text_classification.py) | Mostly self-contained variant combining extraction and classification. Its `main()` sketches the short-story/author task using a local `story.data.csv` file that is not included. |
| [chatbotstuff.py](chatbotstuff.py) | A single VADER sentiment-analyzer import, separate from the classification pipeline; not a chatbot implementation. |

The three classifier files are alternatives, not sequential pipeline stages. Their constructors and data handling differ, and file numbering does not identify a recommended or fully working version.

### Suggested reading path

1. Read `Document`, `Corpus`, and the extraction helpers in `text_extraction.py` to see how text, names, and labels are represented.
2. Follow `TextClassifier.__init__` in `text_classifier2.py` through label encoding, the train/test split, and `TfidfVectorizer`. Compare its ingestion paths with the shared corpus helpers.
3. Pick one model method, such as `MNB` or `SVM`, then follow `model_comparison`, `mod_predict`, and `dim_red` to see the intended evaluation and exploration workflow.

## Inputs, outputs, and results

**Inputs:** the shared corpus dataframe uses `Name`, `Text`, and `Label` columns. For file-based documents, the parent directory name becomes the label. CSV conversion helpers accept column names for text, labels, and titles; the CSV path in `text_classifier2.py` also attempts interactive column selection. These paths are experimental rather than stable interfaces.

**Outputs in the code:** model-comparison tables and printed evaluation reports, category-prediction helpers, and image-writing paths for confusion matrices and dimensionality reduction. Some variants also contain pickle serialization code. Output behavior varies by file and method.

**Results:** no sample corpus, trained model, benchmark scores, or reproduced experiment is included. The evaluation methods show how comparisons were intended to be made, but do not establish which model performs best. No Colorism-specific dataset, labels, or validated research findings are provided.

## Local exploration

Reading the source requires no dataset or dependency installation. To experiment locally, use a separate Python environment rather than your system environment:

```powershell
git clone https://github.com/vvaynethomas/text_classification_project.git
cd text_classification_project
python -m venv .venv
```

This creates an isolated environment; it does **not** install dependencies or make the prototypes runnable. Keep the environment out of commits.

| Area | Packages referenced |
| --- | --- |
| Data and modeling | NumPy, pandas, scikit-learn |
| Language processing | NLTK |
| Visualization | matplotlib, seaborn |
| Document extraction | PyPDF2, python-docx |
| Separate sentiment stub | vaderSentiment |

There is no dependency manifest, recorded compatible Python/package version set, setup script, or test suite. Older APIs such as pandas `DataFrame.append` and PyPDF2 `PdfFileReader` mean installing the latest packages is not a verified setup recipe.

NLTK resources include `words`, `stopwords`, `punkt`, and `wordnet`; some document-analysis methods also use part-of-speech tagging. Imports can read NLTK corpora, trigger downloads, and create log files before you explicitly train anything.

Before trying a run, select one variant, resolve its data-handling/API issues, and use a small labeled dataset you are allowed to process. There is no verified CLI command to recommend: the `main()` functions are incomplete or tied to local inputs, and constructing a classifier can immediately start training and hyperparameter searches.

## Practical limitations

- **Ingestion needs care.** The directory collector selects DOCX, PDF, and legacy DOC files, but legacy DOC extraction is not implemented and TXT files are not collected. The PDF extractor also has page-handling limitations, so inspect extracted text before modeling.
- **The pipeline needs a smoke test.** Dataframe assembly, column naming, preprocessing order, and model selection differ across variants and include unfinished paths. Do not treat prediction helpers or plots as a supported API.
- **Data stays your responsibility.** Scripts can write corpus-derived text to debug logs and produce model/plot files. Use appropriate data, review generated outputs before sharing, and load pickle files only from trusted sources.

The useful starting point is the source-level exploration of document representations and classical classifiers, not a production-ready model or a reproducible performance claim.
