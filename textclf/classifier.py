"""TF-IDF features plus a family of scikit-learn classifiers, tuned and compared on one split."""
import logging
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from .config import OUTPUT_DIR
from .documents import Corpus, Document, csv_to_corpus
from .preprocessing import parse_text_column, parse_words

logger = logging.getLogger(__name__)

RANDOM_STATE = 8
# Model key -> name shown in the comparison table.
MODELS = {
    "random_forest": "Random_forest",
    "SVM": "SVM",
    "KNN": "KNN",
    "MNB": "MNB",
    "MLR": "MLR",
    "GBM": "GBM",
}
COMPARISON_COLUMNS = ["Model", "Training Set Accuracy", "Test Set Accuracy"]


def _shuffle_cv(test_size=.33):
    # GridSearchCV has no random_state, so the CV splits are fixed by hand.
    return ShuffleSplit(n_splits=3, test_size=test_size, random_state=RANDOM_STATE)


def _grid_from_best(best_params):
    return {key: [value] for key, value in best_params.items()}


class TextClassifier:
    """Load labeled texts, build TF-IDF features and train/compare classifiers.

    `inp` is a Corpus, a CSV path (pass `text_column` and `label_column`), a directory whose
    sub-folders name the labels, or a pickled classifier (`.p`). `model` is a key of MODELS,
    a list of keys, or "best" to run all of them and keep the best non-overfitted model.
    """

    def __init__(self, inp,
                 delimiter=None,
                 method="tfidf",
                 model=None,
                 parameters=None,
                 plot_bool=False,
                 project=None,
                 pickle_bool=True,
                 threshold=3,
                 text_column=None,
                 label_column=None,
                 title_column=None,
                 clean_texts=False,
                 load_pickle=False,
                 output_dir=OUTPUT_DIR):
        logger.info("Initializing Classifier Object")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_mod_key = None

        if project and load_pickle and self._pickle_path(project).exists():
            logger.info("Loading pickled classifier %s", self._pickle_path(project))
            with open(self._pickle_path(project), "rb") as f:
                self.__dict__.update(pickle.load(f).__dict__)
            return

        if isinstance(inp, (str, os.PathLike)) and str(inp).endswith(".p"):
            with open(inp, "rb") as f:
                self.best_mod = pickle.load(f)
            return

        self.plot_bool = plot_bool
        self.models = {}
        self.model_comparison = pd.DataFrame(columns=COMPARISON_COLUMNS)
        self.corpora = [self._load_corpus(inp, delimiter, text_column, label_column, title_column)]
        self.project = project or self.corpora[0].name
        self.all_data = pd.concat([c.df for c in self.corpora], ignore_index=True)
        logger.debug("ingested corpus has %d texts", len(self.all_data))

        # Drop labels with `threshold` or fewer examples.
        counts = self.all_data["Label"].value_counts()
        self.all_data = self.all_data[self.all_data["Label"].isin(counts[counts > threshold].index)].copy()
        logger.debug("Summary of text data to be classified: %s", self.all_data.describe())

        self.clean_texts = clean_texts
        if clean_texts:
            self.all_data = parse_text_column(self.all_data.copy())

        self.method = method
        self.categories = set(self.all_data["Label"])
        self.category_codes = {label: str(i) for i, label in enumerate(sorted(self.categories))}
        self.category_names = {code: label for label, code in self.category_codes.items()}
        self.all_data["Category_Code"] = self.all_data["Label"].map(self.category_codes)
        logger.debug("category codes: %s", self.category_codes)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.all_data["Text"], self.all_data["Category_Code"], test_size=0.15, random_state=RANDOM_STATE)

        if method == "tfidf":
            self._build_tfidf(parameters, threshold)

        if model:
            keys = list(MODELS) if model == "best" else ([model] if isinstance(model, str) else list(model))
            for key in keys:
                self.run_model(key)
            print(self.model_comparison.sort_values("Test Set Accuracy", ascending=False))
            self._select_best()
            self.model_comparison.to_csv(self.output_dir / f"model_comparison_{self.project}.csv", index=False)

        if pickle_bool:
            with open(self._pickle_path(self.project), "wb") as f:
                pickle.dump(self, f)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_corpus(inp, delimiter, text_column, label_column, title_column):
        if isinstance(inp, Corpus):
            return inp
        path = str(inp)
        if path.endswith(".csv"):
            if not (text_column and label_column):
                text_column, label_column = TextClassifier._guess_columns(pd.read_csv(path, delimiter=delimiter))
            return csv_to_corpus(path, text_column, label_column, title_column, delimiter=delimiter)
        if os.path.isdir(path):
            logger.info("Reading documents from %s (labels are sub-folder names)", path)
            return Corpus(directory=path)
        raise ValueError(f"Unsupported input: {inp!r}")

    @staticmethod
    def _guess_columns(df):
        """For a two-column CSV, the column with the longest entries is the text."""
        if len(df.columns) != 2:
            raise ValueError(f"Pass text_column and label_column; CSV has columns {list(df.columns)}")
        first, second = df.columns
        if df[first].astype(str).str.len().max() > df[second].astype(str).str.len().max():
            return first, second
        return second, first

    def _pickle_path(self, project):
        return self.output_dir / f"{project}.p"

    def _build_tfidf(self, parameters, threshold):
        if parameters:
            ngram_range, min_df, max_df, max_features = parameters
        else:
            ngram_range, min_df, max_df, max_features = (1, 2), threshold, 1., 300
        self.tfidf = TfidfVectorizer(encoding="utf-8",
                                     ngram_range=ngram_range,
                                     stop_words=None,
                                     lowercase=False,
                                     max_df=max_df,
                                     min_df=min_df,
                                     max_features=max_features,
                                     norm="l2",
                                     sublinear_tf=True)
        self.features_train = self.tfidf.fit_transform(self.X_train).toarray()
        self.labels_train = self.y_train
        self.features_test = self.tfidf.transform(self.X_test).toarray()
        self.labels_test = self.y_test
        logger.info("tfidf complete, training and test features and labels stored")

    # ------------------------------------------------------------------ models
    def run_model(self, key):
        if key not in MODELS:
            raise ValueError(f"Unknown model {key!r}; choose from {list(MODELS)} or 'best'")
        logger.info("Fitting %s", key)
        model, comp = getattr(self, key)()
        self.models[key] = model
        self.model_comparison = (comp if self.model_comparison.empty
                                 else pd.concat([self.model_comparison, comp], ignore_index=True))
        logger.debug("Model: %s", model)
        logger.debug("model comparison so far: %s", self.model_comparison)
        return model, comp

    def _select_best(self, prefer_not_overfitted=True):
        df = self.model_comparison
        if prefer_not_overfitted and (df["Training Set Accuracy"] <= .98).any():
            df = df[df["Training Set Accuracy"] <= .98]
        best_name = df.loc[df["Test Set Accuracy"].astype(float).idxmax(), "Model"]
        self.best_mod_key = next(k for k, v in MODELS.items() if v == best_name)
        self.best_mod = self.models[self.best_mod_key]
        return self.best_mod

    def best(self):
        for key in MODELS:
            self.run_model(key)
        return self._select_best(), self.model_comparison

    def _search(self, estimator, param_grid, label):
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring="accuracy",
                                   cv=_shuffle_cv(), verbose=1)
        grid_search.fit(self.features_train, self.labels_train)
        print(f"{label} grid search best hyperparameters: {grid_search.best_params_}")
        print(f"Mean CV accuracy with these hyperparameters: {grid_search.best_score_}")
        return grid_search.best_estimator_

    def _random_then_grid(self, estimator, random_grid, label):
        """Randomized search over `random_grid`, then a grid search pinned to its best point."""
        random_search = RandomizedSearchCV(estimator=estimator, param_distributions=random_grid, n_iter=50,
                                           scoring="accuracy", cv=3, verbose=1, random_state=RANDOM_STATE)
        random_search.fit(self.features_train, self.labels_train)
        logger.debug("%s random search best hyperparameters: %s", label, random_search.best_params_)
        print(f"{label} random search best hyperparameters: {random_search.best_params_}")
        print(f"Mean CV accuracy with these hyperparameters: {random_search.best_score_}")
        return self._search(estimator, _grid_from_best(random_search.best_params_), label)

    def _evaluate(self, key, estimator):
        """Fit on the training set, report train/test accuracy and optionally plot a confusion matrix."""
        estimator.fit(self.features_train, self.labels_train)
        test_pred = estimator.predict(self.features_test)
        train_acc = accuracy_score(self.labels_train, estimator.predict(self.features_train))
        test_acc = accuracy_score(self.labels_test, test_pred)
        print(f"{MODELS[key]} training accuracy: {train_acc}")
        print(f"{MODELS[key]} test accuracy: {test_acc}")
        print(classification_report(self.labels_test, test_pred, zero_division=0))

        if self.plot_bool:
            codes = sorted(self.category_names)
            conf_matrix = confusion_matrix(self.labels_test, test_pred, labels=codes)
            names = [self.category_names[c] for c in codes]
            plt.figure(figsize=(12.8, 6))
            sns.heatmap(conf_matrix, annot=True, xticklabels=names, yticklabels=names, cmap="Blues")
            plt.ylabel("Predicted")
            plt.xlabel("Actual")
            plt.title("Confusion matrix")
            plt.savefig(self.output_dir / f"{key}_heatmap_{self.project}.png", bbox_inches="tight")
            plt.close()

        comp = pd.DataFrame({"Model": MODELS[key], "Training Set Accuracy": train_acc,
                             "Test Set Accuracy": test_acc}, index=[0])
        return estimator, comp

    def random_forest(self):
        random_grid = {"n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=5)],
                       "max_features": ["sqrt"],
                       "max_depth": [int(x) for x in np.linspace(20, 100, num=5)] + [None],
                       "min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 2, 4],
                       "bootstrap": [True, False]}
        best = self._random_then_grid(RandomForestClassifier(random_state=RANDOM_STATE), random_grid, "random forest")
        return self._evaluate("random_forest", best)

    def SVM(self):
        C = [.0001, .001, .01, .1]
        param_grid = [
            {"C": C, "kernel": ["linear"], "probability": [True]},
            {"C": C, "kernel": ["poly"], "degree": [3, 4, 5], "probability": [True]},
            {"C": C, "kernel": ["rbf"], "gamma": [1, 10, 100], "probability": [True]},
        ]
        best = self._search(svm.SVC(random_state=RANDOM_STATE), param_grid, "SVM")
        return self._evaluate("SVM", best)

    def KNN(self):
        param_grid = {"n_neighbors": [int(x) for x in np.linspace(start=1, stop=200, num=100)]}
        grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring="accuracy",
                                   cv=_shuffle_cv(test_size=.15), verbose=1)
        grid_search.fit(self.features_train, self.labels_train)
        print(f"KNN grid search best hyperparameters: {grid_search.best_params_}")
        return self._evaluate("KNN", grid_search.best_estimator_)

    def MNB(self):
        return self._evaluate("MNB", MultinomialNB())

    def MLR(self):
        # With the 'sag' solver and more than two classes scikit-learn fits a multinomial model.
        param_grid = {"C": [float(x) for x in np.linspace(start=0.6, stop=1, num=10)],
                      "solver": ["sag"],
                      "class_weight": ["balanced"]}
        best = self._search(LogisticRegression(random_state=RANDOM_STATE), param_grid, "logistic regression")
        return self._evaluate("MLR", best)

    def GBM(self):
        random_grid = {"n_estimators": [200, 800],
                       "max_features": ["sqrt"],
                       "max_depth": [10, 40, None],
                       "min_samples_split": [10, 30, 50],
                       "min_samples_leaf": [1, 2, 4],
                       "learning_rate": [.1, .5],
                       "subsample": [.5, 1.]}
        best = self._random_then_grid(GradientBoostingClassifier(random_state=RANDOM_STATE), random_grid,
                                      "gradient boosting")
        return self._evaluate("GBM", best)

    # ------------------------------------------------------------------ inspection
    def cor_unigrams_and_bigrams(self, top_unigrams=5, top_bigrams=2):
        """Print the unigrams and bigrams most correlated (chi-squared) with each label."""
        feature_names = np.array(self.tfidf.get_feature_names_out())
        for label, category_id in sorted(self.category_codes.items()):
            scores, _ = chi2(self.features_train, self.labels_train == category_id)
            ranked = feature_names[np.argsort(scores)]
            unigrams = [v for v in ranked if len(v.split(" ")) == 1]
            bigrams = [v for v in ranked if len(v.split(" ")) == 2]
            print(f"# '{label}' category:")
            print("  . Most correlated unigrams:\n. " + "\n. ".join(unigrams[-top_unigrams:]))
            print("  . Most correlated bigrams:\n. " + "\n. ".join(bigrams[-top_bigrams:]))
            print("")

    def plot_dim_red(self, model, features, labels, n_components=2):
        if model == "PCA":
            mod, title = PCA(n_components=n_components), "PCA decomposition"
        elif model == "TSNE":
            mod, title = TSNE(n_components=2), "t-SNE decomposition"
        else:
            raise ValueError("model must be 'PCA' or 'TSNE'")
        components = mod.fit_transform(features)
        df_full = pd.DataFrame(components[:, :2], columns=["PC1", "PC2"])
        df_full["label_name"] = [self.category_names[str(c)] for c in labels]
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x="PC1", y="PC2", hue="label_name", data=df_full, alpha=.7).set_title(title)
        plt.savefig(self.output_dir / f"dimension_reduction_{model}_{self.project}.png", bbox_inches="tight")
        plt.close()

    def dim_red(self):
        features = np.concatenate((self.features_train, self.features_test), axis=0)
        labels = np.concatenate((self.labels_train, self.labels_test), axis=0)
        self.plot_dim_red("PCA", features=features, labels=labels, n_components=2)
        self.plot_dim_red("TSNE", features=features, labels=labels, n_components=2)

    def mod_test(self):
        """Test-set rows the best model gets wrong."""
        df_test = self.all_data.loc[self.X_test.index, ["Name", "Label", "Category_Code"]].copy()
        df_test["Category_Predicted"] = [self.category_names[p] for p in self.best_mod.predict(self.features_test)]
        return df_test[df_test["Label"] != df_test["Category_Predicted"]]

    def mod_predict(self, text_to_classify):
        """Predict the label of a raw string or Document with the best model."""
        text = text_to_classify.text if isinstance(text_to_classify, Document) else text_to_classify
        if self.clean_texts:
            text = parse_words(text)
        features = self.tfidf.transform([text]).toarray()
        category = self.category_names[self.best_mod.predict(features)[0]]
        probability = self.best_mod.predict_proba(features)[0].max() * 100
        print(f"The predicted category using the {self.best_mod_key or 'given'} model is {category}.")
        print(f"The conditional probability is: {probability:.1f}%")
        return category
