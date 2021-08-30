#! Python3 -- text_classifier utilizes Document and Corpus objects to classify texts by category
import argparse
from text_extraction import Document, Corpus
import nltk
import numpy as np
import pandas as pd
import os
import logging
import sys
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
fh = logging.FileHandler('text_classifier.log', 'w+')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

punctuation_signs = list("?:!.,;")
nltk.download('stopwords')
stop_words = list(nltk.corpus.stopwords.words('english'))


def parse_text_column(df):
    df['Text'] = df.apply(parse_words, axis=1)
    return df


def parse_words(row):
    row['Text'] = clean_text(row)
    row['Text'] = lemmatize_text(row)
    row['Text'] = remove_stopwords(row)


def clean_text(row):
    row['Text'] = row['Text'].str.replace("\r", " ")
    row['Text'] = row['Text'].str.replace("\n", " ")
    row['Text'] = row['Text'].str.replace("    ", " ")
    row['Text'] = row['Text'].str.lower()
    for punct_sign in punctuation_signs:
        row['Text'] = row['Text'].str.replace(punct_sign, '')
    row['Text'] = row['Text'].str.replace("'s", "")
    return row['Text']


def lemmatize_text(row):
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    text = row['Text']

    # Create an empty list containing lemmatized words
    lemmatized_list = []

    # Save the text and its words into an object
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

    # Join the list
    row['Text'] = " ".join(lemmatized_list)
    return row['Text']


def remove_stopwords(row):
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        row['Text'] = row['Text'].str.replace(regex_stopword, '')
    return row['Text']


class TextClassifier:
    nltk.download('punkt')
    nltk.download('wordnet')
    logger.info("Initializing Classifier Object")

    def __init__(self, directory, method='tfidf', model='random_forest', parameters=None, plot_bool=False):
        logger.debug(f'Directory: {directory}, Method: {method}, Given Parameters: {parameters}')
        self.plot_bool = plot_bool
        self.models = {}
        self.model_comparison = pd.DataFrame(columns=['Model', 'Training Set Accuracy', 'Test Set Accuracy'])
        if parameters is None:
            parameters = [(None, None), None, None, None]
        self.all_data = pd.DataFrame(columns=['Name', 'Text', 'Label'])
        self.corpora = []
        if directory:
            self.project = os.path.basename(os.path.dirname(directory))
            subdirs = [x[0] for x in os.walk(directory)]
            for dir in subdirs:
                cur_corp = Corpus(dir)
                self.corpora.append(cur_corp)
                self.all_data.append(cur_corp.df, ignore_index=True)

        self.method = method
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.all_data['Text'],
                                                                                self.all_data['Category_code'],
                                                                                test_size=0.33,
                                                                                random_state=8)
        if method == 'tfidf':
            self.clean_texts()
            self.lemmatize_texts()
            self.remove_stopwords()
            self._encode_labels()
            if parameters:
                ngram_range = parameters[0]
                min_df = parameters[1]
                max_df = parameters[2]
                max_features = parameters[3]
            else:
                ngram_range = (1, 2)
                min_df = 10
                max_df = 1
                max_features = 300

            self.tfidf = TfidfVectorizer(encoding='utf-8',
                                         ngram_range=ngram_range,
                                         stop_words=None,
                                         lowercase=False,
                                         max_df=max_df,
                                         min_df=min_df,
                                         max_features=max_features,
                                         norm='l2',
                                         sublinear_tf=True)

            self.features_train = self.tfidf.fit_transform(self.X_train).toarray()
            self.labels_train = self.y_train

            self.features_test = self.tfidf.transform(self.X_test).toarray()
            self.labels_test = self.y_test

        if model == 'random_forest' or model == 'best':
            model, comp = self.random_forest()
            self.models['random_forest'] = model
            self.model_comparison.append(comp, ignore_index=True)

        if model == 'SVM' or model == 'best':
            model, comp = self.SVM()
            self.models['SVM'] = model
            self.model_comparison.append(comp, ignore_index=True)

        if model == 'KNN' or model == 'best':
            model, comp = self.KNN()
            self.models['KNN'] = model
            self.model_comparison.append(comp, ignore_index=True)

        if model == 'MNB' or model == 'best':
            model, comp = self.MNB()
            self.models['MNB'] = model
            self.model_comparison.append(comp, ignore_index=True)

        if model == 'MLR' or model == 'best':
            model, comp = self.MLR()
            self.models['MLR'] = model
            self.model_comparison.append(comp, ignore_index=True)

        if model == 'GBM' or model == 'best':
            model, comp = self.GBM()
            self.models['GBM'] = model
            self.model_comparison.append(comp, ignore_index=True)

        df_summary = self.model_comparison.reset_index().drop('index', axis=1)
        print(df_summary.sort_values('Test Set Accuracy', ascending=False))

        if model == 'best':
            not_overfitted = df_summary[df_summary['Training Set Accuracy'] <= .98]
            best_Test = not_overfitted[not_overfitted['Test Set Accuracy'] == not_overfitted['Test Set Accuracy'].max()]
            self.best_mod_key = str(best_Test['Model'])
        else:
            self.best_mod_key = str(df_summary['Model'])
        self.best_mod = self.models[self.best_mod_key]

    # function that encodes each of the labels to an integer
    def _encode_labels(self):
        self.categories = set(self.all_data['Label'])
        cat_range = [*range(0, len(self.categories))]
        cat_range = [str(i) for i in cat_range]
        self.category_codes = dict(zip(self.categories, cat_range))
        self.all_data['Category_Code'] = self.all_data['Label']
        self.all_data = self.all_data.replace({'Category_Code': self.category_codes})

    def _parse_text(self):
        self.all_data = parse_text_column(self.all_data)

    # function that cleans strings in 'Text' column
    def clean_texts(self):
        self.all_data['Text'] = self.all_data['Text'].str.replace("\r", " ")
        self.all_data['Text'] = self.all_data['Text'].str.replace("\n", " ")
        self.all_data['Text'] = self.all_data['Text'].str.replace("    ", " ")
        self.all_data['Text'] = self.all_data['Text'].str.lower()

        self.all_data['Text'] = self.all_data['Text']
        for punct_sign in self.punctuation_signs:
            self.all_data['Text'] = self.all_data['Text'].str.replace(punct_sign, '')
        self.all_data['Text'] = self.all_data['Text'].str.replace("'s", "")

    # lemmatize texts (flattens words to their established root words)
    def lemmatize_texts(self):
        wordnet_lemmatizer = nltk.WordNetLemmatizer()
        nrows = len(self.all_data)
        lemmatized_text_list = []

        for row in range(0, nrows):

            # Create an empty list containing lemmatized words
            lemmatized_list = []

            # Save the text and its words into an object
            text = self.all_data.loc[row]['Text']
            text_words = text.split(" ")

            # Iterate through every word to lemmatize
            for word in text_words:
                lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

            # Join the list
            lemmatized_text = " ".join(lemmatized_list)

            # Append to the list containing the texts
            lemmatized_text_list.append(lemmatized_text)
        self.all_data['Text'] = lemmatized_text_list

    # remove stopwords from the 'text' column
    def remove_stopwords(self):
        nltk.download('stopwords')
        stop_words = list(nltk.corpus.stopwords.words('english'))
        for stop_word in stop_words:
            regex_stopword = r"\b" + stop_word + r"\b"
            self.all_data['Text'] = self.all_data['Text'].str.replace(regex_stopword, '')

    # show what words and bigrams are most correlated to which labels
    def cor_unigrams_and_bigrams(self):
        for label, category_id in sorted(self.category_codes.items()):
            features_chi2 = chi2(self.features_train, self.labels_train == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(self.tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}' category:".format(label))
            print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
            print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
            print("")

    # produce random forest model
    def random_forest(self):
        logger.info("Beginning Random Forest Modeling")
        # # initialize random forest classifier
        # rf_0 = RandomForestClassifier(random_state=8)
        # logger.debug(f"Current parameters: \n {rf_0.get_params()}")

        # n_estimators
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]

        # max_features
        max_features = ['auto', 'sqrt']

        # max_depth
        max_depth = [int(x) for x in np.linspace(20, 100, num=5)]
        max_depth.append(None)

        # min_samples_split
        min_samples_split = [2, 5, 10]

        # min_samples_leaf
        min_samples_leaf = [1, 2, 4]

        # bootstrap
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        logger.debug(f"random grid \n {random_grid}")

        # contruct base model
        rfc = RandomForestClassifier(random_state=8)

        # Definition of the random search
        random_search = RandomizedSearchCV(estimator=rfc,
                                           param_distributions=random_grid,
                                           n_iter=50,
                                           scoring='accuracy',
                                           cv=3,
                                           verbose=1,
                                           random_state=8)

        # Fit the random search model
        random_search.fit(self.features_train, self.labels_train)

        print("The best hyperparameters from Random Search are:")
        print(random_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(random_search.best_score_)

        # Create the parameter grid based on the results of random search
        bootstrap = [False]
        max_depth = [30, 40, 50]
        max_features = ['sqrt']
        min_samples_leaf = [1, 2, 4]
        min_samples_split = [5, 10, 15]
        n_estimators = [800]

        param_grid = {
            'bootstrap': bootstrap,
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'n_estimators': n_estimators
        }

        # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=rfc,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=1)

        # Fit the grid search to the data
        grid_search.fit(self.features_train, self.labels_train)

        print("The best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        best_rfc = grid_search.best_estimator_
        print(best_rfc)
        best_rfc.fit(self.features_train, self.labels_train)

        rfc_pred = best_rfc.predict(self.features_test)

        # Training accuracy
        print("The training accuracy is: ")
        print(accuracy_score(self.labels_train, best_rfc.predict(self.features_train)))

        # Test accuracy
        print("The test accuracy is: ")
        print(accuracy_score(self.labels_test, rfc_pred))

        # Classification report
        print("Classification report")
        print(classification_report(self.labels_test, rfc_pred))

        if self.plot_bool:
            aux_df = self.all_data[['Label', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
            conf_matrix = confusion_matrix(self.labels_test, rfc_pred)
            plt.figure(figsize=(12.8, 6))
            sns.heatmap(conf_matrix,
                        annot=True,
                        xticklabels=aux_df['Label'].values,
                        yticklabels=aux_df['Label'].values,
                        cmap="Blues")
            plt.ylabel('Predicted')
            plt.xlabel('Actual')
            plt.title('Confusion matrix')
            plt.savefig(f'rf_heatmap_{self.project}.png')
            plt.show()

        d = {
            'Model': 'Random_forest',
            'Training Set Accuracy': accuracy_score(self.labels_train, best_rfc.predict(self.features_train)),
            'Test Set Accuracy': accuracy_score(self.labels_test, rfc_pred)
        }

        df_models_rfc = pd.DataFrame(d, index=[0])
        return best_rfc, df_models_rfc

    # produce support vector machine model
    def SVM(self):
        logger.info("Initializing Support Vector Machine modeling")
        # # hyperparameter tuning with randomized search cross validation
        #
        # # C
        # C = [.0001, .001, .01]
        #
        # # gamma
        # gamma = [.0001, .001, .01, .1, 1, 10, 100]
        #
        # # degree
        # degree = [1, 2, 3, 4, 5]
        #
        # # kernel
        # kernel = ['linear', 'rbf', 'poly']
        #
        # # probability
        # probability = [True]
        #
        # # Create the random grid
        # random_grid = {'C': C,
        #                'kernel': kernel,
        #                'gamma': gamma,
        #                'degree': degree,
        #                'probability': probability
        #                }
        #
        # svc = svm.SVC(random_state=8)
        #
        # # Definition of the random search
        # random_search = RandomizedSearchCV(estimator=svc,
        #                                    param_distributions=random_grid,
        #                                    n_iter=50,
        #                                    scoring='accuracy',
        #                                    cv=3,
        #                                    verbose=1,
        #                                    random_state=8)
        #
        # # Fit the random search model
        # random_search.fit(self.features_train, self.labels_train)
        #
        # print("The best hyperparameters from Random Search are:")
        # print(random_search.best_params_)
        # print("")
        # print("The mean accuracy of a model with these hyperparameters is:")
        # print(random_search.best_score_)

        # svm hyperparameter tuning with grid search cross validation

        # Create the parameter grid based on the results of random search
        C = [.0001, .001, .01, .1]
        degree = [3, 4, 5]
        gamma = [1, 10, 100]
        probability = [True]

        param_grid = [
            {'C': C, 'kernel': ['linear'], 'probability': probability},
            {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
            {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
        ]

        # Create a base model
        svc = svm.SVC(random_state=8)

        # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=svc,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=1)

        # Fit the grid search to the data
        grid_search.fit(self.features_train, self.labels_train)

        print("The best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        best_svc = grid_search.best_estimator_
        best_svc.fit(self.features_train, self.labels_train)
        svc_pred = best_svc.predict(self.features_test)

        # Training accuracy
        print("The training accuracy is: ")
        print(accuracy_score(self.labels_train, best_svc.predict(self.features_train)))

        # Test accuracy
        print("The test accuracy is: ")
        print(accuracy_score(self.labels_test, svc_pred))

        # Classification report
        print("Classification report")
        print(classification_report(self.labels_test, svc_pred))

        if self.plot_bool:
            aux_df = self.all_data[['Label', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
            conf_matrix = confusion_matrix(self.labels_test, svc_pred)
            plt.figure(figsize=(12.8, 6))
            sns.heatmap(conf_matrix,
                        annot=True,
                        xticklabels=aux_df['Label'].values,
                        yticklabels=aux_df['Label'].values,
                        cmap="Blues")
            plt.ylabel('Predicted')
            plt.xlabel('Actual')
            plt.title('Confusion matrix')
            plt.savefig(f'svm_heatmap_{self.project}')
            plt.show()

        d = {
            'Model': 'SVM',
            'Training Set Accuracy': accuracy_score(self.labels_train, best_svc.predict(self.features_train)),
            'Test Set Accuracy': accuracy_score(self.labels_test, svc_pred)
        }

        df_models_svc = pd.DataFrame(d, index=[0])
        return best_svc, df_models_svc

    # produce KNN model
    def KNN(self):
        n_neighbors = [int(x) for x in np.linspace(start=1, stop=500, num=100)]

        param_grid = {'n_neighbors': n_neighbors}

        # Create a base model
        knnc = KNeighborsClassifier()

        # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=knnc,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=1)

        # Fit the grid search to the data
        grid_search.fit(self.features_train, self.labels_train)

        print("The best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        best_knnc = grid_search.best_estimator_

        best_knnc.fit(self.features_train, self.labels_train)
        knnc_pred = best_knnc.predict(self.features_test)

        # Training accuracy
        print("The training accuracy is: ")
        print(accuracy_score(self.labels_train, best_knnc.predict(self.features_train)))

        # Test accuracy
        print("The test accuracy is: ")
        print(accuracy_score(self.labels_test, knnc_pred))

        # Classification report
        print("Classification report")
        print(classification_report(self.labels_test, knnc_pred))
        if self.plot_bool:
            aux_df = self.all_data[['Label', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
            conf_matrix = confusion_matrix(self.labels_test, knnc_pred)
            plt.figure(figsize=(12.8, 6))
            sns.heatmap(conf_matrix,
                        annot=True,
                        xticklabels=aux_df['Label'].values,
                        yticklabels=aux_df['Label'].values,
                        cmap="Blues")
            plt.ylabel('Predicted')
            plt.xlabel('Actual')
            plt.title('Confusion matrix')
            plt.savefig(f'knn_heatmap_{self.project}.png')
            plt.show()

        d = {
            'Model': 'KNN',
            'Training Set Accuracy': accuracy_score(self.labels_train, best_knnc.predict(self.features_train)),
            'Test Set Accuracy': accuracy_score(self.labels_test, knnc_pred)
        }

        df_models_knnc = pd.DataFrame(d, index=[0])
        return best_knnc, df_models_knnc

    # produce multinomial naive bayes model
    def MNB(self):
        mnbc = MultinomialNB
        mnbc.fit(self.features_train, self.labels_train)
        mnbc_pred = mnbc.predict(self.features_test)

        # Training accuracy
        print("The training accuracy is: ")
        print(accuracy_score(self.labels_train, mnbc.predict(self.features_train)))

        # Test accuracy
        print("The test accuracy is: ")
        print(accuracy_score(self.labels_test, mnbc_pred))

        # Classification report
        print("Classification report")
        print(classification_report(self.labels_test, mnbc_pred))

        if self.plot_bool:
            aux_df = self.all_data[['Label', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
            conf_matrix = confusion_matrix(self.labels_test, mnbc_pred)
            plt.figure(figsize=(12.8, 6))
            sns.heatmap(conf_matrix,
                        annot=True,
                        xticklabels=aux_df['Label'].values,
                        yticklabels=aux_df['Label'].values,
                        cmap="Blues")
            plt.ylabel('Predicted')
            plt.xlabel('Actual')
            plt.title('Confusion matrix')
            plt.savefig(f'mnb_heatmap_{self.project}.png')
            plt.show()

        d = {
            'Model': 'MNB',
            'Training Set Accuracy': accuracy_score(self.labels_train, mnbc.predict(self.features_train)),
            'Test Set Accuracy': accuracy_score(self.labels_test, mnbc_pred)
        }

        df_models_mnbc = pd.DataFrame(d, index=[0])
        return mnbc, df_models_mnbc

    # produce multinomial logistic regression model
    def MLR(self):
        C = [float(x) for x in np.linspace(start=0.6, stop=1, num=10)]
        multi_class = ['multinomial']
        solver = ['sag']
        class_weight = ['balanced']
        penalty = ['l2']

        param_grid = {'C': C,
                      'multi_class': multi_class,
                      'solver': solver,
                      'class_weight': class_weight,
                      'penalty': penalty}

        # Create a base model
        lrc = LogisticRegression(random_state=8)

        # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=lrc,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=1)

        # Fit the grid search to the data
        grid_search.fit(self.features_train, self.labels_train)

        print("The best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        best_lrc = grid_search.best_estimator_

        best_lrc.fit(self.features_train, self.labels_train)
        lrc_pred = best_lrc.predict(self.features_test)

        # Training accuracy
        print("The training accuracy is: ")
        print(accuracy_score(self.labels_train, best_lrc.predict(self.features_train)))

        # Test accuracy
        print("The test accuracy is: ")
        print(accuracy_score(self.labels_test, lrc_pred))

        # Classification report
        print("Classification report")
        print(classification_report(self.labels_test, lrc_pred))

        if self.plot_bool:
            aux_df = self.all_data[['Label', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
            conf_matrix = confusion_matrix(self.labels_test, lrc_pred)
            plt.figure(figsize=(12.8, 6))
            sns.heatmap(conf_matrix,
                        annot=True,
                        xticklabels=aux_df['Label'].values,
                        yticklabels=aux_df['Label'].values,
                        cmap="Blues")
            plt.ylabel('Predicted')
            plt.xlabel('Actual')
            plt.title('Confusion matrix')
            plt.savefig(f'MLR_heatmap_{self.project}.png')
            plt.show()

        d = {
            'Model': 'MLR',
            'Training Set Accuracy': accuracy_score(self.labels_train, best_lrc.predict(self.features_train)),
            'Test Set Accuracy': accuracy_score(self.labels_test, lrc_pred)
        }

        df_models_lrc = pd.DataFrame(d, index=[0])
        return best_lrc, df_models_lrc

    # produce gradient boosting machine model
    def GBM(self):
        from sklearn.ensemble import GradientBoostingClassifier
        # Create the parameter grid based on the results of random search
        max_depth = [5, 10, 15]
        max_features = ['sqrt']
        min_samples_leaf = [2]
        min_samples_split = [50, 100]
        n_estimators = [800]
        learning_rate = [.1, .5]
        subsample = [1.]

        param_grid = {
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'subsample': subsample

        }

        # Create a base model
        gbc = GradientBoostingClassifier(random_state=8)

        # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=gbc,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=1)

        # Fit the grid search to the data
        grid_search.fit(self.features_train, self.labels_train)

        print("The best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        best_gbc = grid_search.best_estimator_

        best_gbc.fit(self.features_train, self.labels_train)
        gbc_pred = best_gbc.predict(self.features_test)

        # Training accuracy
        print("The training accuracy is: ")
        print(accuracy_score(self.labels_train, best_gbc.predict(self.features_train)))

        # Test accuracy
        print("The test accuracy is: ")
        print(accuracy_score(self.labels_test, gbc_pred))

        # Classification report
        print("Classification report")
        print(classification_report(self.labels_test, gbc_pred))
        if self.plot_bool:
            aux_df = self.all_data[['Label', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
            conf_matrix = confusion_matrix(self.labels_test, gbc_pred)
            plt.figure(figsize=(12.8, 6))
            sns.heatmap(conf_matrix,
                        annot=True,
                        xticklabels=aux_df['Label'].values,
                        yticklabels=aux_df['Label'].values,
                        cmap="Blues")
            plt.ylabel('Predicted')
            plt.xlabel('Actual')
            plt.title('Confusion matrix')
            plt.savefig(f'GBM_heatmap_{self.project}.png')
            plt.show()

        d = {
            'Model': 'GBM',
            'Training Set Accuracy': accuracy_score(self.labels_train, best_gbc.predict(self.features_train)),
            'Test Set Accuracy': accuracy_score(self.labels_test, gbc_pred)
        }

        df_models_gbc = pd.DataFrame(d, index=[0])

        return best_gbc, df_models_gbc

    # dimensionality reduction
    def plot_dim_red(self, model, features, labels, n_components=2):
        # Creation of the model
        if (model == 'PCA'):
            mod = PCA(n_components=n_components)
            title = "PCA decomposition"  # for the plot

        elif (model == 'TSNE'):
            mod = TSNE(n_components=2)
            title = "t-SNE decomposition"

        else:
            return "Error"

        # Fit and transform the features
        principal_components = mod.fit_transform(features)

        # Put them into a dataframe
        df_features = pd.DataFrame(data=principal_components,
                                   columns=['PC1', 'PC2'])

        # Now we have to paste each row's label and its meaning
        # Convert labels array to df
        df_labels = pd.DataFrame(data=labels,
                                 columns=['label'])

        df_full = pd.concat([df_features, df_labels], axis=1)
        df_full['label'] = df_full['label'].astype(str)

        # Get labels name
        category_names = self.categories

        # And map labels
        df_full['label_name'] = df_full['label']
        df_full = df_full.replace({'label_name': category_names})

        # Plot
        plt.figure(figsize=(10, 10))
        sns_plot = sns.scatterplot(x='PC1',
                        y='PC2',
                        hue="label_name",
                        data=df_full,
                        palette=["red", "pink", "royalblue", "greenyellow", "lightseagreen"],
                        alpha=.7).set_title(title)
        sns_plot.savefig(f"dimension_reduction_{self.project}.png")

    def dim_red(self):
        features = np.concatenate((self.features_train, self.features_test), axis=0)
        labels = np.concatenate((self.labels_train, self.labels_test), axis=0)
        self.plot_dim_red("PCA", features=features, labels=labels, n_components=2)
        self.plot_dim_red("TSNE", features=features, labels=labels, n_components=2)

    def mod_test(self, ):
        predictions = self.best_mod.predict(self.features_test)
        # Indexes of the test set
        index_X_test = self.X_test.index

        # We get them from the original df
        df_test = self.all_data.loc[index_X_test]

        # Add the predictions
        df_test['Prediction'] = predictions

        # Clean columns
        df_test = df_test[['Text', 'Label', 'Category_Code', 'Prediction']]

        # Decode
        df_test['Category_Predicted'] = df_test['Prediction']
        df_test = df_test.replace({'Category_Predicted': self.categories})

        # Clean columns again
        df_test = df_test[['Content', 'Category', 'Category_Predicted']]

        condition = (df_test['Label'] != df_test['Category_Predicted'])

        df_misclassified = df_test[condition]

    def mod_predict(self, text_to_classify):
        def create_features_from_text(text_object):
            df = pd.DataFrame(columns=['Text'])
            if type(text_object) == str:
                df.loc[0] = text_to_classify
            elif type(text_object) == Document:
                df.loc[0] = text_object.text

            df = parse_text_column(df)

            # TF-IDF
            features = self.tfidf.transform(df).toarray()
            return features

        def get_category_name(category_id):
            for category, id in self.category_codes.items():
                if id == category_id:
                    return category

        def predict_from_text(text):
            # Predict using the input model
            prediction_best = self.best_mod.predict(create_features_from_text(text))[0]
            prediction_best_proba = self.best_mod.predict_proba(create_features_from_text(text))[0]

            # Return result
            category_best = get_category_name(prediction_best)

            print(f"The predicted category using the {self.best_mod_key} model is {category_best}.")
            print("The conditional probability is: %a" % (prediction_best_proba.max() * 100))
            return category_best
        prediction = predict_from_text(text_to_classify)

def main():
    if len(sys.argv) == 1:  # no arguments, so print help message
        print("no commands passed")
        return

    parser = argparse.ArgumentParser(description="classify texts")  # initialize argument parser
    parser.add_argument("command", type=str, help="takes command from the following: 'print', 'length_by_author', "
                                                  "'unique_by_author', 'ratio_by_author', 'plot_unique_by_length',"
                                                  "'frequency_distribution', word_commonality'")
    # add positional 'command' argument to parser
    parser.add_argument("-d", "--directory", type=str, help="<directory where sub-directories are stored>>")  # add optional directoryargument to parser
    parser.add_argument("-o", "--ofile", type=str, help="<outfile>")  # add optional 'outfile name' argument to parser
    parser.add_argument("-p", "--plot", action='store_true',
                        help="plots output")  # add optional 'plot' boolean argument to parser
    parser.add_argument("-l", "--language", type=str, help="<language of stories>")  # add optional 'language' arg
    args = parser.parse_args()  # create list of arguments passed to parser

    if args.ofile:  # check for explicit outfile
        out_name = args.ofile
    else:
        out_name = 'storydata_' + args.command + '.csv'
