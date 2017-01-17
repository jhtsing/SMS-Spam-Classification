from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob

import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv('SMSSpamCollection', sep='\t', header=-1)
df.columns = ['class','text']

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2)
#print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]


def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words

if __name__ == "__main__":

	msgs = pd.read_csv('SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,names=["label", "message"])

	m_trn, m_tst, l_trn, l_tst = \
		train_test_split(msgs['message'], msgs['label'], test_size=0.2)

	classifiers = [
    	KNeighborsClassifier(3),
    	SVC(kernel="linear", C=0.025),
    	SVC(gamma=2, C=1),
    	DecisionTreeClassifier(max_depth=5),
    	RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    	MultinomialNB()]
	names_classifier = ['KNN','Linear SVM','RBF SVM', 'Decision Tree', 'RandomForestClassifier' , 'Naive Bayes']

	print "Complete Data Set:"
	for i,classifier in enumerate(classifiers):
		pipeline=Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', classifier)])
		scores=cross_val_score(pipeline, df['text'], df['class'], cv=10, scoring='accuracy', n_jobs=-1)
		#print names_classifier[i], sum(scores)/float(len(scores))
		print names_classifier[i], scores.mean(), scores.std()

	plot_learning_curve(pipeline, "accuracy vs. training set size", m_trn, l_trn, cv=5)
	plt.show()

	print "Training Data and Validation Data:"
	for i,classifier in enumerate(classifiers):
		pipeline=Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', classifier)])
		scores=cross_val_score(pipeline, m_trn, l_trn, cv=10, scoring='accuracy', n_jobs=-1)
		#print names_classifier[i], sum(scores)/float(len(scores))
		print names_classifier[i], scores.mean(), scores.std()

	plot_learning_curve(pipeline, "accuracy vs. training set size", m_trn, l_trn, cv=5)
	plt.show()


	# Applying few models on testing data.
	pipeline = Pipeline([('bow', CountVectorizer(analyzer=split_into_lemmas)), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])
	params = {
		'tfidf__use_idf': (True, False),
		'bow__analyzer': (split_into_lemmas, split_into_tokens),
	}

	grid = GridSearchCV(
		pipeline,  # pipeline from above
		params, # parameters to tune via cross validation
		refit=True,  # fit using all available data at the end, on the best found param combination
		n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
		scoring='accuracy',  # what score are we optimizing?
		cv=StratifiedKFold(l_trn, n_folds=5),  # what type of cross validation to use
	)

	nb_detector = grid.fit(m_trn, l_trn)
	predictions = nb_detector.predict(m_tst)
	print "Naive Bayes:"
	print 'accuracy', accuracy_score(l_tst, predictions)
	print confusion_matrix(l_tst, predictions, labels = ['spam','ham'])
	print classification_report(l_tst, predictions)

	plot_learning_curve(pipeline, "accuracy vs. Size of the Tesing set", m_tst, l_tst, cv=5)
	plt.show()


	#SVM
	pipeline_svm = Pipeline([
		('bow', CountVectorizer(analyzer=split_into_lemmas)),
		('tfidf', TfidfTransformer()),
		('classifier', SVC()),  # <== change here
	])

	param_svm = [
		{'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
		{'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
	]

	grid_svm = GridSearchCV(
		pipeline_svm,  # pipeline from above
		param_grid=param_svm,  # parameters to tune via cross validation
		refit=True,  # fit using all data, on the best detected classifier
		n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
		scoring='accuracy',  # what score are we optimizing?
		cv=StratifiedKFold(l_trn, n_folds=5),  # what type of cross validation to use
	)

	svm_detector = grid_svm.fit(m_trn, l_trn)
	print "SVM:"
	print 'accuracy', accuracy_score(l_tst, svm_detector.predict(m_tst))
	print confusion_matrix(l_tst, svm_detector.predict(m_tst), labels = ['spam','ham'])
	print classification_report(l_tst, svm_detector.predict(m_tst))

	plot_learning_curve(pipeline, "accuracy vs. Size of the Tesing set", m_tst, l_tst, cv=5)
	plt.show()