import sklearn
from sklearn import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from utils import report_parser
from sklearn import metrics
from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier


max_iter = 50


def get_model_performance(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    res = metrics.classification_report(y_test, y_pred)
    return report_parser(res)


def model_training(X, Y, test_ratio, verbose_mode, model):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio)

    if model == "MLP":
        clf = MLPClassifier(
            hidden_layer_sizes=(1000, 1000),
            activation='relu',
            solver='adam',
            alpha=0.0002,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.01,
            power_t=0.5,
            max_iter=max_iter,
            shuffle=True,
            random_state=None,
            tol=0.0001,
            verbose=verbose_mode,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            n_iter_no_change=10).fit(X_train, y_train)

        return get_model_performance(clf, X_test, y_test)

    elif model == "NaiveBayes":
        clf = GaussianNB().fit(X_train, y_train)
        return get_model_performance(clf, X_test, y_test)

    elif model == "SVM":
        clf = svm.LinearSVC(random_state=0, tol=1e-5).fit(X_train, y_train)
        return get_model_performance(clf, X_test, y_test)

    elif model == "DT":
        clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
        return get_model_performance(clf, X_test, y_test)
    
    elif model == "KNN":
        clf = neighbors.KNeighborsClassifier().fit(X_train, y_train)
        return get_model_performance(clf, X_test, y_test)
    
    elif model == "RandomForest":
        clf = RandomForestClassifier().fit(X_train, y_train)
        return get_model_performance(clf, X_test, y_test)
    
    elif model == "Adaboost":
        clf = AdaBoostClassifier().fit(X_train, y_train)
        return get_model_performance(clf, X_test, y_test)
    
    elif model == "GradientBoost":
        clf = GradientBoostingClassifier().fit(X_train, y_train)
        return get_model_performance(clf, X_test, y_test)
    
    else:
        ret = dict()
        print("no available model")
        return ret
    