import pickle
import os
from sklearn import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from utils import report_parser

max_iter = 50


def get_model_performance(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    res = metrics.classification_report(y_test, y_pred)
    return report_parser(res)


def model_training(X, Y, test_ratio, verbose_mode, model, model_snapshot, data_type):
    model_snapshot_fname = "models/model_ss_" + model + "_" + data_type + '.pkl'

    try:
        os.mkdir("models")
    except:
        print()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio)

    if model_snapshot == 'load':
        infile = open(model_snapshot_fname, 'rb')
        clf = pickle.load(infile)
        infile.close()
        return get_model_performance(clf, X_test, y_test)

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

    elif model == "NaiveBayes":
        clf = GaussianNB().fit(X_train, y_train)
        
    elif model == "SVM":
        clf = svm.LinearSVC(random_state=0, tol=1e-5).fit(X_train, y_train)
        
    elif model == "DT":
        clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
        
    elif model == "KNN":
        clf = neighbors.KNeighborsClassifier().fit(X_train, y_train)
        
    elif model == "RandomForest":
        clf = RandomForestClassifier().fit(X_train, y_train)
        
    elif model == "Adaboost":
        clf = AdaBoostClassifier().fit(X_train, y_train)
        
    elif model == "GradientBoost":
        clf = GradientBoostingClassifier().fit(X_train, y_train)
        
    else:
        ret = dict()
        print("no available model")
        return ret

    if model_snapshot == 'save':
        outfile = open(model_snapshot_fname, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()
    
    return get_model_performance(clf, X_test, y_test)
    