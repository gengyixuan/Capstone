from sklearn import *
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor



def get_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    res = metrics.regression.mean_squared_error(y_test, y_pred)
    return {"MeanSquaredError": res}


def model_training_regressor(X, Y, test_ratio, verbose_mode, name):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, shuffle=False)

    if name == "MLP":
        model = MLPRegressor(
            hidden_layer_sizes=(200, 50),
            activation='relu',
            solver='adam',
            alpha=0.0002,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.01,
            power_t=0.5,
            max_iter=10000,
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

        return get_model_performance(model, X_test, y_test)

    elif name == "NaiveBayes":
        model = linear_model.BayesianRidge().fit(X_train, y_train)
        return get_model_performance(model, X_test, y_test)

    elif name == "SVM":
        model = svm.LinearSVR(random_state=0, tol=1e-5).fit(X_train, y_train)
        return get_model_performance(model, X_test, y_test)

    elif name == "DT":
        model = tree.DecisionTreeRegressor().fit(X_train, y_train)
        return get_model_performance(model, X_test, y_test)

    elif name == "KNN":
        model = neighbors.KNeighborsRegressor().fit(X_train, y_train)
        return get_model_performance(model, X_test, y_test)

    elif name == "RandomForest":
        model = RandomForestRegressor().fit(X_train, y_train)
        return get_model_performance(model, X_test, y_test)

    elif name == "Adaboost":
        model = AdaBoostRegressor().fit(X_train, y_train)
        return get_model_performance(model, X_test, y_test)

    elif name == "GradientBoost":
        model = GradientBoostingRegressor().fit(X_train, y_train)
        return get_model_performance(model, X_test, y_test)

    else:
        ret = dict()
        print("no available model")
        return ret