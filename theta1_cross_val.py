from sklearn.model_selection import KFold
import numpy as np


def theta1_cross_val(clf, params, X, y):
    kf = KFold(n_splits=params.shape[0], random_state=None, shuffle=True)
    i = 0
    performance = []
    clf_new = clf
    # breakpoin()
    clf_new.model_prepared = False
    clf_new.theta2 = 1

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        clf_new.verbose = False
        X_train = X[train_index, :]
        y_train = y[train_index]

        X_test = X[test_index, :]
        y_test = y[test_index]
        theta1 = params[i]
        clf_new.theta1 = theta1
        theta2 = 1 / (1 - 1 / theta1)

        clf_new.theta2 = theta2
        clf_new.fit(X_train, y_train)

        score_ = clf_new.score(X_test, y_test)
        performance.append(score_)

    loc_best_param = np.argmax(performance)
    theta1 = params[loc_best_param]
    clf_new.theta1 = theta1

    if theta1 == 1:
        theta2 = max(params)
    elif theta1 == max(params):
        theta2 = 1
    else:
        theta2 = 1 / (1 - 1 / theta1)
    clf_new.theta2 = theta2
    clf_new.model_prepared = False
    clf_new.fit(X, y)

    return clf_new
