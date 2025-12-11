from sklearn.model_selection import KFold
import numpy as np


def theta1_cross_val(clf, params, X, y):
    kf = KFold(n_splits=params.shape[0], random_state=None, shuffle=True)
    i = 0
    performance = []
    performance2 = []
    clf_new = clf
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
        # print(theta2)
        clf_new.fit(X_train, y_train)

        score_ = clf_new.score(X_test, y_test)
        performance.append(score_)
        loss_ = clf_new.loss(X_test, y_test)
        performance2.append(loss_)
    # loc_best_params_score = np.argwhere((np.max(performance) == performance))
    # min_loss = np.min(np.array(performance2)[loc_best_params_score])
    # breakpoint()
    loc_best = np.argmax(performance)
    # loc_best = np.argwhere(performance2 == min_loss)[0][-1]
    theta1 = params[loc_best]
    clf_new.theta1 = theta1
    theta2 = 1 / (1 - 1 / theta1)
    clf_new.theta2 = theta2
    clf_new.model_prepared = False
    clf_new.fit(X, y)

    return clf_new
