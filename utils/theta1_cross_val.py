from sklearn.model_selection import KFold
import numpy as np
import copy


def theta2_from_theta1(theta1, big_M=1e6):
    # Enforce 1/theta1 + 1/theta2 = 1 safely
    if theta1 <= 1.0 + 1e-12:
        return big_M
    return theta1 / (theta1 - 1.0)  # same as 1/(1-1/theta1)


def theta1_cross_val(clf, theta1_grid, X, y, n_folds=5, metric="acc", seed=0):
    """
    Proper CV for theta1:
      - evaluates every theta1 on the same folds
      - picks best mean metric
      - refits a fresh clone on full (X,y) with chosen theta1

    metric: "acc" or "loss" (loss will be minimized)
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    theta1_grid = np.asarray(theta1_grid, dtype=float)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scores = []
    for theta1 in theta1_grid:
        theta2 = theta2_from_theta1(theta1)

        fold_scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # fresh model each fold (avoid state leakage)
            model = copy.deepcopy(clf)
            model.theta1 = float(theta1)
            model.theta2 = float(theta2)
            model.model_prepared = False
            model.verbose = False

            model.fit(X_tr, y_tr)

            if metric == "acc":
                fold_scores.append(model.score(X_val, y_val))
            elif metric == "loss":
                fold_scores.append(model.loss(X_val, y_val))
            else:
                raise ValueError("metric must be 'acc' or 'loss'.")

        scores.append(np.mean(fold_scores))

    scores = np.array(scores)

    if metric == "acc":
        best_idx = int(np.argmax(scores))
    else:
        best_idx = int(np.argmin(scores))

    best_theta1 = float(theta1_grid[best_idx])
    best_theta2 = float(theta2_from_theta1(best_theta1))

    # refit on full data with best theta
    best_model = copy.deepcopy(clf)
    best_model.theta1 = best_theta1
    best_model.theta2 = best_theta2
    best_model.model_prepared = False
    best_model.verbose = False
    best_model.fit(X, y)

    return best_model
