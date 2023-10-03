import numpy as np
import cvxpy as cp
import time
from collections import namedtuple
from sklearn.metrics import log_loss


def logSumExp(ns):
    max = np.max(ns)
    ds = ns - max
    sumOfExp = np.exp(ds)
    return max + np.log(sumOfExp)


class WASS_Robust_CLF:
    def __init__(self, fit_intercept=False, c_r=0, p=2, verbose=False):
        self.fit_intercept = fit_intercept
        self.c_r = c_r
        self.p = p
        self.verbose = verbose
        self.training_time = 0
        self.model_prepared = False

    def fit(self, X, y):
        if self.model_prepared:
            self.param_fit()
        else:
            self.prepare_model(X, y)
            self.param_fit()

    def prepare_model(self, X, y):
        N_train = X.shape[0]
        dim = X.shape[1]
        self.r = cp.Parameter(nonneg=True)
        self.beta = cp.Variable(dim)
        self.b = cp.Variable(1)
        if self.p == 1:
            q = "inf"
        elif self.p == 2:
            q = 2
        elif self.p == "inf":
            q = 1

        cons = [
            cp.norm2(self.beta) <= 1
        ]  # Bounded SVM constraint from the original problem

        self.obj = 1 / N_train * cp.sum(
            cp.pos(1 - cp.multiply(y.flatten(), self.beta.T @ X.T + self.b))
        ) + self.r * cp.norm(self.beta, q)
        self.problem = cp.Problem(cp.Minimize(self.obj), cons)
        self.model_prepared = True

    def param_fit(self):
        """Can be only called after prepare_model"""
        self.r.value = self.c_r
        self.problem.solve(
            solver=cp.MOSEK,
            mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6},
            verbose=self.verbose,
        )
        self.coef_ = self.beta.value
        self.intercept_ = self.b.value
        self.obj_val = self.obj.value

    def loss(self, X, y):
        # loss = self.loss_fcn(self.coef_, X, y)]
        loss = np.mean(
            cp.pos(
                1 - np.multiply(y.flatten(), self.coef_.T @ X.T + self.intercept_)
            ).value
        )
        return loss

    def predict(self, X):
        scores = self.coef_.T @ X.T + self.intercept_

        preds = scores.copy()
        preds[scores >= 0] = 1
        preds[scores < 0] = -1
        return preds

    def score(self, X, y):
        # calculate accuracy of the given test data set
        predictions = self.predict(X)
        acc = np.mean([predictions.flatten() == y.flatten()])
        return acc
