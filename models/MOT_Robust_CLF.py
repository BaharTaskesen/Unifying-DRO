"""
DR Logistic Regression
"""
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


class MOT_Robust_CLF:
    def __init__(
        self, fit_intercept=False, theta1=1, theta2=1, c_r=0, p=2, verbose=False
    ):
        self.fit_intercept = fit_intercept
        self.theta1 = theta1
        self.theta2 = theta2
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
        start_time = time.time()
        N_train = X.shape[0]
        dim = X.shape[1]
        self.r = cp.Parameter(nonneg=True)
        self.theta1_par = cp.Parameter(nonneg=True)
        self.theta2_par = cp.Parameter(nonneg=True)

        if self.p == 1:
            q = "inf"
        elif self.p == 2:
            q = 2
        elif self.p == "inf":
            q = 1

        # Decision variables
        t = cp.Variable(1)
        epig_ = cp.Variable([N_train, 1])
        self.lambda_ = cp.Variable(1)
        self.beta = cp.Variable(dim)
        self.b = cp.Variable(1)
        eta = cp.Variable([N_train, 1])

        # Constraints
        cons = []
        cons.append(
            cp.norm2(self.beta) <= 1
        )  # Bounded SVM constraint from the original problem
        cons.append(eta >= 0)
        cons.append(self.lambda_ >= 0)
        for i in range(N_train):
            cons.append(
                cp.constraints.exponential.ExpCone(
                    epig_[i] - t, self.theta2 * self.lambda_, eta[i]
                )
            )
            cons.append(cp.pos(1 - y[i] * (self.beta.T @ X[i, :] + self.b)) <= epig_[i])
        cons.append(self.lambda_ * self.theta1 >= cp.norm(self.beta, q))
        cons.append(N_train * self.lambda_ * self.theta2 >= cp.sum(eta))
        self.obj = self.r * self.lambda_ + t
        self.problem = cp.Problem(cp.Minimize(self.obj), cons)
        self.model_prepared = True
        stop_time = time.time()
        self.model_building_time = stop_time - start_time

    def param_fit(self):
        """Can be only called after prepare_model"""
        start_time = time.time()
        self.r.value = self.c_r
        self.theta1_par.value = self.theta1
        self.theta2_par.value = self.theta2
        self.problem.solve(
            solver=cp.MOSEK,
            mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8},
            verbose=self.verbose,
        )
        self.coef_ = self.beta.value
        self.intercept_ = self.b.value
        self.obj_opt = self.obj.value
        stop_time = time.time()
        self.param_fit_time = stop_time - start_time

    def loss(self, X, y):
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
