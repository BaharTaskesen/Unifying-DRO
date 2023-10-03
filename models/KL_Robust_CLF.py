"""
KL-DR Logistic Regression
"""
import numpy as np
import cvxpy as cp
import time
from collections import namedtuple
from sklearn.metrics import log_loss


loss_fcn = lambda beta, X, y: cp.pos(1 - y * beta.T @ X)
"""DR-Bounded SVM model with KL divergence ambiguity set"""


class KL_Robust_CLF:
    def __init__(self, fit_intercept=False, c_r=0, verbose=False):
        self.fit_intercept = fit_intercept
        self.c_r = c_r
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
        dim = X.shape[1]
        N_train = X.shape[0]

        ## Define model
        # Parameters
        self.r = cp.Parameter(nonneg=True)

        # Variables
        self.beta = cp.Variable(dim)
        self.b = cp.Variable(1)
        eta = cp.Variable([N_train, 1])
        epig_ = cp.Variable([N_train, 1])
        t = cp.Variable(1)
        self.lambda_ = cp.Variable(1)

        cons = []

        cons.append(self.lambda_ >= cp.sum(eta))
        cons.append(self.lambda_ >= 0)
        cons.append(eta >= 0)
        cons.append(
            cp.norm2(self.beta) <= 1
        )  # Bounded SVM constraint from the original problem
        for i in range(N_train):
            cons.append(
                cp.constraints.exponential.ExpCone(epig_[i] - t, self.lambda_, eta[i])
            )
            cons.append(epig_[i] >= cp.pos(1 - y[i] * (self.beta.T @ X[i, :] + self.b)))

        self.obj = self.lambda_ * (self.r - np.log(N_train)) + t

        self.problem = cp.Problem(cp.Minimize(self.obj), cons)
        self.model_prepared = True
        stop_time = time.time()
        self.model_building_time = stop_time - start_time

    def param_fit(self):
        """Can be only called after prepare_model"""
        start_time = time.time()
        self.r.value = self.c_r
        self.problem.solve(
            solver=cp.MOSEK,
            mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6},
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
