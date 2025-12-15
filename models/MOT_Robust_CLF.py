import numpy as np
import cvxpy as cp
import time


class MOT_Robust_CLF:
    def __init__(self, fit_intercept=False, theta1=1.0, theta2=1.0, c_r=0.0, p=2, beta_constrained=False, beta_regularization=True, verbose=False):
        self.fit_intercept = fit_intercept
        self.theta1 = float(theta1)
        self.theta2 = float(theta2)
        self.c_r = float(c_r)
        self.p = p
        self.verbose = verbose
        self.beta_constrained = beta_constrained
        self.bebeta_regularization = beta_regularization

        self.model_prepared = False
        self.model_building_time = 0.0
        self.param_fit_time = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if not self.model_prepared:
            self.prepare_model(X, y)
        self.param_fit()

    def prepare_model(self, X, y):
        start_time = time.time()

        n, d = X.shape

        # parameters 
        self.r = cp.Parameter(nonneg=True, name="r")
        self.theta1_par = cp.Parameter(nonneg=True, name="theta1")
        self.theta2_par = cp.Parameter(nonneg=True, name="theta2")

        # p to q
        if self.p == 1:
            q = np.inf
        elif self.p == 2:
            q = 2
        elif self.p == "inf" or self.p == np.inf:
            q = 1
        else:
            raise ValueError("p must be 1, 2, or 'inf'.")

        # decision variables
        t = cp.Variable(name="t")                    # scalar
        epig = cp.Variable(n, name="epig")           # length-n vector
        self.lambda_ = cp.Variable(nonneg=True, name="lambda")
        self.beta = cp.Variable(d, name="beta")

        if self.fit_intercept:
            self.b = cp.Variable(name="b")           # scalar
        else:
            self.b = cp.Constant(0.0)

        eta = cp.Variable(n, nonneg=True, name="eta")  # length-n vector

        cons = []
        if self.beta_constrained:
            cons.append(
                cp.norm2(self.beta) <= 1
            )  

        # exponential cone constraints
        for i in range(n):
            cons += [cp.constraints.exponential.ExpCone(epig[i] - t,
                                                       self.lambda_ * self.theta2_par,
                                                       eta[i])]

            # empirical hinge epigraph
            cons += [cp.pos(1 - y[i] * (self.beta @ X[i, :] + self.b)) <= epig[i]]

        cons += [cp.norm(self.beta, q) <= self.lambda_ * self.theta1_par]
        # average eta constraint
        cons += [cp.sum(eta) / n <= self.lambda_ * self.theta2_par]

        # objective
        obj = cp.Minimize(self.r * self.lambda_ + t+ 1e-3 * cp.norm2(self.beta))

        self.problem = cp.Problem(obj, cons)
        self._t = t
        self._epig = epig
        self._eta = eta
        self._q = q

        self.model_prepared = True
        self.model_building_time = time.time() - start_time

    def param_fit(self):
        start_time = time.time()

        self.r.value = self.c_r
        self.theta1_par.value = self.theta1
        self.theta2_par.value = self.theta2

        self.problem.solve(
            solver=cp.MOSEK,
            mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6},
            verbose=self.verbose,
        )
        # print(self.lambda_.value)
        self.coef_ = np.array(self.beta.value).reshape(-1)
        self.intercept_ = float(self.b.value) if isinstance(self.b, cp.Variable) else 0.0
        self.obj_opt = float(self.problem.value)

        self.param_fit_time = time.time() - start_time

    def loss(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        scores = X @ self.coef_ + self.intercept_
        return float(np.mean(np.maximum(0.0, 1.0 - y * scores)))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = X @ self.coef_ + self.intercept_
        return np.where(scores >= 0.0, 1.0, -1.0)

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        preds = self.predict(X).reshape(-1)
        return float(np.mean(preds == y))
