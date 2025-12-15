import numpy as np
import cvxpy as cp


def fit_hinge_erm_cvx(X, y, fit_intercept=True, beta_constrained=False, eps=1e-8):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    n, d = X.shape

    beta = cp.Variable(d)
    b = cp.Variable() if fit_intercept else 0.0
    if beta_constrained:
        cons= [cp.norm2(beta) <= 1]
    else:
        cons = []
    hinge = cp.pos(1 - cp.multiply(y, X @ beta + b))
    obj = cp.Minimize(cp.sum(hinge) / n + eps * cp.sum_squares(beta) + 1e-3 * cp.norm2(beta))  # eps picks a stable minimizer
    prob = cp.Problem(obj, constraints=cons)
    prob.solve(solver=cp.MOSEK,  mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8}, verbose=False)

    beta_val = np.array(beta.value).reshape(-1)
    b_val = float(b.value) if fit_intercept else 0.0
    return beta_val, b_val



