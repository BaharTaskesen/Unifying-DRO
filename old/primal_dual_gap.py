import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import matplotlib as mpl

from utils.data_utils import prepare_data
from models.EMP_CLF import fit_hinge_erm_cvx
def hinge_losses(beta, b, X, y):
    margins = 1.0 - y * (X @ beta + b)
    return np.maximum(margins, 0.0)


SEED = 0
np.random.seed(SEED)
# def plot_boundary_and_margins(ax, beta, b, x_range, lw0=2.0, lwm=1.2, alpha_m=0.6):
#     # decision boundary: beta^T x + b = 0
#     y0 = -(beta[0] / beta[1]) * x_range - b / beta[1]
#     # margins: beta^T x + b = +1 and = -1
#     y_plus1  = -(beta[0] / beta[1]) * x_range - (b - 1.0) / beta[1]  # = +1
#     y_minus1 = -(beta[0] / beta[1]) * x_range - (b + 1.0) / beta[1]  # = -1

#     ax.plot(x_range, y0, linewidth=lw0, color="gray")
#     ax.plot(x_range, y_plus1,  linestyle="--", linewidth=lwm, color="gray", alpha=alpha_m)
#     ax.plot(x_range, y_minus1, linestyle="--", linewidth=lwm, color="gray", alpha=alpha_m)


# problem setup
d = 2
N = 10
noise_mag = 0.1
label_noise = 0.0
beta_constrained = False



params = {
    "text.usetex": True,
    "font.size": 20,
    "font.family": "serif",
}
plt.rcParams.update(params)
replications = 10
deltas = np.logspace(-3, 1, 50)# [0.01, 1]  # smaller radii usually show clearer concentration
gap_r = np.zeros([replications, deltas.shape[0]])

for rep in range(replications):
    x, y = prepare_data(d=d, N=N, label_noise=label_noise, noise_mag=noise_mag, beta_constrained=beta_constrained)
    y = np.asarray(y).reshape(-1)
    # empirical SVM
    beta, b = fit_hinge_erm_cvx(
        x,
        y,
        fit_intercept=True,
        beta_constrained=beta_constrained,
    )
    b = float(np.asarray(b).reshape(()))
    beta = np.asarray(beta).reshape(-1)

    print("Intercept b:", b)

    # MOT parameters
    theta1 = 2.0
    theta2 = 2.0

    # plotting helpers
    x1 = x[:, 0]
    x2 = x[:, 1]
    label_1 = (y >= 0).reshape(-1)
    label_0 = (y < 0).reshape(-1)

    for cnt, delta in enumerate(deltas):
        # CVX variables
        lambda_var = cvx.Variable(nonneg=True)
        t = cvx.Variable()
        eta = cvx.Variable(N, nonneg=True)
        zeta = cvx.Variable(N)  # epigraph for hinge d-transform (renamed from p)

        beta_norm_sq = float(np.linalg.norm(beta) ** 2)

        constraints = [
            cvx.sum(eta) / N <= theta2 * lambda_var,
            zeta >= 0,
        ]

        for i in range(N):
            margin_i = 1.0 - y[i] * (beta @ x[i, :] + b)

            # conjugate (quadratic) term
            quad_term = (beta_norm_sq / (4.0 * theta1)) * cvx.inv_pos(lambda_var)

            # IMPORTANT: correct hinge d-transform epigraph:
            # zeta_i >= max(0, margin_i + quad_term)
            constraints += [
                zeta[i] >= margin_i + quad_term,
                cvx.constraints.exponential.ExpCone(zeta[i] - t, theta2 * lambda_var, eta[i]),
            ]

        prob = cvx.Problem(cvx.Minimize(lambda_var * delta + t), constraints)
        prob.solve( solver=cvx.ECOS,
        abstol=1e-9,
        reltol=1e-9,
        feastol=1e-9,
        max_iters=50000,
        verbose=False)

        print("delta:", delta, "status:", prob.status)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Solver failed with status {prob.status}")

        lambda_opt = float(lambda_var.value)
        t_opt = float(t.value)
        zeta_opt = np.asarray(zeta.value).reshape(-1)

        # worst-case moved points: move only where zeta_opt > 0 (active samples)
        step = 1.0 / (2.0 * lambda_opt * theta1)
        x_perturb = x.copy()
        # active = zeta_opt > 1e-8
        m = 1.0 - y * (x @ beta + b)
        quad_term_num = beta_norm_sq / (4.0 * theta1 * lambda_opt)
        active = (m + quad_term_num) > 1e-9
        x_perturb[active] = x[active] - step * (y[active].reshape(-1, 1) * beta.reshape(1, -1))

        # weights from ExpCone:
        # w_i ∝ exp((zeta_i - t)/(lambda*theta2))
        w = np.exp((zeta_opt - t_opt) / (lambda_opt * theta2))

        z = w / np.mean(w)
        p = w / np.sum(w)              # actual worst case probabilities
        arrow_mask = active & (p > 1e-9)   # tune 1e-3 or 5e-4


        # diagnostics (optional)
        print("  lambda:", lambda_opt, "t:", t_opt)
        print("  zeta range:", float(zeta_opt.min()), float(zeta_opt.max()))
        print("  w range:", float(w.min()), float(w.max()))
        print("  active:", int(active.sum()), "/", N)
        loss_wc = hinge_losses(beta, b, x_perturb, y)

    
        primal_val = float(np.mean(w * loss_wc))

        # dual objective value from CVX
        dual_val = float(lambda_opt * delta + t_opt)

        gap = primal_val - dual_val

        print("\n[Verification]")
        print("  primal_val  (mean z * loss):", primal_val)
        print("  dual_val    (lambda*delta+t):", dual_val)
        print("  gap (primal - dual):", gap)
        gap_r[rep, cnt] = gap

# breakpoint()
breakpoint()
# fig, ax = plt.subplots(figsize=(6, 5))
# plt.plot(deltas, gap_r, lw=2.5, c='#64646D')
# plt.xscale("log")
# ax.set_xlabel("$r$")
# ax.set_ylabel("gap")
# ax.grid(alpha=0.2)

# plt.show()

# gap_r shape: (replications, len(deltas))
gap_mean = np.mean(gap_r, axis=0)
gap_std  = np.std(gap_r, axis=0)

fig, ax = plt.subplots(figsize=(6, 5))

# optional: plot individual runs faintly
for rep in range(gap_r.shape[0]):
    ax.plot(deltas, gap_r[rep], color="#B0B0B8", lw=1.0, alpha=0.3)

# mean curve
ax.plot(
    deltas,
    gap_mean,
    lw=2.5,
    color="#64646D",
    label="mean gap",
)

# plus minus 1 std band
ax.fill_between(
    deltas,
    gap_mean - gap_std,
    gap_mean + gap_std,
    color="#64646D",
    alpha=0.25
    # label="plus minus one std",
)

ax.axhline(0.0, lw=2, color="black", alpha=0.6)

ax.set_xscale("log")
ax.set_xlabel("$r$")
ax.set_ylabel("gap")
ax.grid(alpha=0.2)
ax.legend(frameon=True)

plt.show()
breakpoint()
