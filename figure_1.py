import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import matplotlib as mpl

from utils.data_utils import prepare_data
from models.EMP_CLF import fit_hinge_erm_cvx


SEED = 0
np.random.seed(SEED)

# problem setup
d = 2
N = 50
noise_mag = 0.1
label_noise = 0.0
beta_constrained = False


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
sz = 60
x1 = x[:, 0]
x2 = x[:, 1]
label_1 = (y >= 0).reshape(-1)
label_0 = (y < 0).reshape(-1)

params = {
    "text.usetex": True,
    "font.size": 20,
    "font.family": "serif",
}
plt.rcParams.update(params)

# plot ERM boundary on original data
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x1[label_1], x2[label_1], sz, facecolors="r", edgecolors="r", alpha=0.4)
ax.scatter(x1[label_0], x2[label_0], sz, facecolors="b", edgecolors="b", alpha=0.4)

x_plot = np.arange(np.min(x1), np.max(x1), 0.1)
y_plot = -(beta[0] / beta[1] * x_plot) - b / beta[1]
ax.plot(x_plot, y_plot, linewidth=2, color="gray")

ax.legend(["Positive", "Negative"], loc="lower right",
          bbox_to_anchor=(0.98, 0.02), frameon=True)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-3.5, 3.5])
ax.grid(True, alpha=0.2)
plt.show()


# radii
deltas = [0.1, 1]  # smaller radii usually show clearer concentration

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
    prob.solve(verbose=False)

    print("delta:", delta, "status:", prob.status)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver failed with status {prob.status}")

    lambda_opt = float(lambda_var.value)
    t_opt = float(t.value)
    zeta_opt = np.asarray(zeta.value).reshape(-1)

    # worst-case moved points: move only where zeta_opt > 0 (active samples)
    step = 1.0 / (2.0 * lambda_opt * theta1)
    x_perturb = x.copy()
    active = zeta_opt > 1e-8
    x_perturb[active] = x[active] - step * (y[active].reshape(-1, 1) * beta.reshape(1, -1))

    # weights from ExpCone:
    # w_i ∝ exp((zeta_i - t)/(lambda*theta2))
    w = np.exp((zeta_opt - t_opt) / (lambda_opt * theta2))

    z = w / np.mean(w)

    # diagnostics (optional)
    print("  lambda:", lambda_opt, "t:", t_opt)
    print("  zeta range:", float(zeta_opt.min()), float(zeta_opt.max()))
    print("  w range:", float(w.min()), float(w.max()))
    print("  active:", int(active.sum()), "/", N)

    # plot moved points + arrows
    fig, ax = plt.subplots(figsize=(6, 5))

    # original points (faint)
    ax.scatter(x1[label_1], x2[label_1], sz, facecolors="none", edgecolors="r", alpha=0.15)
    ax.scatter(x1[label_0], x2[label_0], sz, facecolors="none", edgecolors="b", alpha=0.15)

    # perturbed points (solid)
    x_new1 = x_perturb[:, 0]
    x_new2 = x_perturb[:, 1]
    ax.scatter(x_new1[label_1], x_new2[label_1], sz, facecolors="r", edgecolors="r", alpha=0.4)
    ax.scatter(x_new1[label_0], x_new2[label_0], sz, facecolors="b", edgecolors="b", alpha=0.4)

    disp = x_perturb - x
    move_mask = np.linalg.norm(disp, axis=1) > 1e-6

    ax.quiver(
        x[move_mask, 0], x[move_mask, 1],
        disp[move_mask, 0], disp[move_mask, 1],
        angles="xy", scale_units="xy", scale=1.0,
        color="g", width=0.005, alpha=0.9
    )

    # decision boundary (ERM beta)
    x_plot = np.arange(np.min(x_new1), np.max(x_new1), 0.1)
    y_plot = -(beta[0] / beta[1] * x_plot) - b / beta[1]
    ax.plot(x_plot, y_plot, linewidth=2, color="gray")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-3.5, 3.5])
    ax.grid(True, alpha=0.2)

    arrow_handle = mpl.lines.Line2D([0], [0], color="g", lw=2)
    pos_handle = mpl.lines.Line2D([0], [0], color="r", marker="o",
                                  linestyle="", markersize=8, alpha=0.6)
    neg_handle = mpl.lines.Line2D([0], [0], color="b", marker="o",
                                  linestyle="", markersize=8, alpha=0.6)
    ax.legend([pos_handle, neg_handle, arrow_handle],
              ["Positive", "Negative", "Adversarial shift"],
              loc="lower right", bbox_to_anchor=(0.98, 0.02), frameon=True)

    fig.savefig(f"worst_case_points_{cnt}_beta_constrained_{beta_constrained}.pdf",
                bbox_inches="tight")
    plt.show()

    # plot weights heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x_new1, x_new2, s=sz + 10, c=z, cmap=mpl.colormaps["copper_r"])
    plt.colorbar(sc, ax=ax, label="Likelihood ratio")

    ax.plot(x_plot, y_plot, linewidth=1.5, color=[128/255, 128/255, 128/255])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-3.5, 3.5])
    ax.grid(True, which="both", color="grey", linewidth=0.2)
    ax.set_facecolor("w")

    fig.tight_layout()
    fig.savefig(f"figures/worst_case_weights_{cnt}.pdf", bbox_inches="tight")
    plt.show()


# "plain" plots (no perturbation) for comparison
x_plain = x.copy()
x_new1 = x_plain[:, 0]
x_new2 = x_plain[:, 1]

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x_new1[label_1], x_new2[label_1], sz, facecolors="r", edgecolors="r", alpha=0.4)
ax.scatter(x_new1[label_0], x_new2[label_0], sz, facecolors="b", edgecolors="b", alpha=0.4)

x_plot = np.arange(np.min(x_new1), np.max(x_new1), 0.1)
y_plot = -(beta[0] / beta[1] * x_plot) - b / beta[1]
ax.plot(x_plot, y_plot, linewidth=2, color="gray")

ax.legend(["Positive", "Negative"], loc="lower right",
          bbox_to_anchor=(0.98, 0.02), frameon=True)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-3.5, 3.5])
ax.grid(True, alpha=0.2)

fig.savefig(f"worst_case_points_plain_beta_constrained_{beta_constrained}.pdf",
            bbox_inches="tight")
plt.show()

# uniform weights baseline
z_uni = np.ones(N) / N
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(x_new1, x_new2, s=sz + 10, c=z_uni, cmap=mpl.colormaps["copper_r"])
plt.colorbar(sc, ax=ax, label="Likelihood ratio")

ax.plot(x_plot, y_plot, linewidth=2, color=[128/255, 128/255, 128/255])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-3.5, 3.5])
ax.grid(True, which="both", color="grey", linewidth=0.2)
ax.set_facecolor("w")

fig.tight_layout()
fig.savefig(f"worst_case_weights_plain_beta_cons_{beta_constrained}.pdf",
            bbox_inches="tight")
plt.show()
