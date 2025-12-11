import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import cvxpy as cvx
import matplotlib as mpl

np.random.seed(0)

d = 2
N = 32

beta_true = np.random.randn(d, 1)
x = np.random.randn(N, d)
y_true = np.matmul(x, beta_true)
y = y_true # + np.random.randn(N, 1)
label_1 = y >= 0
label_0 = y < 0
z = np.ones(N)
z = z/np.sum(z)

y[y >= 0] = 1
y[y < 0] = -1

y_noisy = y + 0.1 * np.random.randn(N, 1)
y_noisy[y_noisy >= 0] = 1
y_noisy[y_noisy < 0] = -1

############## Empirical SVM ##############
SVMModel = svm.SVC(kernel='linear').fit(x, y.ravel())
sv = SVMModel.support_vectors_
beta = SVMModel.coef_[0]
b = SVMModel.intercept_
###########################################

sz = 60
x1 = x[:, 0]
x2 = x[:, 1]

params = {
    'text.usetex': True,
    'font.size': 20,
    'font.family': 'serif', # Matplotlib family name
}
plt.rcParams.update(params)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x1[label_1.ravel()], x2[label_1.ravel()], sz, facecolors='r', edgecolors='r', alpha=0.4)
ax.scatter(x1[label_0.ravel()], x2[label_0.ravel()], sz, facecolors='b', edgecolors='b', alpha=0.4)
x_plot = np.arange(np.min(x1), np.max(x1), 0.1)
y_plot = -(beta[0]/beta[1]*x_plot) - b/beta[1]
ax.plot(x_plot, y_plot, linewidth=2, color='gray')
ax.legend(['Positive', 'Negative'],loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    frameon=True)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim([-2, 3])
ax.set_ylim([-3, 3])
ax.grid(True, alpha=0.2)
deltas = [0.01, 0.001]
# plt.rcParams.update({"font.size": 12})

for cnt, delta_ in enumerate(deltas):
    # CVX model
    delta = delta_
    lambda_var = cvx.Variable()
    t = cvx.Variable()
    eta = cvx.Variable(N)
    p = cvx.Variable(N)
    constraints = [cvx.sum(eta)/N <= lambda_var,
                   eta >= 0,
                   p >= 0]
    for i in range(N):
        constraints.append(
            cvx.constraints.exponential.ExpCone(
                p[i] - t, lambda_var, eta[i]
            )
        )
        constraints.append(p[i] >= cvx.pos(1 - y[i] * beta.T @ x[i, :]))

    prob = cvx.Problem(cvx.Minimize(lambda_var * delta + t), constraints)
    prob.solve()

    print("CVX status:", prob.status)
    p_opt = p.value
    lambda_opt = lambda_var.value

    # adversarially perturbed points
    x_perturb = np.zeros([N, d])
    for i in range(N):
        if np.abs(p_opt[i]) < 1e-8:
            x_perturb[i, :] = x[i, :]
        else:
            x_perturb[i, :] = x[i, :] - 0.5 / lambda_opt * y[i] * beta.T

    # plotting: original points, perturbed points, and arrows
    fig, ax = plt.subplots(figsize=(6, 5))

    # original locations (faint)
    ax.scatter(x1[label_1.ravel()], x2[label_1.ravel()], sz,
               facecolors="none", edgecolors="r", alpha=0.15)
    ax.scatter(x1[label_0.ravel()], x2[label_0.ravel()], sz,
               facecolors="none", edgecolors="b", alpha=0.15)

    # perturbed locations (solid)
    x_new1 = x_perturb[:, 0]
    x_new2 = x_perturb[:, 1]
    ax.scatter(x_new1[label_1.ravel()], x_new2[label_1.ravel()], sz,
               facecolors="r", edgecolors="r", alpha=0.4)
    ax.scatter(x_new1[label_0.ravel()], x_new2[label_0.ravel()], sz,
               facecolors="b", edgecolors="b", alpha=0.4)

    # displacement vectors old -> new for all moved points
    disp = x_perturb - x
    move_mask = np.linalg.norm(disp, axis=1) > 1e-6   # ignore numerical zeros

    ax.quiver(
        x[move_mask, 0], x[move_mask, 1],          # start points
        disp[move_mask, 0], disp[move_mask, 1],    # dx, dy
        angles="xy", scale_units="xy", scale=1.0,
        color="g", width=0.003, alpha=0.9
    )

    # decision boundary
    x_plot = np.arange(np.min(x_new1), np.max(x_new1), 0.1)
    y_plot = -(beta[0] / beta[1] * x_plot) - b / beta[1]
    ax.plot(x_plot, y_plot, linewidth=2, color="gray")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim([-2, 3])
    ax.set_ylim([-3, 3])
    ax.grid(True, alpha=0.2)

    # optional legend entry for arrows
    # create a dummy line for arrows in legend
    arrow_handle = mpl.lines.Line2D([0], [0], color="g", lw=2)
    pos_handle = mpl.lines.Line2D([0], [0], color="r", marker="o",
                                  linestyle="", markersize=8, alpha=0.6)
    neg_handle = mpl.lines.Line2D([0], [0], color="b", marker="o",
                                  linestyle="", markersize=8, alpha=0.6)
    ax.legend([pos_handle, neg_handle, arrow_handle],
              ["Positive", "Negative", "Adversarial shift"], loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    frameon=True)

    plt.show()
    fig.savefig(
        ("worst_case_points_{}.pdf").format(cnt),
        bbox_inches="tight",
    )
deltas = [0.01, 0.001]
import matplotlib as mpl

x_perturb = x
fig, ax = plt.subplots(figsize=(6, 5))

x_new1 = x_perturb[:, 0]
x_new2 = x_perturb[:, 1]
ax.scatter(x_new1[label_1.ravel()], x_new2[label_1.ravel()], sz, facecolors='r', edgecolors='r', alpha=0.4)
ax.scatter(x_new1[label_0.ravel()], x_new2[label_0.ravel()], sz, facecolors='b', edgecolors='b', alpha=0.4)
indx = 16
ax.scatter(x1[indx], x2[indx], sz, facecolors='b', edgecolors='b', alpha=0.2);
ax.scatter(x_new1[indx],x_new2[indx], sz, facecolors='b', edgecolors='b', alpha=0.4)
# plot displayment
x_plot = np.arange(np.min(x_new1), np.max(x_new1), 0.1)
y_plot = -(beta[0]/beta[1]*x_plot) - b/beta[1]
ax.plot(x_plot, y_plot,linewidth=2, color='gray')
ax.legend(['Positive', 'Negative'],loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    frameon=True)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim([-2, 3])
ax.set_ylim([-3, 3])
ax.grid(True, alpha=0.2)

plt.show()
fig.savefig(
    ("worst_case_points_plain.pdf"),
    bbox_inches="tight",
)


z = np.ones_like(label_1) * 1 / x.shape[0]

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(x_new1, x_new2, s=sz, c=z, cmap=mpl.colormaps["copper_r"])
plt.colorbar(sc, ax=ax, label='likelihood ratio')

x_plot = np.arange(min(x_new1), max(x_new1), 0.1)
y_plot  = -(beta[0]/beta[1] * x_plot) - b/beta[1]

ax.plot(x_plot, y_plot, linewidth=1.5, color=[128/255, 128/255, 128/255])

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# ax.set_title('$\delta=0.1$, Likelihood Perturbation', fontsize=14)

ax.grid(True, which='both', color='grey', linewidth=0.2)
ax.set_facecolor('w')
ax.set_xlim([-2, 3])
ax.set_ylim([-3, 3])

fig.tight_layout()
fig.savefig(
        ("worst_case_weights_plain.pdf"),
        bbox_inches="tight",
    )
