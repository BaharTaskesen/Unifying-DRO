import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from models.MOT_Robust_CLF import MOT_Robust_CLF
from models.WASS_Robust_CLF import WASS_Robust_CLF
from utils.data_utils import prepare_data

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from utils.plot_utils import plot_figure


BASE_SEED = 123
np.random.seed(BASE_SEED)
params = {
    "text.usetex": True,
    "font.size": 20,
    "font.family": "serif",
}
plt.rcParams.update(params)



N_total = 10000
replications = 10
d = 100
n_features = d
N_train = 2 * d
N_train_all = [2 * d]  # n = 100 for the radius figure
n_rs = 10
p = "inf"
noise_mag = 0.1
beta_constrained = True
nmbrs = np.arange(1, 10, 1)
theta1s = np.hstack([1 + np.logspace(-5, 0, 10), nmbrs[2:], 10, 1e2, 1e3, 1e4, 1e5])
sparsity = 10
label_noise = 0.2


splits = theta1s.shape[0]
mot_acc = np.zeros((splits+1, replications))
c_r = 1
def theta2_from_theta1(theta1, big_M=1e6):
    # Enforce 1/theta1 + 1/theta2 = 1 safely
    if theta1 <= 1.0 + 1e-12:
        return float(big_M)
    return float(theta1 / (theta1 - 1.0))
theta1s = np.hstack([[1], theta1s])

# -----------------------------
# Monte Carlo loop
# -----------------------------
for rep in tqdm(range(replications), desc="theta sweep MC"):
    # fresh dataset each replication
    np.random.seed(BASE_SEED + rep)

    X, y = prepare_data(
        d=n_features,
        N=N_total,
        sparse_beta=True,
        sparsity_degree=sparsity,
        noise_mag=noise_mag,
        label_noise=label_noise,
        beta_constrained=beta_constrained
    )

    # reproducible split with exact size
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=N_train,
        test_size=N_total - N_train,
        shuffle=True,
        random_state=BASE_SEED + 10_000 + rep,
    )
    for i, theta1 in enumerate(theta1s):
        if theta1 == 1:
            clf = WASS_Robust_CLF(fit_intercept=True, p=p, verbose=False)
            clf.c_r = float(c_r)
            clf.fit(X_train, y_train)
            mot_acc[i, rep] = clf.score(X_test, y_test)

        clf_MOT = MOT_Robust_CLF(fit_intercept=True, theta1=1.0, theta2=1.0, p=p, verbose=False)
        clf_MOT.c_r = float(c_r)

        clf_MOT.theta1 = float(theta1)
        clf_MOT.theta2 = theta2_from_theta1(float(theta1))

        clf_MOT.model_prepared = False
        clf_MOT.fit(X_train, y_train)

        mot_acc[i, rep] = clf_MOT.score(X_test, y_test)

# save once
np.savez(
    f"results/theta_performance_d_{n_features}_N_{N_train}_p_{p}_reps_{replications}.npz",
    d=n_features,
    radius=c_r,
    sparsity=sparsity,
    noise_mag=noise_mag,
    mot_acc=mot_acc,
    theta1s=theta1s,
)



cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


mot_color = colors["red"]
wass_color = colors["limegreen"]
kl_color = colors["maroon"]

fig, axis = plt.subplots(1, 1)
plot_figure(axis, theta1s[:-2], mot_acc[:-2], color=kl_color)

# axis.legend()

plt.grid(alpha=0.2)
# plt.title("Out-of-sample correct classification rate $n={}$".format(N))
plt.ylabel("CCR")
plt.xlabel(r"$\theta_1$")
plt.xscale("log")

plt.grid(True, which="both", alpha=0.2)
name_file = "theta_performance"
save_format = "pdf"

save_name_format = "_d_{}_N_{}_p_{}_reps_{}."

plt.show()


fig.savefig(
    (
        "figures/theta_acc_test_beta_constrained_{}" + name_file + save_name_format + save_format
    ).format(
        beta_constrained,
        d,
        N_train,
        p,
        replications,
    ),
    format=save_format,
    bbox_inches="tight",
)