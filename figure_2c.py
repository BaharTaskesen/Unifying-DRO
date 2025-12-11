import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from models.MOT_Robust_CLF import MOT_Robust_CLF
from models.WASS_Robust_CLF import WASS_Robust_CLF
from models.KL_Robust_CLF import KL_Robust_CLF
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from utils.theta1_cross_val import theta1_cross_val
from utils.data_utils import prepare_data

rc("font", family="serif")
rc("text", usetex=True)

cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
################################################################################

#### Data
np.random.seed(0)


N_total = 10_000
replications = 10  # number of independent runs
d = 100

N_train_all = [d, 2*d,  3*d, 4*d, 5*d]
sparsity_degree = 25

n_rs = 10
p = "inf"
noise_mag = 0.1
c_rs = np.logspace(-3, 0, n_rs)  # range of radius
# ################################################################################
nmbrs = np.arange(1, 10, 1)
theta1s = np.hstack([1 + np.logspace(-4, 0, 10), nmbrs[2:], 10, 1e2, 1e3, 1e4, 1e5])
# ################################################################################


for N_train in N_train_all:
    mot_acc = np.zeros([n_rs, replications])
    mot_loss_train = np.zeros([n_rs, replications])
    mot_loss_test = np.zeros([n_rs, replications])

    wass_acc = np.zeros([n_rs, replications])
    wass_loss_train = np.zeros([n_rs, replications])
    wass_loss_test = np.zeros([n_rs, replications])

    kl_acc = np.zeros([n_rs, replications])
    kl_loss_train = np.zeros([n_rs, replications])
    kl_loss_test = np.zeros([n_rs, replications])

    for rep in tqdm(range(replications)):
        X, y = prepare_data(
            d=d,
            N=N_total,
            sparsity_degree=sparsity_degree,
            sparse_beta=True,
            noise_mag=noise_mag,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=N_train/N_total
        )
        clf_MOT = MOT_Robust_CLF(
            fit_intercept=True, theta1=1, theta2=1, p=p, verbose=False
        )
        clf_WASS = WASS_Robust_CLF(fit_intercept=True, p=p, verbose=False)
        clf_KL = KL_Robust_CLF(fit_intercept=True, verbose=False)
        for i, c_r in enumerate(c_rs):
            clf_MOT.c_r = c_r
            clf_MOT_tuned = theta1_cross_val(clf_MOT, theta1s, X_train, y_train)
            mot_acc[i, rep] = clf_MOT_tuned.score(X_test, y_test)
            mot_loss_train[i, rep] = clf_MOT_tuned.loss(X_train, y_train)
            mot_loss_test[i, rep] = clf_MOT_tuned.loss(X_test, y_test)

            clf_WASS.c_r = c_r
            clf_WASS.fit(X_train, y_train)
            wass_acc[i, rep] = clf_WASS.score(X_test, y_test)
            wass_loss_train[i, rep] = clf_WASS.loss(X_train, y_train)
            wass_loss_test[i, rep] = clf_WASS.loss(X_test, y_test)

            clf_KL.c_r = c_r
            clf_KL.fit(X_train, y_train)
            kl_acc[i, rep] = clf_KL.score(X_test, y_test)
            kl_loss_train[i, rep] = clf_KL.loss(X_train, y_train)
            kl_loss_test[i, rep] = clf_KL.loss(X_test, y_test)
            # breakpoint()

    # SAVE results

    np.savez(
        "results/NEW_results_dn_simulation_d_{}_N_{}_p_{}_noise_mag_{}_sparsity_deg_{}_reps_{}.npz".format(
            d,
            N_train,
            p,
            noise_mag*10,
            sparsity_degree,
            replications,
        ),
        number_radius=n_rs,
        noise_mag=noise_mag,
        theta_splits=theta1s.shape[0],
        sparsity_beta=sparsity_degree,
        d=d,
        radius_range=c_rs,
        mot_acc=mot_acc,
        mot_loss_train=mot_loss_train,
        mot_loss_test=mot_loss_test,
        wass_acc=wass_acc,
        wass_loss_train=wass_loss_train,
        wass_loss_test=wass_loss_test,
        kl_acc=kl_acc,
        kl_loss_train=kl_loss_train,
        kl_loss_test=kl_loss_test,
    )
