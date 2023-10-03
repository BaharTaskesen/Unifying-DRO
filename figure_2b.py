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
from utils.data_utils import load_MNIST, prepare_data
from matplotlib import rc
from theta1_cross_val import theta1_cross_val


rc("font", family="serif")
rc("text", usetex=True)

cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
################################################################################

#### Data
np.random.seed(0)

N_train = 100  # 784  # , 500, 1000]  #  [100]  # , 1000]  # , 2000]

N_total = 10000
replications = 10  # number of independent runs


p = "inf"
n_rs = 10  # discretization of radius


################################################################################
# theta1s = np.logspace(0, 5, splits)
nmbrs = np.arange(1, 10, 1)
theta1s = np.hstack([1 + np.logspace(-5, 0, 10), nmbrs[2:], 10, 1e2, 1e3, 1e4, 1e5])
################################################################################
c_r = 1e-1 * 5

# DIGITS = [1, 8]
n_features = N_train  # 784
splits = theta1s.shape[0]

X, y = prepare_data(d=n_features, N=N_total, sparse_beta=True, noise_mag=0.1)
# load_MNIST(DIGITS, n_features)
mot_acc = np.zeros([splits, replications])


for rep in tqdm(range(replications)):
    SEED = rep
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_train / X.shape[0]
    )
    clf_MOT = MOT_Robust_CLF(fit_intercept=True, theta1=1, theta2=1, p=p, verbose=False)
    clf_MOT.c_r = c_r
    for i, theta1 in enumerate(theta1s):
        clf_MOT.theta1 = theta1
        if theta1 == 1:
            theta2 = max(theta1s)
            print("corner")
        elif theta1 == max(theta1s):
            theta2 = 1
            print("corner")
        else:
            theta2 = 1 / (1 - 1 / theta1)
        clf_MOT.theta2 = theta2
        clf_MOT.model_prepared = False
        clf_MOT.fit(X_train, y_train)
        perf = clf_MOT.score(X_test, y_test)
        mot_acc[i, rep] = perf

    np.savez(
        "results/theta_performance_MNIST_theta_d_{}_N_{}_p_{}_reps_{}.npz".format(
            n_features,
            N_train,
            p,
            replications,
        ),
        d=n_features,
        radius=c_r,
        mot_acc=mot_acc,
        theta1s=theta1s,
    )
