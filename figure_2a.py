import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from models.MOT_Robust_CLF import MOT_Robust_CLF
from models.WASS_Robust_CLF import WASS_Robust_CLF
from models.KL_Robust_CLF import KL_Robust_CLF
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as LR_sklearn
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from utils.data_utils import generate_data_LR4C, ClassificationDataGenerator
from tqdm import tqdm
from utils.plot_utils import plot_figure

cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
################################################################################

#### Data
N_train_all = [32, 64, 256, 1024]

N_total = 10000
d = 32
mot_color = colors["brown"]
wass_color = colors["royalblue"]
kl_color = colors["seagreen"]
replications = 10  # number of independent runs

statistical_error = 1
corruption_error = 0

n_rs = 10
theta1 = 1
theta2 = 1
p = 1
c_rs = np.logspace(-7, 0, n_rs)  # The range of radius we are scanning
epsilon = np.mean(c_rs)

np.random.seed(0)
sparse_beta = True

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
    if sparse_beta == True:
        beta_true = np.random.rand(d)
        for i in range(d - 1):
            beta_true[np.random.randint(d)] = 0
    else:
        beta_true = np.random.rand(d)
    beta_true = beta_true / np.linalg.norm(beta_true)
    data_generator = ClassificationDataGenerator(beta_true)

    for rep in tqdm(range(replications)):
        SEED = rep
        X_corrupt, y_corrupt, X_true, y_true = data_generator.generate_data_corrupt(
            N_total, seed=SEED, plotting=False
        )
        if statistical_error == 1 and corruption_error == 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_true, y_true, train_size=N_train / N_total
            )
        elif statistical_error == 0 and corruption_error == 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_corrupt, y_corrupt, train_size=N_train / N_total
            )
            X_test = X_true
            y_test = y_true
        elif statistical_error == 1 and corruption_error == 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_corrupt, y_corrupt, train_size=N_train / N_total
            )

        clf_MOT = MOT_Robust_CLF(
            fit_intercept=True, theta1=theta1, theta2=theta2, p=p, verbose=False
        )
        clf_WASS = WASS_Robust_CLF(fit_intercept=True, p=p, verbose=False)
        clf_KL = KL_Robust_CLF(fit_intercept=True, verbose=False)
        for i, c_r in enumerate(c_rs):
            clf_MOT.c_r = c_r
            clf_MOT.fit(X_train, y_train)
            mot_acc[i, rep] = clf_MOT.score(X_test, y_test)
            mot_loss_train[i, rep] = clf_MOT.loss(X_train, y_train)
            mot_loss_test[i, rep] = clf_MOT.loss(X_test, y_test)

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

    # SAVE results

    np.savez(
        "results/results_SVM_sparse_beta_{}_d_{}_N_{}_p_{}_theta1_{}_theta2_{}_se_{}_ce_{}_reps_{}.npz".format(
            sparse_beta,
            d,
            N_train,
            p,
            theta1,
            theta2,
            statistical_error,
            corruption_error,
            replications,
        ),
        d=d,
        statistical_error=statistical_error,
        corruption_error=corruption_error,
        sparse_beta=sparse_beta,
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
