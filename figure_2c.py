import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from configs.exp_radius import ExpConfig, to_jsonable_dict
from utils.experiment_io import make_run_name, save_config_json, savez_results, ensure_dir

from utils.data_utils import prepare_data
from utils.theta1_cross_val import theta1_cross_val

from models.MOT_Robust_CLF import MOT_Robust_CLF
from models.WASS_Robust_CLF import WASS_Robust_CLF
from models.KL_Robust_CLF import KL_Robust_CLF


def plot_curve(ax, x, y, color, label, linewidth=2.5, with_shade=True):
    # y expected shape: (len(x), replications)
    mean_ = np.mean(y, axis=1)
    err_ = np.std(y, axis=1) * 0.5
    ax.plot(x, mean_, color=color, linewidth=linewidth, label=label)
    if with_shade:
        ax.fill_between(x, mean_ - err_, mean_ + err_, alpha=0.1, color=color)


def main():
    cfg = ExpConfig()
    c_rs, theta1s = cfg.build_grids()

    out_npz_dir = "results"
    out_pdf_dir = "figures"
    ensure_dir(out_npz_dir)
    ensure_dir(out_pdf_dir)

    run_name = make_run_name("n_over_dx_" + cfg.tag())
    save_config_json(to_jsonable_dict(cfg), out_npz_dir, run_name)

    np.random.seed(cfg.base_seed)

    # build N_train list: [d, 2d, 3d, 4d, 5d]
    n_train_all = tuple(int(cfg.d * m) for m in (1, 2, 3, 4, 5))

    # aggregated containers: for each N_train we store best over radii per replication
    best_kl = []
    best_wass = []
    best_mot = []

    for n_train in n_train_all:
        mot_acc = np.zeros((cfg.n_rs, cfg.replications))
        mot_loss_train = np.zeros((cfg.n_rs, cfg.replications))
        mot_loss_test = np.zeros((cfg.n_rs, cfg.replications))

        wass_acc = np.zeros((cfg.n_rs, cfg.replications))
        wass_loss_train = np.zeros((cfg.n_rs, cfg.replications))
        wass_loss_test = np.zeros((cfg.n_rs, cfg.replications))

        kl_acc = np.zeros((cfg.n_rs, cfg.replications))
        kl_loss_train = np.zeros((cfg.n_rs, cfg.replications))
        kl_loss_test = np.zeros((cfg.n_rs, cfg.replications))

        for rep in tqdm(range(cfg.replications), desc=f"N_train={n_train}"):
            np.random.seed(cfg.base_seed + rep)

            X, y = prepare_data(
                d=cfg.d,
                N=cfg.n_total,
                sparse_beta=True,
                sparsity_degree=cfg.sparsity,
                noise_mag=cfg.noise_mag,
                label_noise=cfg.label_noise,
                beta_constrained=cfg.beta_constrained
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=n_train,
                test_size=cfg.n_total - n_train,
                shuffle=True,
                random_state=cfg.base_seed + 10_000 + rep,
            )

            clf_MOT = MOT_Robust_CLF(
                fit_intercept=True,
                theta1=1,
                theta2=1,
                p=cfg.p,
                beta_constrained=cfg.beta_constrained,
                verbose=False,
            )
            clf_WASS = WASS_Robust_CLF(
                fit_intercept=True,
                p=cfg.p,
                beta_constrained=cfg.beta_constrained,
                verbose=False,
            )
            clf_KL = KL_Robust_CLF(
                fit_intercept=True,
                beta_constrained=cfg.beta_constrained,
                verbose=False,
            )

            for i, c_r in enumerate(c_rs):
                # MOT: tune theta1
                clf_MOT.c_r = float(c_r)
                clf_MOT_tuned = theta1_cross_val(
                    clf_MOT,
                    theta1s,
                    X_train,
                    y_train,
                    n_folds=5,
                    metric="acc",
                    seed=cfg.base_seed + 20_000 + rep,
                )
                mot_acc[i, rep] = float(clf_MOT_tuned.score(X_test, y_test))
                mot_loss_train[i, rep] = float(clf_MOT_tuned.loss(X_train, y_train))
                mot_loss_test[i, rep] = float(clf_MOT_tuned.loss(X_test, y_test))

                # WASS
                clf_WASS.c_r = float(c_r)
                clf_WASS.fit(X_train, y_train)
                wass_acc[i, rep] = float(clf_WASS.score(X_test, y_test))
                wass_loss_train[i, rep] = float(clf_WASS.loss(X_train, y_train))
                wass_loss_test[i, rep] = float(clf_WASS.loss(X_test, y_test))

                # KL
                clf_KL.c_r = float(c_r)
                clf_KL.fit(X_train, y_train)
                kl_acc[i, rep] = float(clf_KL.score(X_test, y_test))
                kl_loss_train[i, rep] = float(clf_KL.loss(X_train, y_train))
                kl_loss_test[i, rep] = float(clf_KL.loss(X_test, y_test))

        # save per n_train
        run_name_n = f"{run_name}_ntrain{n_train}"
        npz_path = savez_results(
            {
                "run_name": run_name_n,
                "cfg_tag": cfg.tag(),
                "base_seed": cfg.base_seed,
                "d": cfg.d,
                "n_total": cfg.n_total,
                "n_train": n_train,
                "replications": cfg.replications,
                "p": cfg.p,
                "noise_mag": cfg.noise_mag,
                "sparsity": cfg.sparsity,
                "label_noise": cfg.label_noise,
                "beta_constrained": cfg.beta_constrained,
                "number_radius": cfg.n_rs,
                "radius_range": c_rs,
                "theta1s": theta1s,
                "mot_acc": mot_acc,
                "mot_loss_train": mot_loss_train,
                "mot_loss_test": mot_loss_test,
                "wass_acc": wass_acc,
                "wass_loss_train": wass_loss_train,
                "wass_loss_test": wass_loss_test,
                "kl_acc": kl_acc,
                "kl_loss_train": kl_loss_train,
                "kl_loss_test": kl_loss_test,
                "beta_constrained":cfg.beta_constrained
            },
            out_npz_dir,
            run_name_n,
        )

        # compute best over radii for each replication and store for the final plot
        res = np.load(npz_path, allow_pickle=True)
        best_kl.append(np.max(res["kl_acc"], axis=0))
        best_wass.append(np.max(res["wass_acc"], axis=0))
        best_mot.append(np.max(res["mot_acc"], axis=0))

    # # aggregate arrays: shape (len(n_train_all), replications)
    # best_kl = np.array(best_kl)
    # best_wass = np.array(best_wass)
    # best_mot = np.array(best_mot)

    # # save aggregated npz
    # agg_name = f"{run_name}_aggregate"
    # savez_results(
    #     {
    #         "run_name": agg_name,
    #         "cfg_tag": cfg.tag(),
    #         "base_seed": cfg.base_seed,
    #         "d": cfg.d,
    #         "n_total": cfg.n_total,
    #         "n_train_all": np.array(n_train_all, dtype=int),
    #         "replications": cfg.replications,
    #         "p": cfg.p,
    #         "noise_mag": cfg.noise_mag,
    #         "sparsity": cfg.sparsity,
    #         "label_noise": cfg.label_noise,
    #         "beta_constrained": cfg.beta_constrained,
    #         "best_kl": best_kl,
    #         "best_wass": best_wass,
    #         "best_mot": best_mot,
    #     },
    #     out_npz_dir,
    #     agg_name,
    # )

    # # plot pdf
    # colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # kl_color = colors["blue"]
    # wass_color = colors["limegreen"]
    # mot_color = colors["red"]

    # n_ds = np.array(n_train_all, dtype=float) / float(cfg.d)

    # fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    # plot_curve(ax, n_ds, best_kl, kl_color, label="Kullback-Leibler")
    # plot_curve(ax, n_ds, best_wass, wass_color, label="Wasserstein")
    # plot_curve(ax, n_ds, best_mot, mot_color, label="MOT")

    # ax.set_xlabel(r"$n/d_x$")
    # ax.set_ylabel("CCR")
    # ax.set_xticks(list(n_ds))
    # ax.grid(True, which="both", alpha=0.2)
    # ax.legend()

    # pdf_path = os.path.join(out_pdf_dir, f"{run_name}_n_over_dx.pdf")
    # fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    # plt.close(fig)


if __name__ == "__main__":
    main()
