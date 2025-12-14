import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from configs.exp_radius import ExpConfig, to_jsonable_dict
from utils.experiment_io import make_run_name, save_config_json, savez_results, ensure_dir

from utils.data_utils import prepare_data
from utils.theta1_cross_val import theta1_cross_val
from utils.plot_utils import plot_results1 as plot_results
from utils.plot_utils import plot_objectives

from models.MOT_Robust_CLF import MOT_Robust_CLF
from models.WASS_Robust_CLF import WASS_Robust_CLF
from models.KL_Robust_CLF import KL_Robust_CLF

# if you use this helper
from models.EMP_CLF import fit_hinge_erm_cvx

""""
NOTE that MOT's theta value is not tuned
"""
def main():
    cfg = ExpConfig()
    c_rs, theta1s = cfg.build_grids()

    # basic checks
    assert len(c_rs) == cfg.n_rs

    np.random.seed(cfg.base_seed)

    out_npz_dir = "results"
    out_pdf_dir = "figures"
    ensure_dir(out_npz_dir)
    ensure_dir(out_pdf_dir)

    run_name = make_run_name(cfg.tag())
    save_config_json(to_jsonable_dict(cfg), out_npz_dir, run_name)

    for n_train in cfg.n_train_all:
        # allocate
        mot_acc = np.zeros((cfg.n_rs, cfg.replications))
        mot_loss_train = np.zeros((cfg.n_rs, cfg.replications))
        mot_loss_test = np.zeros((cfg.n_rs, cfg.replications))

        wass_acc = np.zeros((cfg.n_rs, cfg.replications))
        wass_loss_train = np.zeros((cfg.n_rs, cfg.replications))
        wass_loss_test = np.zeros((cfg.n_rs, cfg.replications))

        kl_acc = np.zeros((cfg.n_rs, cfg.replications))
        kl_loss_train = np.zeros((cfg.n_rs, cfg.replications))
        kl_loss_test = np.zeros((cfg.n_rs, cfg.replications))

        mot_obj = np.zeros((cfg.n_rs, cfg.replications))
        wass_obj = np.zeros((cfg.n_rs, cfg.replications))
        kl_obj = np.zeros((cfg.n_rs, cfg.replications))

        erm_obj = np.zeros((1, cfg.replications))
        svm_acc = np.zeros((1, cfg.replications))

        for rep in tqdm(range(cfg.replications), desc=f"N_train={n_train}"):
            np.random.seed(cfg.base_seed + rep)

            X, y = prepare_data(
                d=cfg.d,
                N=cfg.n_total,
                sparse_beta=True,
                sparsity_degree=cfg.sparsity,
                noise_mag=cfg.noise_mag,
                label_noise=cfg.label_noise,
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=n_train,
                test_size=cfg.n_total - n_train,
                shuffle=True,
                random_state=cfg.base_seed + 10_000 + rep,
            )

            beta0, b0 = fit_hinge_erm_cvx(
                X_train,
                y_train,
                fit_intercept=True,
                beta_constrained=cfg.beta_constrained,
            )
            erm_obj[0, rep] = float(np.mean(np.maximum(0.0, 1.0 - y_train * (X_train @ beta0 + b0))))
            svm_acc[0, rep] = float(np.mean(((X_test @ beta0 + b0) >= 0).astype(int) * 2 - 1 == y_test))

            clf_MOT = MOT_Robust_CLF(
                fit_intercept=True,
                theta1=1,
                theta2=1,
                p=cfg.p,
                beta_constrained=cfg.beta_constrained,
                verbose=True,
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
                # MOT
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
                mot_obj[i, rep] = float(clf_MOT_tuned.obj_opt)
                mot_acc[i, rep] = float(clf_MOT_tuned.score(X_test, y_test))
                mot_loss_train[i, rep] = float(clf_MOT_tuned.loss(X_train, y_train))
                mot_loss_test[i, rep] = float(clf_MOT_tuned.loss(X_test, y_test))

                # WASS
                clf_WASS.c_r = float(c_r)
                clf_WASS.fit(X_train, y_train)
                wass_acc[i, rep] = float(clf_WASS.score(X_test, y_test))
                wass_loss_train[i, rep] = float(clf_WASS.loss(X_train, y_train))
                wass_loss_test[i, rep] = float(clf_WASS.loss(X_test, y_test))
                wass_obj[i, rep] = float(clf_WASS.obj.value)

                # KL
                clf_KL.c_r = float(c_r) 
                clf_KL.fit(X_train, y_train)
                kl_acc[i, rep] = float(clf_KL.score(X_test, y_test))
                kl_loss_train[i, rep] = float(clf_KL.loss(X_train, y_train))
                kl_loss_test[i, rep] = float(clf_KL.loss(X_test, y_test))
                kl_obj[i, rep] = float(clf_KL.obj.value[0])

                # breakpoint()

        # save per n_train
        run_name_n = f"{run_name}_ntrain{n_train}"
        npz_path = savez_results(
            {
                "run_name": run_name_n,
                "cfg_tag": cfg.tag(),
                "base_seed": cfg.base_seed,
                "number_radius": cfg.n_rs,
                "noise_mag": cfg.noise_mag,
                "theta_splits": theta1s.shape[0],
                "d": cfg.d,
                "n_total": cfg.n_total,
                "n_train": n_train,
                "p": cfg.p,
                "sparsity": cfg.sparsity,
                "label_noise": cfg.label_noise,
                "beta_constrained": cfg.beta_constrained,
                "radius_range": c_rs,
                "theta1s": theta1s,
                "svm_acc": svm_acc,
                "erm_obj": erm_obj,
                "mot_acc": mot_acc,
                "mot_loss_train": mot_loss_train,
                "mot_loss_test": mot_loss_test,
                "wass_acc": wass_acc,
                "wass_loss_train": wass_loss_train,
                "wass_loss_test": wass_loss_test,
                "kl_acc": kl_acc,
                "kl_loss_train": kl_loss_train,
                "kl_loss_test": kl_loss_test,
                "mot_obj": mot_obj,
                "wass_obj": wass_obj,
                "kl_obj": kl_obj,
            },
            out_npz_dir,
            run_name_n,
        )

        # plot and save PDFs with matching name
        res = np.load(npz_path, allow_pickle=True)

        plot_results(
            res,
            cfg.d,
            n_train,
            cfg.p,
            cfg.replications,
            name_file=os.path.join(out_pdf_dir, f"{run_name_n}_simulation"),
            plot_emp=True,
            save_format="pdf",
        )
        plot_objectives(
            res,
            name_file=os.path.join(out_pdf_dir, f"{run_name_n}_objectives"),
            save_format="pdf",
        )


if __name__ == "__main__":
    main()
