import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import LogLocator, NullFormatter

cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def plot_objectives(res, name_file="objective_vs_radius", save_format="pdf"):
    r = res["radius_range"]

    mot = res["mot_obj"]
    wass = res["wass_obj"]
    kl = res["kl_obj"]

    mot_m = mot.mean(axis=1); mot_s = mot.std(axis=1, ddof=1)
    wass_m = wass.mean(axis=1); wass_s = wass.std(axis=1, ddof=1)
    kl_m = kl.mean(axis=1); kl_s = kl.std(axis=1, ddof=1)

    plt.figure()
    plt.xscale("log")

    plt.plot(r, mot_m, label="MOT objective")
    plt.fill_between(r, mot_m - mot_s, mot_m + mot_s, alpha=0.15)

    plt.plot(r, wass_m, label="WASS objective")
    plt.fill_between(r, wass_m - wass_s, wass_m + wass_s, alpha=0.15)

    plt.plot(r, kl_m, label="KL objective")
    plt.fill_between(r, kl_m - kl_s, kl_m + kl_s, alpha=0.15)

    if "erm_obj" in res.files:
        erm = np.asarray(res["erm_obj"]).reshape(-1)
        plt.axhline(float(erm.mean()), linestyle="--", label="ERM train objective")

    plt.xlabel("radius r")
    plt.ylabel("objective value")
    plt.legend()
    plt.tight_layout()
    if save_format is not None:
        plt.savefig(name_file+"objective_vals."+save_format,
        format=save_format,
        bbox_inches="tight")
    # plt.close()
    plt.show()


def plot_figure(plt, x, y, color="b", label=None, linestyle='solid', with_shade=True):
    mean_ = np.mean(y, axis=1)
    err_ = np.std(y, axis=1)

    plt.plot(x, mean_, color=color, linewidth=2, linestyle=linestyle, label=label)
    plt.plot(x, mean_ + err_, color=color, linewidth=1, alpha=0.2)
    plt.plot(x, mean_ - err_, color=color, linewidth=1, alpha=0.2)
    if with_shade:
        plt.fill_between(x, mean_ - err_, mean_ + err_, alpha=0.1, facecolor=color)


def plot_results1(
    results,
    d,
    N,
    p,
    replications,
    plot_emp=False,
    name_file="_",
    save_format="png",
):
    c_rs = results["radius_range"]

    mot_color = '#CA6378'
    wass_color = '#015B76'
    kl_color = '#739842'
    fig, axis = plt.subplots(1, 1)
    plt.grid(True, which="both", alpha=0.2)

    plt.xscale("log")

    plot_figure(axis, c_rs, results["kl_acc"], color=kl_color, linestyle='dashed', label="Kullback-Leibler")
    plot_figure(axis, c_rs, results["wass_acc"], color=wass_color, linestyle='dashdot', label="Wasserstein")
    plot_figure(axis, c_rs, results["mot_acc"], color=mot_color, label="MOT")

    plt.ylabel("CCR")
    plt.xlabel("$r$")
    if plot_emp and ("svm_acc" in results.files):
        svm_acc = results["svm_acc"]  # shape (replications,)
        svm_acc = np.asarray(svm_acc).reshape(1, -1)  # (1, replications)
        svm_acc_mat = np.repeat(svm_acc, repeats=len(c_rs), axis=0)  # (n_rs, replications)

        plot_figure(axis, c_rs, svm_acc_mat, color=colors["black"], label="Empirical SVM")
    axis.legend(loc="upper left")
        # Force log scale on the axis object
    axis.set_xscale("log")

    # Force major and minor tick locations for log grid
    axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    axis.xaxis.set_minor_formatter(NullFormatter())

    # Make sure minor ticks are turned on and visible
    axis.tick_params(axis="x", which="major", length=7)
    axis.tick_params(axis="x", which="minor", length=4)

    # Make grid visible and in front
    axis.set_axisbelow(False)
    axis.grid(True, which="major", axis="x", alpha=0.45, linewidth=1.0)
    axis.grid(True, which="minor", axis="x", alpha=0.30, linewidth=0.6)
    axis.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.8)
    plt.show()

    fig.savefig(name_file+"CCR."+save_format,
        format=save_format,
        bbox_inches="tight",
    )

