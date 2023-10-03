import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def plot_figure(plt, x, y, color="b", label=None, with_shade=True):
    mean_ = np.mean(y, axis=1)
    err_ = np.std(y, axis=1)

    plt.plot(x, mean_, color=color, linewidth=2, label=label)
    plt.plot(x, mean_ + err_, color=color, linewidth=1, alpha=0.2)
    plt.plot(x, mean_ - err_, color=color, linewidth=1, alpha=0.2)
    if with_shade:
        plt.fill_between(x, mean_ - err_, mean_ + err_, alpha=0.1, facecolor=color)


def plot_results_MNIST(
    results,
    d,
    N,
    p,
    replications,
    name_file="_",
    save_format="png",
):
    c_rs = results["radius_range"]
    plt.rcParams.update({"font.size": 12})

    mot_color = colors["red"]
    wass_color = colors["limegreen"]
    kl_color = colors["blue"]

    fig, axis = plt.subplots(1, 1)
    plot_figure(axis, c_rs, results["kl_acc"], color=kl_color, label="Kullback-Leibler")
    plot_figure(axis, c_rs, results["wass_acc"], color=wass_color, label="Wasserstein")
    plot_figure(axis, c_rs, results["mot_acc"], color=mot_color, label="MOT")

    axis.legend()
    plt.grid(alpha=0.2)
    # plt.title("Out-of-sample correct classification rate $n={}$".format(N))
    plt.ylabel("CCR")
    plt.xlabel("$r$")
    plt.xscale("log")

    plt.grid(True, which="both", alpha=0.2)

    save_name_format = "_d_{}_N_{}_p_{}_reps_{}."
    fig.savefig(
        (
            "results/figures/acc_test_" + name_file + save_name_format + save_format
        ).format(
            d,
            N,
            p,
            replications,
        ),
        format=save_format,
        bbox_inches="tight",
    )

    # LOSS TEST
    fig, axis = plt.subplots(1, 1)
    plot_figure(axis, c_rs, results["mot_loss_test"], color=mot_color, label="MOT")
    plot_figure(
        axis, c_rs, results["kl_loss_test"], color=kl_color, label="Kullback-Leibler"
    )
    plot_figure(
        axis, c_rs, results["wass_loss_test"], color=wass_color, label="Wasserstein"
    )
    axis.legend()
    plt.yscale("log")
    plt.grid(alpha=0.2)
    plt.title("Test Loss")
    plt.xlabel("$r$")
    plt.show()
    fig.savefig(
        (
            "results/figures/loss_test_" + name_file + save_name_format + save_format
        ).format(
            d,
            N,
            p,
            replications,
        ),
        format=save_format,
        bbox_inches="tight",
    )

    # LOSS TRAIN
    fig, axis = plt.subplots(1, 1)
    plot_figure(axis, c_rs, results["mot_loss_train"], color=mot_color, label="MOT")
    plot_figure(
        axis, c_rs, results["kl_loss_train"], color=kl_color, label="Kullback-Leibler"
    )
    plot_figure(
        axis, c_rs, results["wass_loss_train"], color=wass_color, label="Wasserstein"
    )
    axis.legend()
    plt.yscale("log")
    plt.grid(alpha=0.2)
    plt.title("Train Loss")
    plt.xlabel("$r$")
    plt.show()
    fig.savefig(
        (
            "results/figures/loss_train_" + name_file + save_name_format + save_format
        ).format(
            d,
            N,
            p,
            replications,
        ),
        format=save_format,
        bbox_inches="tight",
    )


def plot_results(
    results,
    d,
    N,
    sparse_beta,
    p,
    statistical_error,
    corruption_error,
    replications,
    theta1=1,
    theta2=1,
    name_file="_",
    save_format="png",
):
    c_rs = results["radius_range"]
    mot_color = colors["brown"]
    wass_color = colors["royalblue"]
    kl_color = colors["seagreen"]

    fig, axis = plt.subplots(1, 1)
    plot_figure(axis, c_rs, results["kl_acc"], color=kl_color, label="Kullback-Leibler")
    plot_figure(axis, c_rs, results["wass_acc"], color=wass_color, label="Wasserstein")
    plot_figure(axis, c_rs, results["mot_acc"], color=mot_color, label="MOT")
    axis.legend()
    plt.grid(alpha=0.2)
    plt.title("Test - Accuracy")
    plt.xlabel("$r$")
    plt.show()

    fig.savefig(
        (
            "results/figures/acc_test_"
            + name_file
            + "_sparse_beta_{}_d_{}_N_{}_p_{}_theta1_{}_theta2_{}_se_{}_ce_{}_reps_{}."
            + save_format
        ).format(
            sparse_beta,
            d,
            N,
            p,
            theta1,
            theta2,
            statistical_error,
            corruption_error,
            replications,
        ),
        format=save_format,
        bbox_inches="tight",
    )

    # LOSS TEST
    fig, axis = plt.subplots(1, 1)
    plot_figure(axis, c_rs, results["mot_loss_test"], color=mot_color, label="MOT")
    plot_figure(
        axis, c_rs, results["kl_loss_test"], color=kl_color, label="Kullback-Leibler"
    )
    plot_figure(
        axis, c_rs, results["wass_loss_test"], color=wass_color, label="Wasserstein"
    )
    axis.legend()
    plt.yscale("log")
    plt.grid(alpha=0.2)
    plt.title("Test Loss")
    plt.xlabel("$r$")
    plt.show()
    fig.savefig(
        (
            "results/figures/loss_test_"
            + name_file
            + "_sparse_beta_{}_d_{}_N_{}_p_{}_theta1_{}_theta2_{}_se_{}_ce_{}_reps_{}."
            + save_format
        ).format(
            sparse_beta,
            d,
            N,
            p,
            theta1,
            theta2,
            statistical_error,
            corruption_error,
            replications,
        ),
        format=save_format,
        bbox_inches="tight",
    )

    # LOSS TRAIN
    fig, axis = plt.subplots(1, 1)
    plot_figure(axis, c_rs, results["mot_loss_train"], color=mot_color, label="MOT")
    plot_figure(
        axis, c_rs, results["kl_loss_train"], color=kl_color, label="Kullback-Leibler"
    )
    plot_figure(
        axis, c_rs, results["wass_loss_train"], color=wass_color, label="Wasserstein"
    )
    axis.legend()
    plt.yscale("log")
    plt.grid(alpha=0.2)
    plt.title("Train Loss")
    plt.xlabel("$r$")
    plt.show()
    fig.savefig(
        (
            "results/figures/loss_train_"
            + name_file
            + "_sparse_beta_{}_d_{}_N_{}_p_{}_theta1_{}_theta2_{}_se_{}_ce_{}_reps_{}."
            + save_format
        ).format(
            sparse_beta,
            d,
            N,
            p,
            theta1,
            theta2,
            statistical_error,
            corruption_error,
            replications,
        ),
        format=save_format,
        bbox_inches="tight",
    )
