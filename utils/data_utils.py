import numpy as np
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression as LR_sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

from scipy.sparse import vstack

cm_piyg = plt.cm.PiYG
cm_bright = ListedColormap(["#b30065", "#178000"])


def load_MNIST(DIGITS, n_features):
    idx = 0
    n_class = 5000

    Z1 = load_svmlight_file(
        "datasets/MNIST_train_" + str(DIGITS[0]) + ".txt", n_features=n_features
    )
    Z2 = load_svmlight_file(
        "datasets/MNIST_train_" + str(DIGITS[1]) + ".txt", n_features=n_features
    )
    X = (
        vstack([Z1[0][idx : idx + n_class, :], Z2[0][idx : idx + n_class, :]]).toarray()
        / 255
    )
    y = np.hstack([Z1[1][idx : idx + n_class], Z2[1][idx : idx + n_class]])
    y = np.hstack([Z1[1][idx : idx + n_class], Z2[1][idx : idx + n_class]])

    y[y == DIGITS[0]] = -1
    y[y == DIGITS[1]] = 1
    return X, y


def prepare_data(d=2, N=100, sparse_beta=True, noise_mag=0.2):
    X = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
    if sparse_beta:
        beta_true = np.zeros(d)  # np.random.rand(d)
        beta_true[0] = 1
    else:
        beta_true = np.random.rand(d)

    y = np.zeros(N)
    # add a standard normal noise when we have 2 norm
    # shold
    scores = 1 / (1 + np.exp(-beta_true @ X.T)) + noise_mag * np.random.standard_normal(
        N
    )

    y[scores >= 0.5] = 1
    y[scores < 0.5] = -1
    if d == 2:
        fig = plt.figure()
        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        # plt.tight_layout()
        fig.savefig("vanilla-lr")

    return X, y


def generate_data_LR4C(d=2, N=100, SEED=0, pnr_ratio=0.01):
    np.random.seed(SEED)
    # Generate Data
    # d = 2
    # N = 32
    beta = np.random.randn(d, 1)
    x = np.random.randn(N, d)

    y_true = x @ beta
    # y_true[y_true >= 0] = 1
    # y_true[y_true < 0] = -1
    y = y_true + pnr_ratio * np.random.randn(N, 1)
    label_1 = np.where(y > 0)[0]
    label_0 = np.where(y < 0)[0]
    z = np.ones(N) / N

    y[y >= 0] = 1
    y[y < 0] = -1
    beta_LR = np.linalg.inv(x.T @ x) @ (x.T @ y)
    x1 = x[:, 0]
    x2 = x[:, 1]
    if d == 2:
        fig, ax = plt.subplots()
        ax.scatter(x1[label_1], x2[label_1], s=60, color="r", alpha=0.4)
        ax.scatter(x1[label_0], x2[label_0], s=60, color="b", alpha=0.4)

        x_plot = np.arange(np.min(x1), np.max(x1), 0.1)
        y_plot1 = -(beta_LR[0] * x_plot) / beta_LR[1]
        ax.plot(x_plot, y_plot1, linewidth=1.5, color=[0, 0.4471, 0.7412])
        ax.legend(["Positive", "Negative", r"$\beta^Tx=0$"], loc="upper left")
        ax.set_xlabel(r"$x_1$", fontsize=14)
        ax.set_ylabel(r"$x_2$", fontsize=14)
        ax.set_title(r"$\delta=0$, Outcome Perturbation", fontsize=14)
        ax.grid(True, which="both", alpha=0.2)
        # ax.set_axisbelow(True)
        # ax.set_aspect('equal', 'box')
        plt.show()
        fig.savefig("synthetic_data.png")

    return x, y


class ClassificationDataGenerator:
    def __init__(self, beta):
        self.beta = np.array(beta)
        self.dim = self.beta.size

    def generate_data_corrupt(self, data_num, epsilon=1, seed=0, plotting=False):
        np.random.seed(seed)
        d = self.dim
        beta_true = self.beta
        X = np.random.randn(data_num, d)

        norms = np.random.uniform(0, 1, data_num)  # bounded_noise =
        points = np.random.normal(0, 1, (data_num, d))
        normalized_points = (
            points
            / np.linalg.norm(points, axis=1)[:, np.newaxis]
            * norms[:, np.newaxis]
        )
        X_corrupt = X + normalized_points * epsilon

        prob = 1 / (1 + np.exp(X.dot(beta_true)))

        y = (0.5 < prob).astype(np.int_)
        y = y * 2 - 1
        y_corrupt = y
        if plotting:
            X = X_corrupt
            label_1 = np.where(y > 0)[0]
            label_0 = np.where(y < 0)[0]
            x1 = X[:, 0]
            x2 = X[:, 1]
            fig, ax = plt.subplots()
            ax.scatter(x1[label_1], x2[label_1], s=60, color="r", alpha=0.4)
            ax.scatter(x1[label_0], x2[label_0], s=60, color="b", alpha=0.4)

            x_plot = np.arange(np.min(x1), np.max(x1), 0.1)
            y_plot1 = -(self.beta[0] * x_plot) / self.beta[1]
            ax.plot(x_plot, y_plot1, linewidth=1.5, color=[0, 0.4471, 0.7412])
            ax.legend(["Positive", "Negative", r"$\beta^Tx=0$"], loc="upper left")
            ax.set_xlabel(r"$x_1$", fontsize=14)
            ax.set_ylabel(r"$x_2$", fontsize=14)
            ax.set_title(r"$\delta=0$, Outcome Perturbation", fontsize=14)
            ax.grid(True, which="both", alpha=0.2)
            # ax.set_axisbelow(True)
            # ax.set_aspect('equal', 'box')
            plt.show()
            fig.savefig("synthetic_data.png")
        return X_corrupt, y_corrupt, X, y

    def generate(self, data_num, seed=0, plotting=False):
        np.random.seed(seed)
        X = np.random.randn(data_num, self.dim)
        prob = 1 / (1 + np.exp(X.dot(self.beta)))
        y = (np.random.rand(data_num) < prob).astype(np.int_)
        y = y * 2 - 1
        if plotting:
            label_1 = np.where(y > 0)[0]
            label_0 = np.where(y < 0)[0]
            x1 = X[:, 0]
            x2 = X[:, 1]
            fig, ax = plt.subplots()
            ax.scatter(x1[label_1], x2[label_1], s=60, color="r", alpha=0.4)
            ax.scatter(x1[label_0], x2[label_0], s=60, color="b", alpha=0.4)

            x_plot = np.arange(np.min(x1), np.max(x1), 0.1)
            y_plot1 = -(self.beta[0] * x_plot) / self.beta[1]
            ax.plot(x_plot, y_plot1, linewidth=1.5, color=[0, 0.4471, 0.7412])
            ax.legend(["Positive", "Negative", r"$\beta^Tx=0$"], loc="upper left")
            ax.set_xlabel(r"$x_1$", fontsize=14)
            ax.set_ylabel(r"$x_2$", fontsize=14)
            ax.set_title(r"$\delta=0$, Outcome Perturbation", fontsize=14)
            ax.grid(True, which="both", alpha=0.2)

            plt.show()
            fig.savefig("synthetic_data.png")

        return X, y
