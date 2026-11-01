"""
Helper utilities for the assignment: data loading, train/test splitting,
evaluation metrics and plotting.

The neural network itself lives in mlp_scratch.py (pure NumPy). This file only
uses sklearn for loading the datasets, splitting, scaling and computing metrics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

from mlp_scratch import MLP

# ----------------------------------------------------------------------------
# Plot styling
# ----------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# sequential (counts) and diverging (signed weights) colour maps
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_red_blue", ["#a8302f", "#e34948", "#f0a3a2", "#f0efec", "#86b6ef", "#2a78d6", "#184f95"])


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "figure.dpi": 110,
    })


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(name):
    """Returns X, y, feature_names, class_names.

    For the breast-cancer data sklearn encodes 0 = malignant, 1 = benign.
    We flip it so that the sigmoid output is P(malignant), i.e. malignant = 1
    is the "positive" class for precision / recall.
    """
    if name == "iris":
        d = load_iris()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data, 1 - d.target, list(d.feature_names), ["benign", "malignant"]
    raise ValueError(name)


def split_and_scale(X, y, test_size, seed=42):
    """Stratified split, then z-score scaling fitted on the TRAINING set only
    (so no information from the test set leaks into training)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, scaler


def split_name(test_size):
    return f"{round((1 - test_size) * 100)}:{round(test_size * 100)}"


def arch_name(layer_sizes):
    return "-".join(map(str, layer_sizes))


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, binary):
    """Accuracy, precision, recall, F1 and confusion matrix.
    Multi-class -> macro average over classes; binary -> positive class = 1."""
    avg = "binary" if binary else "macro"
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def run_experiment(X, y, layer_sizes, output, test_size, lr, epochs, seed=42, split_seed=42):
    """Split + scale + train + evaluate one architecture. Returns a result dict."""
    X_tr, X_te, y_tr, y_te, _ = split_and_scale(X, y, test_size, split_seed)
    model = MLP(layer_sizes, output=output, seed=seed)
    model.fit(X_tr, y_tr, epochs=epochs, lr=lr, X_test=X_te, y_test=y_te)
    binary = output == "sigmoid"
    return {
        "arch": arch_name(layer_sizes), "split": split_name(test_size), "model": model,
        "n_train": len(y_tr), "n_test": len(y_te),
        "y_test": y_te, "y_pred_test": model.predict(X_te),
        "train": evaluate(y_tr, model.predict(X_tr), binary),
        "test": evaluate(y_te, model.predict(X_te), binary),
    }


def metrics_table(results):
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "Exp": i, "Architecture": r["arch"], "Split": r["split"],
            "Train / Test": f"{r['n_train']} / {r['n_test']}",
            "Params": r["model"].n_params(),
            "Final train loss": r["model"].history["train_loss"][-1],
            "Train acc": r["train"]["accuracy"],
            "Test acc": r["test"]["accuracy"],
            "Precision": r["test"]["precision"],
            "Recall": r["test"]["recall"],
            "F1-score": r["test"]["f1"],
        })
    return pd.DataFrame(rows).set_index("Exp")


def weight_change_table(results):
    """Per layer: size of initial weights, of final weights and of the change."""
    rows = []
    for r in results:
        m = r["model"]
        for l in range(m.n_layers):
            rows.append({
                "Architecture": r["arch"], "Split": r["split"], "Layer": f"W{l + 1} {m.W[l].shape}",
                "||W_init||": np.linalg.norm(m.W_init[l]), "||W_final||": np.linalg.norm(m.W[l]),
                "||ΔW||": np.linalg.norm(m.W[l] - m.W_init[l]),
                "mean |ΔW|": np.abs(m.W[l] - m.W_init[l]).mean(),
                "||b_final||": np.linalg.norm(m.b[l]),
            })
    return pd.DataFrame(rows)


def _neuron_labels(model, layer, in_labels=None):
    """Row/column names for weight matrix `layer`: inputs x1.., hidden neurons
    h1.. (h2_1.. for the 2nd hidden layer etc.), outputs out1.."""
    def names(k, size):        # k = index in layer_sizes
        if k == 0:
            return in_labels if in_labels is not None else [f"x{i + 1}" for i in range(size)]
        if k == model.n_layers:
            return [f"out{i + 1}" for i in range(size)]
        prefix = "h" if model.n_layers == 2 else f"h{k}_"
        return [f"{prefix}{i + 1}" for i in range(size)]
    return names(layer, model.layer_sizes[layer]), names(layer + 1, model.layer_sizes[layer + 1])


def weights_as_frames(model, layer, in_labels=None, decimals=3):
    """Initial and final weight matrix of one layer as labelled DataFrames."""
    W0, W1 = model.W_init[layer], model.W[layer]
    rows, cols = _neuron_labels(model, layer, in_labels)
    f = lambda W: pd.DataFrame(W, index=rows, columns=cols).round(decimals)
    return f(W0), f(W1)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_loss_by_split(results, title, max_epoch=None, save=None):
    """One panel per split, one line per architecture (training loss)."""
    splits = list(dict.fromkeys(r["split"] for r in results))
    archs = list(dict.fromkeys(r["arch"] for r in results))
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, splits):
        for r in results:
            if r["split"] != sp:
                continue
            loss = r["model"].history["train_loss"][:max_epoch]
            ax.plot(np.arange(1, len(loss) + 1), loss, color=SERIES[archs.index(r["arch"])], label=r["arch"])
        ax.set_title(f"{sp} split"); ax.set_xlabel("Epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel("Training loss (log scale)")
    axes[-1].legend(title="Architecture")
    fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_train_test_loss(results, title, save=None):
    """Grid: one panel per experiment, training vs test loss."""
    n = len(results)
    cols = 3
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(4.4 * cols, 3.0 * rows_), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        h = r["model"].history
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=SERIES[0], label="train")
        ax.plot(ep, h["test_loss"], color=SERIES[1], label="test")
        ax.set_title(f"{r['arch']}  ({r['split']})", fontsize=10)
        ax.set_yscale("log"); ax.set_xlabel("Epoch")
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("Loss (log)"); axes[0].legend()
    fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(results, class_names, title, save=None):
    n = len(results)
    cols = 3
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.9 * cols, 3.5 * rows_))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        cm = r["test"]["cm"]
        ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=cm.max())
        ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                dark_cell = cm[i, j] > 0.55 * cm.max()
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if dark_cell else INK)
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=20)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{r['arch']} ({r['split']})  acc={r['test']['accuracy']:.3f}", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_weights(model, title, in_labels=None, save=None):
    """For every layer: initial weights | trained weights | change (diverging map,
    symmetric around 0 so blue = positive, red = negative, grey = ~0)."""
    L = model.n_layers
    fig, axes = plt.subplots(L, 3, figsize=(11, 1.6 + 3.0 * L), squeeze=False, layout="constrained",
                             gridspec_kw={"height_ratios": [max(w.shape[0], 5) for w in model.W]})
    for l in range(L):
        mats = [model.W_init[l], model.W[l], model.W[l] - model.W_init[l]]
        names = ["Initial", "After training", "Change (ΔW)"]
        lim = max(np.abs(m).max() for m in mats)
        rows, cols = _neuron_labels(model, l, in_labels)
        small = 6 if len(rows) > 10 else 8
        for k, (ax, M, nm) in enumerate(zip(axes[l], mats, names)):
            im = ax.imshow(M, cmap=DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
            ax.grid(False)
            ax.set_title(f"W{l + 1} {nm}  {M.shape}", fontsize=9.5)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, fontsize=small if len(cols) <= 10 else 5.5, rotation=90 if len(cols) > 10 else 0)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows if k == 0 else [], fontsize=small)
            for s in ax.spines.values():
                s.set_visible(False)
        fig.colorbar(im, ax=axes[l].tolist(), shrink=0.9, pad=0.01)
    fig.suptitle(title, fontweight="bold", color=INK)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_bars(df, metric, title, save=None, ylim=None):
    """Grouped bar chart: x = architecture, one bar per split."""
    archs = list(dict.fromkeys(df["Architecture"]))
    splits = list(dict.fromkeys(df["Split"]))
    x = np.arange(len(archs))
    w = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(archs) + 2), 3.6))
    for k, sp in enumerate(splits):
        vals = [df[(df["Architecture"] == a) & (df["Split"] == sp)][metric].iloc[0] for a in archs]
        bars = ax.bar(x + (k - (len(splits) - 1) / 2) * w, vals, w * 0.92, color=SERIES[k], label=sp, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(archs)
    ax.set_ylabel(metric)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Split", ncol=len(splits), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def format_table(df, decimals=4):
    """Round float columns for display."""
    return df.style.format({c: f"{{:.{decimals}f}}" for c in df.select_dtypes("float").columns})
