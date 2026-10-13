import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.pnl import optimal_threshold

METRIC_COLS = ["pr_auc", "roc_auc", "ks", "recall", "precision", "block_rate"]


def plot_model_results(metrics: pd.DataFrame, curve: pd.DataFrame, model_name: str = "modelo"):
    """Métricas por split, curva de otimização do threshold e economia vs aprovar tudo.

    `metrics`: saída de evaluate_splits (index = split).
    `curve`: saída de threshold_cost_curve na validação.
    """
    threshold = optimal_threshold(curve)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Métricas por split
    metrics[METRIC_COLS].astype(float).T.plot.bar(ax=axes[0], rot=30)
    axes[0].set_title(f"{model_name} - métricas por split")
    axes[0].set_ylim(0, 1)
    axes[0].legend(title="split")

    # 2) Curva de otimização do threshold (validação)
    ax = axes[1]
    ax.plot(curve["block_rate"], curve["total_cost"], marker="o", label="custo total")
    ax.plot(curve["block_rate"], curve["savings_vs_approve_all"], marker="s", ls="--",
            label="economia vs aprova tudo")
    best = curve.loc[curve["total_cost"].idxmin()]
    ax.scatter(best["block_rate"], best["total_cost"], s=200, facecolors="none", edgecolors="red", lw=2,
               label=f"ótimo: thr={threshold:.3f}")
    # rótulos apenas em curvas curtas (ex.: modelos com poucos scores distintos)
    if len(curve) <= 10:
        for _, r in curve.iterrows():
            label = "aprova tudo" if np.isinf(r["threshold"]) else f"thr={r['threshold']:.3f}"
            ax.annotate(label, (r["block_rate"], r["total_cost"]), textcoords="offset points", xytext=(5, 8))
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("taxa de bloqueio")
    ax.set_ylabel("R$")
    ax.set_title("Otimização de threshold (validação)")
    ax.legend()

    # 3) Economia vs aprova tudo por split no threshold ótimo
    savings = metrics["savings_vs_approve_all"].astype(float)
    savings.plot.bar(ax=axes[2], rot=0, color=["tab:green" if v > 0 else "tab:red" for v in savings])
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title(f"Economia vs aprova tudo (thr={threshold:.3f})")
    axes[2].set_ylabel("R$")

    plt.tight_layout()
    return fig
