import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """KS: máxima distância entre as distribuições acumuladas de fraude e não fraude."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, y_score) -> dict[str, float]:
    """PR AUC (average precision), ROC AUC e KS."""
    return {
        "n_txns": len(y_true),
        "fraud_rate": float(np.mean(y_true)),
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
    }


def metrics_by_split(splits: dict[str, tuple]) -> pd.DataFrame:
    """Recebe {nome: (y_true, y_score)} e devolve uma tabela de métricas por base."""
    return pd.DataFrame(
        {name: classification_metrics(y, s) for name, (y, s) in splits.items()}
    ).T
