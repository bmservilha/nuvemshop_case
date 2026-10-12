"""P&L da decisão de bloqueio.

FN (fraude aprovada):   amount + taxa de chargeback
FP (legítima bloqueada): margem * amount + custo de análise manual
TP e TN: custo zero (fraude evitada / receita gerada)
"""

import numpy as np
import pandas as pd

CHARGEBACK_FEE = 15.0
MARGIN = 0.10
REVIEW_COST = 5.0


def decision_pnl(
    y_true,
    blocked,
    amount,
    chargeback_fee: float = CHARGEBACK_FEE,
    margin: float = MARGIN,
    review_cost: float = REVIEW_COST,
) -> dict[str, float]:
    """Matriz de confusão e custo total (R$) de uma decisão de bloqueio."""
    y = np.asarray(y_true).astype(bool)
    b = np.asarray(blocked).astype(bool)
    a = np.asarray(amount, dtype=float)

    fn = y & ~b
    fp = ~y & b
    fn_cost = (a[fn] + chargeback_fee).sum()
    fp_cost = (margin * a[fp] + review_cost).sum()
    approve_all_cost = (a[y] + chargeback_fee).sum()
    total_cost = fn_cost + fp_cost

    return {
        "n_txns": len(y),
        "block_rate": b.mean(),
        "tp": int((y & b).sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "tn": int((~y & ~b).sum()),
        "recall": (y & b).sum() / max(y.sum(), 1),
        "precision": (y & b).sum() / max(b.sum(), 1),
        "fraud_amount_blocked_pct": a[y & b].sum() / max(a[y].sum(), 1e-9),
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "total_cost": total_cost,
        "savings_vs_approve_all": approve_all_cost - total_cost,
    }


def threshold_cost_curve(
    y_true,
    y_score,
    amount,
    chargeback_fee: float = CHARGEBACK_FEE,
    margin: float = MARGIN,
    review_cost: float = REVIEW_COST,
) -> pd.DataFrame:
    """Custo total para cada threshold candidato (bloqueia se score >= threshold).

    Os candidatos são os scores únicos observados, mais um threshold acima do
    máximo (aprovar tudo).
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, dtype=float)
    a = np.asarray(amount, dtype=float)

    order = np.argsort(-s, kind="stable")
    s, y, a = s[order], y[order], a[order]

    # Custo de bloquear cada transação em relação a aprová-la:
    # fraude bloqueada economiza amount + fee; legítima bloqueada custa margem + análise.
    approve_all_cost = (a[y] + chargeback_fee).sum()
    delta = np.where(y, -(a + chargeback_fee), margin * a + review_cost)

    # Bloquear o top-k: considera apenas o último índice de cada score (empates)
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    k = np.flatnonzero(last_of_tie) + 1

    cum_delta = np.cumsum(delta)[k - 1]
    cum_fraud = np.cumsum(y)[k - 1]
    n_fraud = y.sum()

    curve = pd.DataFrame(
        {
            "threshold": s[k - 1],
            "n_blocked": k,
            "block_rate": k / len(s),
            "recall": cum_fraud / max(n_fraud, 1),
            "precision": cum_fraud / k,
            "total_cost": approve_all_cost + cum_delta,
        }
    )
    approve_all = pd.DataFrame(
        {
            "threshold": [np.inf],
            "n_blocked": [0],
            "block_rate": [0.0],
            "recall": [0.0],
            "precision": [np.nan],
            "total_cost": [approve_all_cost],
        }
    )
    curve = pd.concat([approve_all, curve], ignore_index=True)
    curve["savings_vs_approve_all"] = approve_all_cost - curve["total_cost"]
    return curve


def optimal_threshold(curve: pd.DataFrame) -> float:
    """Threshold de menor custo total na curva."""
    return float(curve.loc[curve["total_cost"].idxmin(), "threshold"])
