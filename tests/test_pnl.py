import numpy as np
import pytest

from utils.pnl import CHARGEBACK_FEE, MARGIN, decision_pnl, optimal_threshold, threshold_cost_curve

FEE, MRG = 15.0, 0.10


@pytest.fixture
def txns():
    # fraudes: 100 (score .9), 50 (score .3) | legítimas: 200 (score .8), 10 (score .1)
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.3, 0.1])
    a = np.array([100.0, 200.0, 50.0, 10.0])
    return y, s, a


def test_defaults():
    assert (CHARGEBACK_FEE, MARGIN) == (FEE, MRG)


class TestDecisionPnl:
    def test_confusion_and_costs(self, txns):
        y, s, a = txns
        p = decision_pnl(y, s >= 0.5, a, FEE, MRG)  # bloqueia 100 (TP) e 200 (FP)
        assert (p["tp"], p["fp"], p["fn"], p["tn"]) == (1, 1, 1, 1)
        assert p["fn_cost"] == pytest.approx(50 + FEE)
        assert p["fp_cost"] == pytest.approx(MRG * 200)
        assert p["total_cost"] == pytest.approx(65 + 20)
        assert p["block_rate"] == pytest.approx(0.5)
        assert p["recall"] == pytest.approx(0.5)
        assert p["precision"] == pytest.approx(0.5)
        assert p["fraud_amount_blocked_pct"] == pytest.approx(100 / 150)
        assert p["savings_vs_approve_all"] == pytest.approx((150 + 2 * FEE) - 85)

    def test_approve_all_has_zero_savings(self, txns):
        y, _, a = txns
        p = decision_pnl(y, np.zeros(4, bool), a, FEE, MRG)
        assert p["total_cost"] == pytest.approx(150 + 2 * FEE)
        assert p["savings_vs_approve_all"] == pytest.approx(0)
        assert p["precision"] == 0  # sem bloqueios, sem divisão por zero

    def test_block_all_costs_only_margin(self, txns):
        y, _, a = txns
        p = decision_pnl(y, np.ones(4, bool), a, FEE, MRG)
        assert p["fn_cost"] == 0
        assert p["total_cost"] == pytest.approx(MRG * 210)
        assert p["recall"] == pytest.approx(1.0)

    def test_no_fraud(self):
        p = decision_pnl([0, 0], [1, 0], [100.0, 50.0], FEE, MRG)
        assert p["recall"] == 0
        assert p["fraud_amount_blocked_pct"] == 0
        assert p["total_cost"] == pytest.approx(10.0)

    def test_confusion_sums_to_n(self, txns):
        y, s, a = txns
        p = decision_pnl(y, s >= 0.2, a)
        assert p["tp"] + p["fp"] + p["fn"] + p["tn"] == p["n_txns"] == 4


class TestThresholdCostCurve:
    def test_shape_and_first_row(self, txns):
        curve = threshold_cost_curve(*txns, FEE, MRG)
        assert len(curve) == 5  # aprova tudo + 4 scores distintos
        first = curve.iloc[0]
        assert np.isinf(first["threshold"]) and first["n_blocked"] == 0
        assert first["total_cost"] == pytest.approx(150 + 2 * FEE)
        assert curve["threshold"].is_monotonic_decreasing
        assert curve["recall"].is_monotonic_increasing

    def test_matches_decision_pnl(self, txns):
        y, s, a = txns
        curve = threshold_cost_curve(y, s, a, FEE, MRG)
        for _, row in curve.iloc[1:].iterrows():
            p = decision_pnl(y, s >= row["threshold"], a, FEE, MRG)
            assert row["total_cost"] == pytest.approx(p["total_cost"])
            assert row["savings_vs_approve_all"] == pytest.approx(p["savings_vs_approve_all"])
            assert row["recall"] == pytest.approx(p["recall"])
            assert row["precision"] == pytest.approx(p["precision"])
            assert row["block_rate"] == pytest.approx(p["block_rate"])

    def test_ties_grouped(self):
        curve = threshold_cost_curve([1, 0, 0], [0.5, 0.5, 0.1], [10.0, 10.0, 10.0])
        assert list(curve["n_blocked"]) == [0, 2, 3]

    def test_matches_decision_pnl_random(self):
        rng = np.random.default_rng(0)
        y = rng.random(500) < 0.05
        s = np.round(rng.random(500), 2)  # força empates
        a = rng.exponential(100, 500)
        curve = threshold_cost_curve(y, s, a)
        for thr, cost in curve.iloc[1:][["threshold", "total_cost"]].to_numpy():
            assert cost == pytest.approx(decision_pnl(y, s >= thr, a)["total_cost"])


def test_optimal_threshold(txns):
    curve = threshold_cost_curve(*txns, FEE, MRG)
    # custos: aprova tudo 180 | >=.9: 65 | >=.8: 85 | >=.3: 20 (só FP de 200) | tudo: 21
    assert list(curve["total_cost"]) == pytest.approx([180, 65, 85, 20, 21])
    assert optimal_threshold(curve) == pytest.approx(0.3)
