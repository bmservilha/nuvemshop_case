import numpy as np
import pandas as pd
import pytest

from utils.metrics import classification_metrics, ks_statistic, metrics_by_split


@pytest.fixture
def perfect():
    return np.array([0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.8, 0.9])


@pytest.fixture
def mixed():
    # uma fraude abaixo de uma legítima
    return np.array([0, 0, 1, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.9])


class TestKS:
    def test_perfect_separation(self, perfect):
        assert ks_statistic(*perfect) == pytest.approx(1.0)

    def test_inverted_scores(self, perfect):
        y, s = perfect
        assert ks_statistic(y, -s) == pytest.approx(0.0)

    def test_constant_score(self):
        assert ks_statistic([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.0)

    def test_hand_computed(self, mixed):
        # threshold 0.9: tpr 0.5, fpr 0 -> 0.5; threshold 0.3: tpr 1.0, fpr 1/3 -> 0.667
        assert ks_statistic(*mixed) == pytest.approx(2 / 3)

    def test_invariant_to_monotonic_transform(self, mixed):
        y, s = mixed
        assert ks_statistic(y, np.log(s)) == pytest.approx(ks_statistic(y, s))


class TestClassificationMetrics:
    def test_keys(self, mixed):
        assert set(classification_metrics(*mixed)) == {"n_txns", "fraud_rate", "pr_auc", "roc_auc", "ks"}

    def test_perfect(self, perfect):
        m = classification_metrics(*perfect)
        assert m["n_txns"] == 5
        assert m["fraud_rate"] == pytest.approx(0.4)
        assert m["pr_auc"] == pytest.approx(1.0)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["ks"] == pytest.approx(1.0)

    def test_hand_computed(self, mixed):
        m = classification_metrics(*mixed)
        # pares (fraude, legítima) ordenados corretamente: 5 de 6
        assert m["roc_auc"] == pytest.approx(5 / 6)
        # AP = média da precisão em cada fraude (ordem .9 F, .4 L, .3 F): (1/1 + 2/3) / 2
        assert m["pr_auc"] == pytest.approx(5 / 6)

    def test_accepts_pandas(self, mixed):
        y, s = mixed
        assert classification_metrics(pd.Series(y), pd.Series(s)) == pytest.approx(classification_metrics(y, s))

    def test_single_class_is_undefined(self):
        # sklearn não levanta erro: avisa e devolve NaN
        with pytest.warns(UserWarning):
            m = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3])
        assert np.isnan(m["roc_auc"])


def test_metrics_by_split(perfect, mixed):
    table = metrics_by_split({"train": perfect, "test": mixed})
    assert list(table.index) == ["train", "test"]
    assert table.loc["train", "ks"] == pytest.approx(1.0)
    assert table.loc["test", "roc_auc"] == pytest.approx(5 / 6)
