import numpy as np
import pandas as pd


def iv_woe_category(
    df: pd.DataFrame,
    category: str,
    target: str = "is_fraud",
) -> pd.DataFrame:
    """
    Calcula IV, WoE, fraud rate e lift por categoria.

    Target:
        0 = Não fraude
        1 = Fraude
    """
    aux = (
        df.groupby([category, target], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )

    aux = aux.rename(columns={
        0: "n_legit",
        1: "n_fraud",
    })

    # Volume e taxa de fraude
    aux["n_txns"] = aux["n_legit"] + aux["n_fraud"]
    aux["pct_population"] = aux["n_txns"] / len(df)

    aux["fraud_rate"] = (
        aux["n_fraud"] / aux["n_txns"]
    )

    base_rate = df[target].mean()

    aux["lift"] = (
        aux["fraud_rate"] / base_rate
        if base_rate > 0 else np.nan
    )

    # Distribuições para WoE / IV
    # Smoothing para evitar log(0)
    eps = 0.5

    n_categories = len(aux)

    dist_fraud = (
        (aux["n_fraud"] + eps)
        / (aux["n_fraud"].sum() + eps * n_categories)
    )

    dist_legit = (
        (aux["n_legit"] + eps)
        / (aux["n_legit"].sum() + eps * n_categories)
    )

    aux["WOE"] = np.log(dist_fraud / dist_legit)

    aux["IV"] = (
        (dist_fraud - dist_legit) * aux["WOE"]
    )

    return aux.sort_values("lift", ascending=False)


class WoEEncoder:
    """
    Substitui cada categoria pelo WoE calculado com `iv_woe_category`.

    O WoE é aprendido apenas no `fit` (base de treino) e reaplicado
    nas demais bases. Categorias não vistas no treino recebem WoE = 0
    (neutro, equivalente à taxa média).
    """

    def __init__(self, columns: list[str], target: str = "is_fraud"):
        self.columns = columns
        self.target = target
        self.tables_: dict[str, pd.DataFrame] = {}

    def fit(self, df: pd.DataFrame) -> "WoEEncoder":
        self.tables_ = {
            col: iv_woe_category(df, col, self.target)
            for col in self.columns
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"{col}_woe": df[col].map(self.tables_[col]["WOE"]).fillna(0.0)
                for col in self.columns
            },
            index=df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def iv(self) -> pd.Series:
        """IV total de cada variável na base de treino."""
        return pd.Series(
            {col: table["IV"].sum() for col, table in self.tables_.items()},
            name="IV",
        ).sort_values(ascending=False)
