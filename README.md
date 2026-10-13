# nuvemshop_case

Case técnico para a vaga de Senior Data Scientist na **Nuvemshop**: modelo de detecção de fraude em transações (EDA, modelagem champion/challenger e proposta de produtização).

## Setup

Requer [uv](https://docs.astral.sh/uv/) e Python ≥ 3.12.

```bash
uv sync
```

Suba o MLflow antes de rodar os notebooks de modelagem (tracking em `http://127.0.0.1:5000`, experimento `nuvemshop-fraude`):

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Em outro terminal, abra o Jupyter:

```bash
uv run jupyter lab
```

## Ordem de execução

Os dados ficam em `data/nuvem_fraud_dataset.csv`.

1. [notebooks/1_EDA_revisado.ipynb](notebooks/1_EDA_revisado.ipynb) — análise exploratória; gera `data/nuvem_fraud_dataset_pos_eda.csv`.
2. [notebooks/2_Modeling.ipynb](notebooks/2_Modeling.ipynb) — comparação de modelos, tuning (Optuna) e registro no MLflow.
3. [notebooks/3_Modelo_final.ipynb](notebooks/3_Modelo_final.ipynb) — modelo final (champion) e threshold de decisão.

Testes: `uv run pytest`

## Arquitetura

O desenho da arquitetura de produtização (champion/challenger, feature store, decisão em tempo quase real) está em [docs/produtizacao.md](docs/produtizacao.md).
