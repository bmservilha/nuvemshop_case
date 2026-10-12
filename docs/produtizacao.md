# Produtização do modelo de fraude

Esboço de como colocar o LightGBM campeão (notebooks 3 e 4) atrás de um endpoint e integrá-lo a um motor de decisão em tempo quase real (NRT).

## Visão geral

```
 Checkout ──► Motor de decisão ──► Serviço de features ──┬─► Feature store online (Redis/DynamoDB)
                 │       ▲                               │     ├─ perfil do cliente (batch diário)
                 │       │                               │     ├─ perfil da loja (batch diário)
                 │       │                               │     └─ contadores de velocidade (streaming)
                 │       │                               └─► features da requisição (valor, horário)
                 ▼       │
           Endpoint do modelo (score)
                 │
                 ▼
     Regras + threshold ──► aprovar / revisar / bloquear
                 │
                 ▼
     Log de decisão (Kafka) ──► data lake ──► monitoramento, re-treino, chargebacks (rótulos)
```

Meta de latência: p99 < 100 ms para a cadeia inteira; o modelo sozinho (5 variáveis, ~100 árvores) responde em menos de 5 ms.

## 1. Endpoint do modelo

* **Empacotamento**: o modelo já está registrado no MLflow. A versão campeã é promovida no Model Registry (alias `champion`), junto com o threshold de decisão (0,113), a lista de variáveis e o mapeamento vertical → grupo de risco como artefatos da mesma versão. Assim modelo, pré-processamento e threshold são versionados juntos.
* **Serviço**: um container stateless (FastAPI ou `mlflow models serve`) com uma rota `POST /score` que recebe as variáveis já montadas e devolve `score`, `model_version` e, opcionalmente, os 3 maiores valores SHAP (TreeExplainer é barato com 5 variáveis) para o analista de revisão manual.
* **Infra**: Kubernetes/ECS com autoscaling por CPU ou RPS, no mínimo duas réplicas em zonas diferentes. Alternativa gerenciada: endpoint do SageMaker.
* **Implantação segura**: primeiro em *shadow* (pontua o tráfego real e loga, sem decidir), depois canário (5% → 50% → 100%) comparando taxa de bloqueio e distribuição do score com o modelo/regra atual. Rollback = trocar o alias no registry.

## 2. Features: offline e online

O ponto crítico é **paridade treino/produção**: a mesma definição de cada variável é usada para gerar a base de treino (point-in-time) e para servir em produção. Uma feature store (Feast, Tecton, SageMaker Feature Store) organiza isso.

| Variável | Origem | Atualização | Como chega na decisão |
|---|---|---|---|
| `amount`, `high_amount` | requisição | — | calculada no motor |
| `night` (22h–03h UTC) | requisição (timestamp) | — | calculada no motor, sempre em UTC |
| `vertical_group` | **perfil da loja** (offline) | batch diário | lookup por `store_id` na store online |
| `amount_to_customer_median_ratio_30d` | **perfil do cliente** (offline) | batch diário | lookup da mediana 30d por `customer_id`; a razão é calculada no motor com o valor da transação |
| `txns_prev_1h` | **velocidade** (online) | streaming, por evento | contador por `customer_id` em janela deslizante de 1h |

**Offline (perfil do cliente e da loja)**: um job diário (Spark/dbt sobre o data lake) calcula a mediana de valor do cliente em 30 dias, a vertical/grupo de risco da loja e outros atributos de perfil. O resultado vai para a store offline (para treino) e é materializado na store online (chave-valor, leitura em ~1 ms). Cliente sem histórico → valor nulo, que o LightGBM já trata nativamente (mesmo comportamento do treino).

**Online (velocidade)**: cada transação aprovada ou tentada é publicada num tópico Kafka. Um consumidor (Flink ou Kafka Streams) mantém, por cliente, a contagem de transações na última hora e grava na store online. Duas precauções:

* a janela é `[t − 1h, t)`, **sem a transação atual**, igual ao treino;
* se o streaming atrasar, o motor pode contar direto no Redis (sorted set com timestamps por cliente: `ZADD` + `ZCOUNT`), uma solução simples e com consistência forte.

**Horário** é derivado do timestamp da própria requisição, sem consulta externa.

## 3. Motor de decisão NRT

Fluxo síncrono para cada transação:

1. **Recebe** a transação do checkout (`transaction_id`, `customer_id`, `store_id`, `amount`, `created_at`).
2. **Enriquece**: busca em paralelo o perfil do cliente, o perfil da loja e o contador de velocidade na store online (timeout curto, ~20 ms) e calcula as variáveis da requisição.
3. **Regras duras** antes do modelo: listas de bloqueio/permissão, limites regulatórios, regras de velocidade extremas (ex.: > 8 transações/hora → bloqueio direto).
4. **Pontua** chamando o endpoint do modelo.
5. **Decide** com o threshold versionado: `score ≥ 0,113` → bloquear. Uma faixa intermediária pode ir para revisão manual, se a operação comportar.
6. **Responde** ao checkout e **publica** o evento de decisão (features usadas, score, versão do modelo, decisão) no Kafka. Esse log é a base do monitoramento e do re-treino, e evita o *training-serving skew*: o modelo seguinte é treinado com as features exatamente como foram servidas.

**Fallback**: se o endpoint ou a store online falharem ou estourarem o timeout, o motor cai para a regra atual (`amount > 200` combinada com vertical/horário) em vez de travar o checkout. A taxa de fallback é monitorada.

## 4. Monitoramento e ciclo de vida

* **Operacional**: latência p50/p99, erros, taxa de fallback, freshness das features (idade do último batch, atraso do streaming).
* **Dados**: PSI/drift das 5 variáveis e do score frente ao período de treino; proporção de nulos na razão de 30d.
* **Negócio**: taxa de bloqueio diária, taxa de revisão e, com o atraso do chargeback (~1 mês, como os gaps do split temporal), recall, precisão e custo realizado.
* **Calibração**: razão score médio / taxa de fraude observada. O notebook 4 mostrou que ela sobe (1,0 no treino → 1,5 no teste) com a queda da taxa de fraude. Recalibração Platt periódica numa janela recente, se o score for usado como probabilidade.
* **Re-treino**: mensal ou disparado por drift/queda de desempenho, sempre com split temporal e gap de maturação. O novo modelo passa por shadow → canário antes de virar `champion`, e o threshold é reotimizado pelo custo a cada versão.
