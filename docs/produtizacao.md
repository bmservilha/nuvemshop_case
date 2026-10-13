# Produtização do modelo de fraude

Como colocar em produção o esquema **champion / challenger** construído nos notebooks 2 e 3, integrado a um motor de decisão em tempo quase real (NRT).

* **Champion**: regressão logística `logreg_6`, registrada no MLflow como `nuvemshop-fraud@champion` (threshold 0,085). Decide as transações.
* **Challenger**: LightGBM com as **mesmas 6 variáveis**. Pontua todo o tráfego em paralelo, mas não decide.

Este documento foca nas decisões de desenho e seus tradeoffs, não em ferramentas. Onde há nomes de tecnologia, são exemplos. Qualquer equivalente serve.

## Visão geral

```
                         ┌──────────────── OFFLINE (batch diário, D-1) ────────────────┐
  data lake (transações, │  job de features ──► feature store offline                  │
  chargebacks)  ─────────┤                      ├─ perfil do cliente  (customer_id)    │──► base de treino
                         │                      └─ perfil do lojista  (store_id)       │    (point-in-time)
                         └──────────────────────────────┬──────────────────────────────┘
                                                        │ materialização diária
                                                        ▼
                                              store online (chave-valor)
                                                        ▲
                                                        │ contadores intradiários
                              eventos de transação ─────┘ (janelas 1h / 24h)

 Checkout ──► Motor de decisão NRT
                 1. features da requisição (valor, horário)
                 2. lookup em paralelo: perfil cliente + perfil lojista + contadores
                 3. regras duras
                 4. score CHAMPION (logreg, in-process) ──► threshold ──► aprovar / bloquear
                 5. responde ao checkout
                 │
                 └─► evento de decisão (features servidas, scores, versão) ──► log
                                    │
                                    ├─► CHALLENGER (LightGBM) pontua assíncrono ──► log
                                    └─► data lake ──► monitoramento, rótulos (chargeback), re-treino
```

## 1. Variáveis e onde cada uma é calculada

Cada uma das 6 variáveis do modelo tem uma origem natural. A divisão entre offline e online é a principal decisão de arquitetura.

| Variável | Origem | Atualização | Como chega na decisão |
|---|---|---|---|
| `high_amount` (valor > 200) | requisição | — | calculada no motor |
| `night` (22h–03h UTC) | requisição (timestamp) | — | calculada no motor, sempre em UTC |
| `risk_vertical` | **perfil do lojista** (offline) | batch diário | lookup por `store_id` |
| `amount_to_customer_median_ratio_30d_gt_5` | **perfil do cliente** (offline) | batch diário | lookup da mediana 30d por `customer_id`; a razão e a flag são calculadas no motor com o valor da transação |
| `n_high_prev_24h` | **perfil do cliente** (offline) + contador intradiário | D-1 + incremental | ver seção 2.3 |
| `txns_prev_1h` | contador de velocidade (online) | por evento | contador por `customer_id`, janela `[t − 1h, t)` |

Pelos coeficientes do notebook 3, as variáveis que **efetivamente fazem bloquear** são as comportamentais (`n_high_prev_24h`, razão sobre a mediana do cliente). `high_amount` e `night` sozinhas não passam do threshold. Isso define a prioridade de engenharia: as features de perfil do cliente são as mais críticas para a qualidade da decisão, e são justamente as que dependem de histórico.

## 2. Feature stores offline: perfil do cliente e perfil do lojista

### 2.1 Perfil do cliente (`customer_id`)

Job diário sobre o data lake, que calcula para cada cliente:

* mediana do valor das transações nos últimos 30 dias;
* número de transações de valor alto no dia anterior (base do `n_high_prev_24h`);
* atributos de apoio para modelos futuros: idade do cliente na base, nº de transações em 30d, nº de chargebacks históricos.

### 2.2 Perfil do lojista (`store_id`)

* vertical da loja e flag `risk_vertical`;
* candidatos para versões seguintes: taxa histórica de chargeback da loja, ticket médio, tempo de plataforma.

O perfil do lojista muda devagar, então um batch diário (ou até semanal) é suficiente.

### 2.3 Integração no cálculo NRT

A store offline serve dois consumidores com **a mesma definição de feature**:

1. **Treino**: a base de treino é montada com *point-in-time join*. Cada transação recebe o perfil como ele estava **antes** dela, nunca depois. É o que os notebooks fazem implicitamente com janelas `[t − janela, t)`.
2. **Produção**: o resultado do batch é materializado numa store online chave-valor (leitura em ~1 ms). O motor consulta `customer_id` e `store_id` em paralelo.

O ponto delicado é a janela de **24h**. Um perfil D-1 sozinho não enxerga as transações de hoje, e é exatamente a "rajada" intradiária que a variável captura. A solução proposta é **híbrida**:

* parte histórica (antes da meia-noite) vem do batch;
* parte do dia corrente vem de um contador incremental, atualizado a cada transação (ex.: estrutura ordenada por timestamp por cliente na própria store online);
* o motor soma as duas e descarta o que estiver fora de `[t − 24h, t)`.

`txns_prev_1h` é puramente online, porque uma janela de 1h não faz sentido em batch.

Cliente sem histórico: a mediana é nula e a flag fica 0, como no treino (`fillna(0)` no notebook 2). Isso deve ser uma regra explícita e compartilhada entre treino e produção, não um detalhe de código.

### Tradeoffs das features

| Decisão | Ganho | Custo / risco |
|---|---|---|
| Perfil por **batch diário** | simples, barato, fácil de reproduzir no treino | feature até 24h desatualizada; cliente novo do dia não tem perfil |
| Contadores em **streaming** | pega a rajada em tempo real | mais infraestrutura, eventos atrasados ou duplicados, mais difícil de reproduzir no treino |
| **Híbrido** (D-1 + incremental) | frescor onde importa (24h, 1h), batch onde não importa (30d, lojista) | duas fontes para somar; risco de contar duas vezes ou perder eventos na virada do dia |
| Calcular a razão/flag **no motor** em vez de pré-calcular | usa o valor da transação atual | a regra (`> 5`) fica no código do motor e precisa ser versionada junto com o modelo |
| Uma feature store dedicada vs. tabelas + cache | paridade treino/produção garantida pela ferramenta | ferramenta pesada para 6 variáveis; no início, uma tabela versionada e um cache chave-valor resolvem, desde que a definição de cada feature seja um código único usado nos dois lados |

**Recomendação inicial**: perfis do cliente e do lojista em batch diário, contadores de 1h e 24h incrementais na store online, e uma única biblioteca de definição de features importada tanto pelo job de treino quanto pelo motor.

## 3. Motor de decisão NRT

Fluxo síncrono para cada transação:

1. **Recebe** a transação (`txn_id`, `customer_id`, `store_id`, `amount`, `created_at`).
2. **Enriquece**: lookup em paralelo do perfil do cliente, do perfil do lojista e dos contadores, com timeout curto (~20 ms). Calcula `high_amount`, `night` e a razão sobre a mediana.
3. **Regras duras**: listas de bloqueio/permissão e limites extremos de velocidade.
4. **Pontua com o champion**: a regressão logística tem 6 coeficientes e um intercepto, então dá para pontuar **dentro do próprio motor**, sem chamar outro serviço. O modelo é carregado do registry (ou do `models/logreg_champion.pkl`) junto com threshold e lista de features.
5. **Decide**: `score ≥ 0,085` → bloquear.
6. **Responde** ao checkout e **publica** o evento de decisão com as features exatamente como foram servidas, o score, a versão do modelo e a decisão.

**Fallback**: se a store online falhar ou estourar o timeout, o motor cai na regra `amount > 200` (o "Run 0" do notebook 2), em vez de travar o checkout. A taxa de fallback é monitorada. A regra é conhecida e segura, mas o notebook 2 mostra que ela tem custo negativo no teste, então o fallback precisa ser exceção.

Meta de latência: p99 < 100 ms para a cadeia inteira, dominada pelos lookups. A pontuação da regressão logística é desprezível (microssegundos).

## 4. Champion / challenger

### Como o challenger roda

O LightGBM **não fica no caminho síncrono**. Ele consome o log de decisões e pontua cada transação com as **mesmas features já servidas ao champion**. Isso garante comparação justa, pois os dois modelos veem exatamente a mesma entrada, e não acrescenta latência ao checkout.

Para cada transação, ficam registrados: score do champion, score do challenger, decisão tomada e decisão que o challenger teria tomado (com o threshold dele, otimizado por custo na validação).

### Como comparar

Os rótulos chegam com atraso (~1 mês de maturação do chargeback, o mesmo gap do split temporal). A comparação é feita por **safra madura**:

* PR-AUC, KS e recall de cada modelo;
* **custo realizado** com `decision_pnl`: o custo do challenger é contrafactual, aplicado às transações que ele teria bloqueado ou aprovado;
* análise das **discordâncias**: transações que um bloqueia e o outro não. É onde a diferença de custo aparece de fato.

Limitação: as transações bloqueadas pelo champion não geram chargeback, então não se sabe com certeza se eram fraude. Para medir o challenger sem viés, é útil reservar uma pequena fração de tráfego (ex.: 1–2% abaixo de um teto de valor) onde a decisão não é aplicada, ou é aplicada pelo challenger.

### Critério de promoção

O challenger vira champion se, por **pelo menos 2–3 safras maduras**:

* economia maior que a do champion por margem relevante (no notebook 2 a diferença foi de ~R$ 3,8 mil no teste, ~7%);
* taxa de bloqueio e precisão estáveis, sem picos em subgrupos de lojistas;
* explicabilidade aceitável para a operação (ver tradeoffs).

A promoção é trocar o alias no registry (`challenger` → `champion`). O modelo anterior vira o novo challenger, o que dá rollback imediato.

### Tradeoffs champion / challenger

| Decisão | Ganho | Custo / risco |
|---|---|---|
| **Logística como champion** | cada decisão se explica pela soma dos coeficientes (notebook 3); auditoria e contestação de lojistas simples; pontuação dentro do motor | deixa ~7% de economia na mesa frente ao LightGBM |
| **LightGBM como champion** | maior economia e menor taxa de bloqueio no teste | explicação por SHAP/LIME; serviço de modelo separado ou biblioteca nativa no motor; mais difícil de auditar |
| Challenger **assíncrono** (shadow) | zero risco e zero latência no checkout | só compara com rótulos maduros; viés de seleção nos bloqueados |
| Challenger em **canário** (decide parte do tráfego) | mede o efeito real da decisão | expõe clientes e lojistas a um modelo menos validado |
| **Mesmas features** nos dois modelos | comparação isola só o algoritmo; uma única pipeline de features | um challenger com features novas exige que elas já estejam na feature store, logadas desde o início |

A escolha da logística como champion inicial é deliberada. A diferença de economia é pequena perto do ganho de explicabilidade numa primeira versão, e o shadow do LightGBM acumula evidência em produção, não só no teste.

## 5. Monitoramento e ciclo de vida

* **Operacional**: latência p50/p99, taxa de fallback, freshness das features (idade do último batch, atraso dos contadores).
* **Dados**: drift (PSI) das 6 variáveis e dos dois scores; proporção de clientes sem perfil.
* **Negócio, por safra madura**: taxa de bloqueio, recall, precisão e custo realizado do champion e do challenger, lado a lado.
* **Threshold**: o notebook 3 mostra que o ótimo do teste (0,26) é mais alto que o da validação (0,085), por causa da queda da taxa de fraude. Reotimizar o threshold por custo a cada safra madura é mais barato que retreinar e recupera parte desses ~R$ 5 mil.
* **Re-treino**: mensal ou disparado por drift, sempre com split temporal e gap de maturação. Todo modelo novo entra como challenger antes de poder virar champion.
