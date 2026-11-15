# Tasks 1 e 2 - Diagnóstico, baselines, SARIMA e modelos com features

**Séries Temporais - FGV EMAp - 2026/2**

Bryan Santos Monteiro.
João Vitor Tomaz Alves Ferreira.
Nicholas Costa.
Roger Vinícius Pereira Augusto.
Sofia Azeredo de Moura Monteiro.
Vinício Vasconcelos Muniz Deusdará.

## Como rodar


```bash
python -m pip install -r requirements.txt && python main.py
```

O comando lê `dados/`, gera as figuras de diagnóstico em `figuras/` e grava `previsoes_validacao.csv` e `metricas.csv` na raiz

Para conferir a entrega com o script de correção:

```bash
python checks/check_task1.py --submission .
python checks/check_task2.py --submission .
```

## Dados e split

| Arquivo | Período | Uso |
|---------|---------|-----|
| `dados/treino.csv` | 2011-01-29 a 2016-03-27 | ajuste dos modelos |
| `dados/validacao.csv` | 2016-03-28 a 2016-04-24 | avaliação (horizonte de 28 dias) |
| `dados/holdout_datas.csv` | 2016-04-25 a 2016-05-22 | só datas, não usado nesta Task |

Os modelos são ajustados só no treino. A validação entra apenas no cálculo das métricas

## Modelos

- **Baselines:** `media`, `naive` (último valor), `naive_sazonal` (repete a última semana, m = 7) e `drift`
- **SARIMA(1,0,0)(0,1,1)₇:** uma diferença sazonal semanal, AR(1) e MA sazonal(1), implementado em [`src/sarima.py`](src/sarima.py). Os coeficientes são estimados por grid search minimizando a soma dos quadrados dos resíduos
- **Linear (Task 2):** regressão por mínimos quadrados, um ajuste por série, estratégia direta, em [`src/linear.py`](src/linear.py). Colunas: dia da semana, Natal e véspera, 3 pares de Fourier anuais, `snap_CA`, indicador de evento e preço da série padronizado com média e desvio do treino. Janela de 365 dias, K = 3 e sem tendência, escolhidos em origens dentro do treino (`python -m src.linear`)
- **Árvore (Task 2):** árvore de regressão do scikit-learn, um ajuste por série, estratégia direta, em [`src/arvore.py`](src/arvore.py). Usa as mesmas colunas e a mesma janela de 365 dias do linear, com o preço sem padronizar. Profundidade máxima 8 e no mínimo uma observação por folha, escolhidos nas mesmas origens do linear (`python -m src.arvore`)
- **Rede (Task 2):** MLP do scikit-learn (`MLPRegressor`, solver L-BFGS), um ajuste por série, estratégia direta, em [`src/rede.py`](src/rede.py). Usa as mesmas colunas do linear, com X e y padronizados pela média e desvio da janela de treino, e a previsão é a média de cinco sementes. Janela de 730 dias, uma camada oculta de 16 neurônios e `alpha` = 10, escolhidos nas mesmas origens do linear (`python -m src.rede`, leva ~10 min)

## Métricas

MAE, RMSE e MASE nos 28 dias de validação. O MASE usa como escala o MAE in-sample do naive sazonal (m = 7) no treino

| Série | Melhor baseline (MASE) | SARIMA (MASE) | Linear (MASE) | Árvore (MASE) | Rede (MASE) |
|-------|------------------------|---------------|---------------|---------------|-------------|
| `store_total` | naive_sazonal (1,056) | 0,724 | 0,441 | 0,646 | **0,408** |
| `FOODS` | naive_sazonal (1,005) | 0,622 | 0,564 | 0,582 | **0,346** |
| `HOBBIES` | naive_sazonal (1,120) | 0,864 | 0,816 | 0,970 | **0,805** |

O SARIMA supera os quatro baselines nas três séries. O linear supera o naive sazonal e o SARIMA nas três séries, em MAE, RMSE e MASE. A árvore supera o naive sazonal nas três séries e métricas e o SARIMA em `store_total` e `FOODS` no MAE e no MASE (no RMSE as duas quase empatam), mas perde para o SARIMA nas três métricas em `HOBBIES` e fica atrás do linear nas três séries. A rede tem o menor MAE e MASE nas três séries e o menor RMSE em `store_total` e `FOODS`. O ganho sobre o linear é grande em `FOODS` e pequeno em `HOBBIES` (MASE 0,805 contra 0,816), onde o linear ainda tem RMSE menor (86,8 contra 89,9). Os valores completos estão em [`metricas.csv`](metricas.csv)

## Estrutura

```
main.py              # pipeline completo (diagnóstico, previsões, métricas)
relatorio.ipynb      # relatório com o diagnóstico das séries
src/
  config.py          # período sazonal (7)
  dados.py           # leitura e checagem do split
  diagnostico.py     # ACF, PACF e figuras
  baselines.py       # média, naive, naive sazonal, drift
  sarima.py          # SARIMA(1,0,0)(0,1,1)7
  linear.py          # regressão linear com features (Task 2)
  arvore.py          # árvore de regressão (Task 2)
  rede.py            # rede neural MLP (Task 2)
  metricas.py        # MAE, RMSE, MASE
figuras/             # série + ACF + PACF de cada série (treino)
checks/              # script público de correção
AI_USAGE.md          # uso de IA generativa
```
