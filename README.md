# Task 1 - Diagnóstico, baselines e ARIMA/SARIMA

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

## Métricas

MAE, RMSE e MASE nos 28 dias de validação. O MASE usa como escala o MAE in-sample do naive sazonal (m = 7) no treino

| Série | Melhor baseline (MASE) | SARIMA (MASE) |
|-------|------------------------|---------------|
| `store_total` | naive_sazonal (1,056) | **0,724** |
| `FOODS` | naive_sazonal (1,005) | **0,622** |
| `HOBBIES` | naive_sazonal (1,120) | **0,864** |

O SARIMA supera os quatro baselines nas três séries. Os valores completos estão em [`metricas.csv`](metricas.csv)

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
  metricas.py        # MAE, RMSE, MASE
figuras/             # série + ACF + PACF de cada série (treino)
checks/              # script público de correção
AI_USAGE.md          # uso de IA generativa
```
