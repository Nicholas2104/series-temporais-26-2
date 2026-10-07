import os
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# CARREGAMENTO DOS DADOS
# Ajuste os caminhos conforme a estrutura das pastas do projeto
DADOS_DIR = Path("dados")
SUBMISSION_DIR = Path(".")

train = pd.read_csv(DADOS_DIR / "treino.csv", parse_dates=["date"])
val = pd.read_csv(DADOS_DIR / "validacao.csv", parse_dates=["date"])

SERIES = ["store_total", "FOODS", "HOBBIES"]
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")

# Filtrar a validação exatamente nas datas permitidas
val = val[(val["date"] >= VAL_START) & (val["date"] <= VAL_END)].copy()
dates_val = val["date"].values
h = len(dates_val)  # 28 dias

previsoes_list = []

# GERAÇÃO DE PREVISÕES
for s in SERIES:
    y_train = train[s].to_numpy()
    
    # Baseline: Média
    yhat_media = np.full(h, y_train.mean())
    
    # Baseline: Naive (último valor)
    yhat_naive = np.full(h, y_train[-1])
    
    # Baseline: Naive Sazonal (repetição dos últimos 7 dias)
    yhat_naive_sazonal = np.tile(y_train[-7:], int(np.ceil(h / 7)))[:h]
    
    # Baseline: Drift
    slope = (y_train[-1] - y_train[0]) / (len(y_train) - 1)
    yhat_drift = y_train[-1] + slope * np.arange(1, h + 1)
    
    # Adiciona Baselines à lista de previsões
    for mod_name, yhat in [
        ("media", yhat_media),
        ("naive", yhat_naive),
        ("naive_sazonal", yhat_naive_sazonal),
        ("drift", yhat_drift),
    ]:
        for dt, val_pred in zip(dates_val, yhat):
            previsoes_list.append({
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "series": s,
                "modelo": mod_name,
                "yhat": float(val_pred)
            })

    # Modelo SARIMA
    # Substitua pelas ordens (p,d,q)x(P,D,Q)_7 ideais para cada série se desejar
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    yhat_sarima = model_fit.forecast(steps=h)
    
    for dt, val_pred in zip(dates_val, yhat_sarima):
        previsoes_list.append({
            "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "series": s,
            "modelo": "sarima",
            "yhat": float(val_pred)
        })

df_prev = pd.DataFrame(previsoes_list)
df_prev.to_csv(SUBMISSION_DIR / "previsoes_validacao.csv", index=False)

# CÁLCULO E RECONCILIAÇÃO DE MÉTRICAS 
def calc_mase_scale(y_train_series, period=7):
    diffs = np.abs(y_train_series[period:] - y_train_series[:-period])
    return float(diffs.mean())

metricas_list = []

for s in SERIES:
    scale = calc_mase_scale(train[s].to_numpy(), period=7)
    y_true = val.set_index("date")[s]
    
    modelos_presentes = df_prev[df_prev["series"] == s]["modelo"].unique()
    
    for mod in modelos_presentes:
        sub = df_prev[(df_prev["series"] == s) & (df_prev["modelo"] == mod)].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.set_index("date")["yhat"].reindex(y_true.index)
        
        err = sub.to_numpy() - y_true.to_numpy()
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        mase = float(mae / scale)
        
        metricas_list.append({
            "series": s,
            "modelo": mod,
            "mae": mae,
            "rmse": rmse,
            "mase": mase
        })

df_metricas = pd.DataFrame(metricas_list)
df_metricas.to_csv(SUBMISSION_DIR / "metricas.csv", index=False)