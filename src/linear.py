from pathlib import Path
import numpy as np
import pandas as pd
import itertools
from .dados import carregar_split, SERIES
from .metricas import mase

DADOS = Path(__file__).resolve().parent.parent / "dados"

# dias finais do treino usados no ajuste
JANELA = 365  
# pares seno/cosseno da sazonalidade anual
K_FOURIER = 3  
TENDENCIA = False 

PRECO = {
    "store_total": "preco_store_total",
    "FOODS": "preco_FOODS",
    "HOBBIES": "preco_HOBBIES",
}


def carregar_covariaveis():
    cov = pd.read_csv(DADOS / "covariaveis.csv", parse_dates=["date"])
    return cov.set_index("date").sort_index()


def _evento(sub: pd.DataFrame):
    tem = pd.Series(False, index=sub.index)
    for c in ("event_type_1", "event_type_2"):
        if c in sub.columns:
            col = sub[c].astype("string").str.strip()
            tem |= col.notna() & (col != "")
    return tem.to_numpy(dtype=float)


def montar_features(datas, cov, serie, t0, preco_media, preco_desvio, k=K_FOURIER, tendencia=TENDENCIA):
    idx = pd.DatetimeIndex(datas)
    sub = cov.reindex(idx)
    X = pd.DataFrame(index=idx)
    X["const"] = 1.0

    # calendário do dia previsto, com domingo sendo a referência
    for d in range(6): 
        X[f"dow_{d}"] = (idx.dayofweek == d).astype(float)

    X["natal"] = ((idx.month == 12) & (idx.day == 25)).astype(float)
    X["vespera_natal"] = ((idx.month == 12) & (idx.day == 24)).astype(float)
    ang = 2 * np.pi * idx.dayofyear.to_numpy() / 365.25

    for j in range(1, k + 1):
        X[f"sin_{j}"] = np.sin(j * ang)
        X[f"cos_{j}"] = np.cos(j * ang)

    if tendencia:
        X["t"] = (idx - t0).days.to_numpy() / 365.25

    # covariáveis conhecidas na origem
    X["snap_CA"] = sub["snap_CA"].fillna(0).astype(float).to_numpy()
    X["evento"] = _evento(sub)
    preco = sub[PRECO[serie]].astype(float).ffill().bfill()
    X["preco_z"] = ((preco - preco_media) / preco_desvio).to_numpy()
    return X


def ajustar_e_prever(y_treino: pd.Series, datas_prev, cov, serie, janela=JANELA, k=K_FOURIER, tendencia=TENDENCIA):
    """Ajusta só em y_treino e prevê datas_prev"""
    y_fit = y_treino.iloc[-janela:] if janela else y_treino
    t0 = y_fit.index[0]

    # média e desvio do preço
    preco_tr = cov.reindex(y_fit.index)[PRECO[serie]].astype(float).ffill().bfill()
    media = preco_tr.mean()
    desvio = preco_tr.std()
    if not np.isfinite(desvio) or desvio == 0:
        desvio = 1.0

    X_fit = montar_features(y_fit.index, cov, serie, t0, media, desvio, k, tendencia)
    X_prev = montar_features(datas_prev, cov, serie, t0, media, desvio, k, tendencia)

    beta, *_ = np.linalg.lstsq(X_fit.to_numpy(), y_fit.to_numpy(dtype=float), rcond=None)
    return np.clip(X_prev.to_numpy() @ beta, 0, None)  # vendas não são negativas


def linear_forecast(treino: pd.DataFrame, datas_prev, serie: str) -> np.ndarray:
    return ajustar_e_prever(treino[serie], datas_prev, carregar_covariaveis(), serie)


# ---------------------------------------------------------------------------
# Escolha dos hiperparâmetros.
# Comparando janela de ajuste, tendência e pares de Fourier pelo MASE médio.
# ---------------------------------------------------------------------------

ORIGENS_SELECAO = pd.to_datetime(["2015-12-06", "2016-01-03", "2016-01-31", "2016-02-28"])

def selecionar(h: int = 28):
    treino, _ = carregar_split()  
    cov = carregar_covariaveis()

    linhas = []
    for janela, tendencia, k in itertools.product([None, 730, 365], [False, True], [0, 3]):
        erros = {s: [] for s in SERIES}

        for o in ORIGENS_SELECAO:
            a = treino.loc[:o]
            b = treino.loc[o + pd.Timedelta(days=1): o + pd.Timedelta(days=h)]
            for s in SERIES:
                yhat = ajustar_e_prever(a[s], b.index, cov, s, janela, k, tendencia)
                erros[s].append(mase(b[s], yhat, a[s]))

        linha = {"janela": janela or "tudo", "tendencia": tendencia, "K": k}
        linha.update({s: round(float(np.mean(v)), 3) for s, v in erros.items()})
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    df["media"] = df[SERIES].mean(axis=1).round(3)
    return df.sort_values("media")


if __name__ == "__main__":
    print(selecionar().to_string(index=False))