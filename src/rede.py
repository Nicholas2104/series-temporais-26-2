import itertools
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from .dados import carregar_split, SERIES
from .linear import carregar_covariaveis, montar_features, PRECO, ORIGENS_SELECAO
from .metricas import mase

JANELA = 730
CAMADAS = (16,)
ALPHA = 10.0
# média de algumas inicializações, a rede é sensível à semente
SEMENTES = (0, 1, 2, 3, 4)


def ajustar_e_prever(y_treino, datas_prev, cov, serie, janela=JANELA, camadas=CAMADAS, alpha=ALPHA):
    y = y_treino.iloc[-janela:] if janela else y_treino
    t0 = y.index[0]

    preco_tr = cov.reindex(y.index)[PRECO[serie]].astype(float).ffill().bfill()
    desvio = preco_tr.std() if preco_tr.std() > 0 else 1.0
    X = montar_features(y.index, cov, serie, t0, preco_tr.mean(), desvio).drop(columns="const")
    X_prev = montar_features(datas_prev, cov, serie, t0, preco_tr.mean(), desvio).drop(columns="const")

    # padronização com média e desvio do treino
    mx, sx = X.mean(), X.std().replace(0, 1.0)
    Xz, Xz_prev = ((X - mx) / sx).to_numpy(), ((X_prev - mx) / sx).to_numpy()
    my, sy = y.mean(), y.std()
    yz = ((y - my) / sy).to_numpy(dtype=float)

    prev = np.zeros(len(Xz_prev))
    for s in SEMENTES:
        m = MLPRegressor(hidden_layer_sizes=camadas, alpha=alpha, solver="lbfgs",
                         max_iter=2000, random_state=s)
        prev += m.fit(Xz, yz).predict(Xz_prev)
    prev = prev / len(SEMENTES) * sy + my
    return np.clip(prev, 0, None)


def rede_forecast(treino, datas_prev, serie):
    return ajustar_e_prever(treino[serie], datas_prev, carregar_covariaveis(), serie)


def selecionar(h=28):
    treino, _ = carregar_split()
    cov = carregar_covariaveis()

    linhas = []
    for janela, camadas, alpha in itertools.product([365, 730], [(8,), (16,), (32, 16)], [0.01, 0.1, 1.0, 10.0]):
        erros = {s: [] for s in SERIES}

        for o in ORIGENS_SELECAO:
            a = treino.loc[:o]
            b = treino.loc[o + pd.Timedelta(days=1): o + pd.Timedelta(days=h)]
            for s in SERIES:
                yhat = ajustar_e_prever(a[s], b.index, cov, s, janela, camadas, alpha)
                erros[s].append(mase(b[s], yhat, a[s]))

        linha = {"janela": janela, "camadas": camadas, "alpha": alpha}
        linha.update({s: round(float(np.mean(v)), 3) for s, v in erros.items()})
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    df["media"] = df[SERIES].mean(axis=1).round(3)
    return df.sort_values("media")


if __name__ == "__main__":
    print(selecionar().to_string(index=False))
