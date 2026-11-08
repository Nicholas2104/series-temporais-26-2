import itertools
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from .dados import carregar_split, SERIES
from .linear import carregar_covariaveis, montar_features, JANELA, ORIGENS_SELECAO
from .metricas import mase

MAX_DEPTH = 8
MIN_FOLHA = 1


def ajustar_e_prever(y_treino, datas_prev, cov, serie, max_depth=MAX_DEPTH, min_folha=MIN_FOLHA):
    y = y_treino.iloc[-JANELA:]
    t0 = y.index[0]
    X = montar_features(y.index, cov, serie, t0, preco_media=0, preco_desvio=1)
    X_prev = montar_features(datas_prev, cov, serie, t0, preco_media=0, preco_desvio=1)
    m = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_folha, random_state=0)
    return np.clip(m.fit(X, y).predict(X_prev), 0, None)


def arvore_forecast(treino, datas_prev, serie):
    return ajustar_e_prever(treino[serie], datas_prev, carregar_covariaveis(), serie)


def selecionar(h=28):
    treino, _ = carregar_split()
    cov = carregar_covariaveis()

    linhas = []
    for max_depth, min_folha in itertools.product([4, 6, 8, 10, None], [1, 3, 7, 14]):
        erros = {s: [] for s in SERIES}

        for o in ORIGENS_SELECAO:
            a = treino.loc[:o]
            b = treino.loc[o + pd.Timedelta(days=1): o + pd.Timedelta(days=h)]
            for s in SERIES:
                yhat = ajustar_e_prever(a[s], b.index, cov, s, max_depth, min_folha)
                erros[s].append(mase(b[s], yhat, a[s]))

        linha = {"max_depth": max_depth or "sem limite", "min_folha": min_folha}
        linha.update({s: round(float(np.mean(v)), 3) for s, v in erros.items()})
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    df["media"] = df[SERIES].mean(axis=1).round(3)
    return df.sort_values("media")


if __name__ == "__main__":
    print(selecionar().to_string(index=False))
