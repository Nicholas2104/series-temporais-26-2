import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import PERIODO

def acf(x, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = np.sum(x**2)
    r = np.empty(nlags + 1)
    r[0] = 1.0
    for k in range(1, nlags + 1):
        r[k] = np.sum(x[k:] * x[:-k]) / denom
    return r

def pacf(x, nlags: int) -> np.ndarray:
    r = acf(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    p = np.ones(nlags + 1)
    if nlags >= 1:
        phi[1, 1] = r[1]
        p[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - np.sum(phi[k - 1, 1:k] * r[k - 1:0:-1])
        den = 1.0 - np.sum(phi[k - 1, 1:k] * r[1:k])
        phi[k, k] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, k - 1:0:-1]
        p[k] = phi[k, k]
    return p

def _n_obs(x) -> int:
    x = np.asarray(x, dtype=float)
    return int(np.sum(~np.isnan(x)))

def _stem_correlograma(ax, valores, n, titulo):
    lags = np.arange(len(valores))
    banda = 1.96 / np.sqrt(n)
    ax.axhspan(-banda, banda, color="tab:blue", alpha=0.12)
    ax.axhline(0, color="black", lw=0.8)
    ax.vlines(lags[1:], 0, valores[1:], color="tab:blue")
    ax.plot(lags[1:], valores[1:], "o", color="tab:blue", ms=4)
    ax.set_title(titulo)
    ax.set_xlabel("lag (dias)")
    ax.margins(x=0.01)

def figura_diagnostico(y: pd.Series, nome: str, m: int = PERIODO, nlags: int = 35, salvar: str | None = None):
    n = _n_obs(y)
    a = acf(y, nlags)
    p = pacf(y, nlags)

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_serie = fig.add_subplot(gs[0, :])
    ax_acf = fig.add_subplot(gs[1, 0])
    ax_pacf = fig.add_subplot(gs[1, 1])

    ax_serie.plot(y.index, y.values, lw=0.7, color="tab:blue")
    ax_serie.set_title(f"{nome} — série diária (treino)")
    ax_serie.set_xlabel("data")
    ax_serie.set_ylabel("vendas")
    ax_serie.margins(x=0.01)

    _stem_correlograma(ax_acf, a, n, f"ACF — {nome}")
    _stem_correlograma(ax_pacf, p, n, f"PACF — {nome}")

    for ax in (ax_acf, ax_pacf):
        for k in range(m, nlags + 1, m):
            ax.axvline(k, color="tab:red", ls=":", lw=0.8, alpha=0.6)

    fig.suptitle(f"Diagnóstico — {nome}", fontsize=14, fontweight="bold")

    if salvar is not None:
        pasta = os.path.dirname(salvar)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        fig.savefig(salvar, dpi=120, bbox_inches="tight")

    return fig

def resumo_zeros(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for col in df.columns:
        for data in df.index[df[col] == 0]:
            linhas.append({"series": col, "date": data.date(), "dia_semana": data.day_name()})
                           
    return pd.DataFrame(linhas)