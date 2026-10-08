import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .dados import SERIES

ORDEM_MODELOS = ["media", "naive", "naive_sazonal", "drift", "sarima"]
ROTULOS = {
    "media": "média",
    "naive": "naive",
    "naive_sazonal": "naive sazonal",
    "drift": "drift",
    "sarima": "SARIMA",
}
CORES = {
    "media": "#b7c9d6",
    "naive": "#7f9aaf",
    "naive_sazonal": "#3d6f99",
    "drift": "#8d8d8d",
    "sarima": "#d85a30",
}
CORES_SERIES = {"store_total": "#1f4e79", "FOODS": "#2a9d8f", "HOBBIES": "#e9c46a"}


def carregar_metricas(caminho: str = "metricas.csv") -> pd.DataFrame:
    metricas = pd.read_csv(caminho)
    metricas["modelo"] = pd.Categorical(metricas["modelo"], ORDEM_MODELOS, ordered=True)
    metricas["series"] = pd.Categorical(metricas["series"], SERIES, ordered=True)
    return metricas.sort_values(["series", "modelo"]).reset_index(drop=True)


def tabela_metricas(metricas: pd.DataFrame) -> pd.DataFrame:
    tabela = metricas.copy()
    tabela["modelo"] = tabela["modelo"].astype(str).map(ROTULOS)
    tabela[["mae", "rmse"]] = tabela[["mae", "rmse"]].round(1)
    tabela["mase"] = tabela["mase"].round(3)
    return tabela


def _barras(ax, sub: pd.DataFrame, metrica: str, decimais: int) -> None:
    rotulos = [ROTULOS[m] for m in sub["modelo"]]
    cores = [CORES[m] for m in sub["modelo"]]
    barras = ax.bar(rotulos, sub[metrica], color=cores)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylim(0, sub[metrica].max() * 1.18)
    for barra, valor in zip(barras, sub[metrica]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            f"{valor:.{decimais}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def figura_mae_rmse(metricas: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for linha, metrica, decimais in ((0, "mae", 0), (1, "rmse", 0)):
        for coluna, serie in enumerate(SERIES):
            ax = axes[linha, coluna]
            _barras(ax, metricas[metricas["series"] == serie], metrica, decimais)
            ax.set_title(str(serie))
            if coluna == 0:
                ax.set_ylabel(metrica.upper())
    fig.suptitle("MAE e RMSE na validação (28 dias)", fontsize=14, fontweight="bold")
    return fig


def figura_mase(metricas: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(ORDEM_MODELOS))
    largura = 0.25
    for i, serie in enumerate(SERIES):
        vals = (
            metricas.loc[metricas["series"] == serie]
            .set_index("modelo")
            .loc[ORDEM_MODELOS, "mase"]
            .to_numpy()
        )
        ax.bar(x + (i - 1) * largura, vals, largura, label=serie, color=CORES_SERIES[serie])
    ax.axhline(1, color="0.35", ls="--", lw=0.9, label="MASE = 1")
    ax.set_xticks(x, [ROTULOS[m] for m in ORDEM_MODELOS])
    ax.set_ylabel("MASE")
    ax.set_ylim(0, metricas["mase"].max() * 1.2)
    ax.set_title("MASE na validação, comparável entre séries")
    ax.legend(ncol=4, frameon=False)
    return fig


def figura_heatmap_mase(metricas: pd.DataFrame):
    mase = metricas.pivot(index="series", columns="modelo", values="mase").loc[SERIES, ORDEM_MODELOS]
    mase.columns = [ROTULOS[str(c)] for c in mase.columns]
    valores = mase.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    im = ax.imshow(valores, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(mase.shape[1]), mase.columns)
    ax.set_yticks(range(mase.shape[0]), mase.index)
    vmin, vmax = valores.min(), valores.max()
    for i in range(mase.shape[0]):
        for j in range(mase.shape[1]):
            nivel = (valores[i, j] - vmin) / (vmax - vmin)
            cor = "black" if 0.25 < nivel < 0.75 else "white"
            ax.text(j, i, f"{valores[i, j]:.3f}", ha="center", va="center", fontsize=9, color=cor)
    fig.colorbar(im, ax=ax, label="MASE", fraction=0.046, pad=0.04)
    ax.set_title("MASE por série e modelo (verde = menor erro)")
    return fig


def figura_sarima_vs_naive(metricas: pd.DataFrame):
    naive = metricas[metricas["modelo"] == "naive_sazonal"].set_index("series")["mase"]
    sarima = metricas[metricas["modelo"] == "sarima"].set_index("series")["mase"]
    reducao = (1 - sarima / naive) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(SERIES))
    axes[0].bar(x - 0.18, naive.loc[SERIES], 0.36, label="naive sazonal", color=CORES["naive_sazonal"])
    axes[0].bar(x + 0.18, sarima.loc[SERIES], 0.36, label="SARIMA", color=CORES["sarima"])
    axes[0].axhline(1, color="0.35", ls="--", lw=0.9)
    axes[0].set_xticks(x, SERIES)
    axes[0].set_ylabel("MASE")
    axes[0].set_title("SARIMA contra o melhor baseline")
    axes[0].set_ylim(0, 1.55)
    axes[0].legend(frameon=False, loc="upper center", ncol=2)

    cores = ["#2a9d8f" if v > 0 else CORES["sarima"] for v in reducao.loc[SERIES]]
    axes[1].bar(SERIES, reducao.loc[SERIES], color=cores)
    axes[1].axhline(0, color="0.35", lw=0.8)
    axes[1].set_ylabel("redução do MASE (%)")
    axes[1].set_title("Quanto o SARIMA reduz o MASE do naive sazonal")
    for i, serie in enumerate(SERIES):
        axes[1].text(i, reducao.loc[serie], f"{reducao.loc[serie]:.0f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, reducao.max() * 1.2)
    return fig


def texto_comparacao(metricas: pd.DataFrame) -> str:
    naive = metricas[metricas["modelo"] == "naive_sazonal"].set_index("series")["mase"]
    sarima = metricas[metricas["modelo"] == "sarima"].set_index("series")["mase"]
    reducao = (1 - sarima / naive) * 100
    linhas = ["Melhor baseline e SARIMA, por série (MASE):"]
    for serie in SERIES:
        bloco = metricas[metricas["series"] == serie]
        base = bloco[bloco["modelo"] != "sarima"].sort_values("mase").iloc[0]
        sar = bloco[bloco["modelo"] == "sarima"].iloc[0]
        linhas.append(
            f"  {serie}: {ROTULOS[str(base.modelo)]} {base.mase:.3f}  →  SARIMA {sar.mase:.3f}"
            f"  ({reducao.loc[serie]:.0f}% menor que o naive sazonal)"
        )
    return "\n".join(linhas)
