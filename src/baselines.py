import numpy as np
import pandas as pd
from .config import PERIODO


def media(y: pd.Series, h: int):
    return np.full(h, y.mean())


def naive(y: pd.Series, h: int):
    return np.full(h, y.iloc[-1])


def naive_sazonal(y: pd.Series, h: int, m: int = PERIODO):
    ultimo_ciclo = y.iloc[-m:].to_numpy()
    return np.resize(ultimo_ciclo, h)


def drift(y: pd.Series, h: int):
    T = len(y)
    slope = (y.iloc[-1] - y.iloc[0]) / (T - 1)
    return y.iloc[-1] + slope * np.arange(1, h + 1)


def get_baseline(baseline: str):
    BASELINES = {
        "media": media,
        "naive": naive,
        "naive_sazonal": naive_sazonal,
        "drift": drift,
    }

    return BASELINES[baseline]
