import numpy as np
from .config import PERIODO


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def escala_mase(y_treino, m: int = PERIODO):
    y = np.asarray(y_treino)
    return float(np.mean(np.abs(y[m:] - y[:-m])))


def mase(y, yhat, y_treino, m: int = PERIODO):
    return mae(y, yhat) / escala_mase(y_treino, m)
