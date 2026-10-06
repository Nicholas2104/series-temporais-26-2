import numpy as np
import pandas as pd
from .config import PERIODO


def sarima(y: pd.Series, h: int, m: int = PERIODO, passos=(0.05, 0.0025), pontos: int = 20, limite: float = 0.99):
    y = y.to_numpy(float)
    z = y[m:] - y[:-m]

    # busca em grid de phi e theta minimizando a SSE
    phi, theta = 0.0, 0.0
    for passo in passos:
        grid = np.arange(-pontos, pontos + 1) * passo
        grid_phi = np.clip(phi + grid, -limite, limite)
        grid_theta = np.clip(theta + grid, -limite, limite)
        P, T = np.array([(a, b) for a in grid_phi for b in grid_theta]).T

        # residuos de todos os pares (phi, theta)
        e = np.zeros((len(z) + h, len(P)))
        for t in range(m, len(z)):
            e[t] = z[t] - P * z[t - 1] - T * e[t - m]

        k = (e**2).sum(axis=0).argmin()
        phi, theta = P[k], T[k]

    # previsao
    e = e[:, k]
    z = np.concatenate([z, np.zeros(h)])
    y = np.concatenate([y, np.zeros(h)])
    for t in range(len(z) - h, len(z)):
        z[t] = phi * z[t - 1] + theta * e[t - m]
        y[t + m] = y[t] + z[t]
    return y[-h:], e[m:-h]
