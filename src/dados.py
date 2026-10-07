from pathlib import Path
import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"
SERIES = ["store_total", "FOODS", "HOBBIES"]
FIM_TREINO = pd.Timestamp("2016-03-27")
INICIO_VAL, FIM_VAL = pd.Timestamp("2016-03-28"), pd.Timestamp("2016-04-24")


def _ler(nome: str) -> pd.DataFrame:
    df = pd.read_csv(DADOS / nome, parse_dates=["date"]).set_index("date").sort_index()
    return df.asfreq("D") 


def carregar_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    treino, val = _ler("treino.csv"), _ler("validacao.csv")

    assert treino.index.max() == FIM_TREINO
    assert val.index.min() == INICIO_VAL and val.index.max() == FIM_VAL

    return treino[SERIES], val[SERIES]


def datas_holdout() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(DADOS / "holdout_datas.csv", parse_dates=["date"])["date"])
