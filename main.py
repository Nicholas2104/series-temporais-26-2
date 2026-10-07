from pathlib import Path
import pandas as pd

# Importações diretas dos seus scripts no módulo local `src`
from src.dados import carregar_split, SERIES
from src.baselines import get_baseline
from src.sarima import sarima
from src.metricas import mae, rmse, mase
from src.diagnostico import figura_diagnostico


def main() -> None:
    # Definir caminhos de saída
    output_dir = Path(".")
    figuras_dir = Path("figuras")
    figuras_dir.mkdir(exist_ok=True)

    treino, val = carregar_split()
    
    datas_val = val.index
    h = len(datas_val)  # Horizonte de validação (28 dias)

    # Lista de baselines a serem avaliados
    nomes_baselines = ["media", "naive", "naive_sazonal", "drift"]

    previsoes_list = []
    metricas_list = []

    # Gerar figuras de diagnóstico e treinar modelos para cada série
    for col in SERIES:
        y_tr = treino[col]
        y_va = val[col]

        # Gerar e salvar figuras de diagnóstico (ACF, PACF e Série)
        fig_path = figuras_dir / f"diagnostico_{col}.png"
        figura_diagnostico(y_tr, nome=col, salvar=str(fig_path))

        # Executar Baselines
        for b_nome in nomes_baselines:
            func_b = get_baseline(b_nome)
            yhat_b = func_b(y_tr, h)

            # Guardar previsões
            for dt, pred in zip(datas_val, yhat_b):
                previsoes_list.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "series": col,
                    "modelo": b_nome,
                    "yhat": float(pred)
                })

            # Guardar métricas
            metricas_list.append({
                "series": col,
                "modelo": b_nome,
                "mae": mae(y_va, yhat_b),
                "rmse": rmse(y_va, yhat_b),
                "mase": mase(y_va, yhat_b, y_tr)
            })

        # Executar Modelo SARIMA
        yhat_sarima, _ = sarima(y_tr, h=h)

        # Guardar previsões do SARIMA
        for dt, pred in zip(datas_val, yhat_sarima):
            previsoes_list.append({
                "date": dt.strftime("%Y-%m-%d"),
                "series": col,
                "modelo": "sarima",
                "yhat": float(pred)
            })

        # Guardar métricas do SARIMA
        metricas_list.append({
            "series": col,
            "modelo": "sarima",
            "mae": mae(y_va, yhat_sarima),
            "rmse": rmse(y_va, yhat_sarima),
            "mase": mase(y_va, yhat_sarima, y_tr)
        })

    # Montar DataFrames e Salvar CSVs
    df_previsoes = pd.DataFrame(previsoes_list)
    df_metricas = pd.DataFrame(metricas_list)

    path_prev = output_dir / "previsoes_validacao.csv"
    path_met = output_dir / "metricas.csv"

    df_previsoes.to_csv(path_prev, index=False)
    df_metricas.to_csv(path_met, index=False)


if __name__ == "__main__":
    main()