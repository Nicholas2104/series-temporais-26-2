import os
import pandas as pd
import matplotlib.pyplot as plt

# Importações dos módulos do projeto
from src.config import PERIODO
from src.dados import carregar_split, SERIES
from src.diagnostico import figura_diagnostico, resumo_zeros
from src.baselines import get_baseline
from src.sarima import sarima
from src.metricas import mae, rmse, mase


def main():
    treino, val = carregar_split()
    assert treino.index.max() < val.index.min(), "Erro no split: O treino invade a validação!"

    # Diagnóstico e visualização
    os.makedirs("figuras", exist_ok=True)
    for col in SERIES:
        fig = figura_diagnostico(
            treino[col], 
            nome=col, 
            m=PERIODO, 
            nlags=35, 
            salvar=f"figuras/{col}_diagnostico.png"
        )
        plt.close(fig)

    # Resumo de zeros
    df_zeros = resumo_zeros(treino)
    if not df_zeros.empty:
        print("\n  • Resumo de valores zero encontrados nas séries:")
        print(df_zeros.head())

    # Avaliação e Geração de Previsões
    print("\n[3/4] Avaliando modelos e gerando previsões...")
    h = len(val)  # Horizonte de previsão
    nomes_baselines = ["media", "naive", "naive_sazonal", "drift"]
    
    lista_metricas = []
    
    # DataFrame base para armazenar as previsões alinhadas com as datas de validação
    df_previsoes = pd.DataFrame(index=val.index)

    for col in SERIES:
        y_tr = treino[col]
        y_val = val[col]

        # Guardar valor real da série no conjunto de validação
        df_previsoes[f"{col}_real"] = y_val

        # Avaliação dos Baselines
        for b_name in nomes_baselines:
            func_baseline = get_baseline(b_name)
            y_hat = func_baseline(y_tr, h)
            
            # Guardar previsão
            df_previsoes[f"{col}_baseline_{b_name}"] = y_hat
            
            # Calcular métricas
            lista_metricas.append({
                "Série": col,
                "Modelo": f"Baseline: {b_name}",
                "MAE": mae(y_val, y_hat),
                "RMSE": rmse(y_val, y_hat),
                "MASE": mase(y_val, y_hat, y_tr)
            })

        # Avaliação do modelo SARIMA
        y_hat_sarima, residuos = sarima(y_tr, h=h, m=PERIODO)
        
        # Guardar previsão SARIMA
        df_previsoes[f"{col}_sarima"] = y_hat_sarima
        
        # Calcular métricas
        lista_metricas.append({
            "Série": col,
            "Modelo": "SARIMA",
            "MAE": mae(y_val, y_hat_sarima),
            "RMSE": rmse(y_val, y_hat_sarima),
            "MASE": mase(y_val, y_hat_sarima, y_tr)
        })

    # Exibição e Salvamento dos Resultados
    os.makedirs("resultados", exist_ok=True)

    # Métricas (metricas.csv)
    df_metricas = pd.DataFrame(lista_metricas)
    caminho_metricas = "resultados/metricas.csv"
    df_metricas.to_csv(caminho_metricas, index=False)
    print(f"  Métrica salvas em: '{caminho_metricas}'")
    print("\nResumo das Métricas:")
    print(df_metricas.to_string(index=False))

    # Previsões na Validação (previsoes_validacao.csv)
    caminho_previsoes = "resultados/previsoes_validacao.csv"
    df_previsoes.to_csv(caminho_previsoes, index=True)  # Mantém o índice de datas
    print(f"\n  Previsões de validação salvas em: '{caminho_previsoes}'")

if __name__ == "__main__":
    main()
