from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import ler_csv, salvar_csv  # noqa: E402


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    pacote = joblib.load(ROOT / "models" / "melhor_modelo.joblib")
    modelo = pacote["modelo"]
    features = pacote["features"]
    melhor_nome = pacote["melhor_nome"]

    dataset = ler_csv(ROOT / "data" / "processed" / "dataset_agro_ml.csv")
    ano_treino_ate = int(config["modelagem"]["ano_treino_ate"])
    teste = dataset[dataset["ano"] > ano_treino_ate].copy()

    x_test = teste[features]
    y_test = teste["risco_quebra"].astype(int)

    print(f"Calculando importância por permutação para: {melhor_nome}")
    resultado = permutation_importance(
        modelo,
        x_test,
        y_test,
        n_repeats=20,
        random_state=int(config["modelagem"]["random_state"]),
        scoring="f1",
        n_jobs=-1,
    )

    importancia = pd.DataFrame(
        {
            "variavel": features,
            "importancia_media_f1": resultado.importances_mean,
            "importancia_desvio_f1": resultado.importances_std,
        }
    ).sort_values("importancia_media_f1", ascending=False)

    caminho = ROOT / "outputs" / "tables" / "importancia_variaveis.csv"
    salvar_csv(importancia, caminho)

    fig, ax = plt.subplots(figsize=(8, 5))
    top = importancia.head(12).sort_values("importancia_media_f1", ascending=True)
    ax.barh(top["variavel"], top["importancia_media_f1"])
    ax.set_title("Importância por permutação — melhor modelo")
    ax.set_xlabel("Redução média do F1-score após permutação")
    fig.tight_layout()

    caminho_fig = ROOT / "outputs" / "figures" / "importancia_permutacao_melhor_modelo.png"
    fig.savefig(caminho_fig, dpi=180)
    plt.close(fig)

    print(f"Tabela salva em: {caminho}")
    print(f"Figura salva em: {caminho_fig}")
    print(importancia.head(10))


if __name__ == "__main__":
    main()
