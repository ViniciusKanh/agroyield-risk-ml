from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import ler_csv, salvar_csv  # noqa: E402


def criar_variaveis_temporais(df: pd.DataFrame, janela: int, limiar: float) -> pd.DataFrame:
    """Cria média móvel histórica, defasagem e variável-alvo de quebra."""
    df = df.sort_values(["codigo_municipio", "ano"]).copy()

    grupo = df.groupby("codigo_municipio", group_keys=False)
    df["produtividade_lag1"] = grupo["rendimento_medio_kg_ha"].shift(1)

    df["media_movel_3anos"] = grupo["rendimento_medio_kg_ha"].apply(
        lambda s: s.shift(1).rolling(window=janela, min_periods=janela).mean()
    )

    df["indice_produtividade_relativa"] = (
        df["rendimento_medio_kg_ha"] / df["media_movel_3anos"]
    )
    df["queda_percentual"] = 1 - df["indice_produtividade_relativa"]
    df["risco_quebra"] = (
        df["rendimento_medio_kg_ha"] < limiar * df["media_movel_3anos"]
    ).astype("Int64")

    return df


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    pam = ler_csv(ROOT / "data" / "interim" / "pam_soja_uf_tratada.csv")
    clima = ler_csv(ROOT / "data" / "interim" / "clima_nasa_power_anual.csv")
    municipios = ler_csv(ROOT / "data" / "interim" / "municipios_uf.csv")

    for df in [pam, clima, municipios]:
        df["codigo_municipio"] = df["codigo_municipio"].astype(str)

    pam["ano"] = pd.to_numeric(pam["ano"], errors="coerce").astype(int)
    clima["ano"] = pd.to_numeric(clima["ano"], errors="coerce").astype(int)

    dataset = pam.merge(clima, on=["codigo_municipio", "ano"], how="inner")
    dataset = dataset.merge(municipios, on="codigo_municipio", how="left")

    janela = int(config["alvo"]["janela_media_movel"])
    limiar = float(config["alvo"]["limiar_quebra"])
    dataset = criar_variaveis_temporais(dataset, janela=janela, limiar=limiar)

    # Remove anos iniciais sem histórico suficiente para definir o alvo.
    dataset = dataset.dropna(subset=["media_movel_3anos", "produtividade_lag1", "risco_quebra"])
    dataset["risco_quebra"] = dataset["risco_quebra"].astype(int)

    # Remove linhas sem variáveis climáticas principais.
    colunas_clima = ["chuva_total_mm", "temperatura_media_c", "dias_secos"]
    dataset = dataset.dropna(subset=colunas_clima)

    caminho = ROOT / "data" / "processed" / "dataset_agro_ml.csv"
    salvar_csv(dataset, caminho)

    print(f"Dataset final salvo em: {caminho}")
    print(f"Linhas: {len(dataset)}")
    print(f"Municípios: {dataset['codigo_municipio'].nunique()}")
    print(f"Anos: {dataset['ano'].min()}–{dataset['ano'].max()}")
    print("Distribuição do alvo:")
    print(dataset["risco_quebra"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
