from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import salvar_csv  # noqa: E402

URL_MUNICIPIOS = (
    "https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/"
    "main/csv/municipios.csv"
)


def main() -> None:
    garantir_diretorios()
    config = carregar_config()
    uf_codigo = int(config["projeto"]["uf_codigo"])

    print("Baixando base operacional de municípios com latitude/longitude...")
    resposta = requests.get(URL_MUNICIPIOS, timeout=60)
    resposta.raise_for_status()

    caminho_raw = ROOT / "data" / "raw" / "municipios" / "municipios_brasileiros.csv"
    caminho_raw.write_bytes(resposta.content)

    municipios = pd.read_csv(caminho_raw)

    # O arquivo possui codigo_uf numérico. Para o Paraná, codigo_uf = 41.
    municipios_uf = municipios[municipios["codigo_uf"] == uf_codigo].copy()
    municipios_uf = municipios_uf.rename(
        columns={
            "codigo_ibge": "codigo_municipio",
            "nome": "municipio_base_coord",
            "latitude": "latitude",
            "longitude": "longitude",
        }
    )

    municipios_uf["codigo_municipio"] = municipios_uf["codigo_municipio"].astype(str)
    colunas = ["codigo_municipio", "municipio_base_coord", "latitude", "longitude", "codigo_uf", "uf"]
    municipios_uf = municipios_uf[[col for col in colunas if col in municipios_uf.columns]]

    saida = ROOT / "data" / "interim" / "municipios_uf.csv"
    salvar_csv(municipios_uf, saida)

    print(f"Municípios salvos em: {saida}")
    print(f"Total de municípios da UF: {len(municipios_uf)}")


if __name__ == "__main__":
    main()
