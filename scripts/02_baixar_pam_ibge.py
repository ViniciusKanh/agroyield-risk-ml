from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import salvar_csv  # noqa: E402
from conict_agro_ml.sidra import (  # noqa: E402
    baixar_sidra_json,
    montar_url_pam_1612,
    sidra_para_dataframe,
    tratar_pam_1612,
)


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    ano_inicio = int(config["projeto"]["ano_inicio"])
    ano_fim = int(config["projeto"]["ano_fim"])
    anos = range(ano_inicio, ano_fim + 1)

    url = montar_url_pam_1612(
        tabela=str(config["sidra"]["tabela_pam_lavouras_temporarias"]),
        uf_codigo=str(config["projeto"]["uf_codigo"]),
        anos=anos,
        classificacao_produto=str(config["sidra"]["classificacao_produto_lavoura_temporaria"]),
        codigo_produto=str(config["sidra"]["codigo_produto_soja"]),
    )

    print("Baixando dados da PAM/IBGE via SIDRA...")
    print(f"URL: {url}")

    dados = baixar_sidra_json(url)

    caminho_json = ROOT / "data" / "raw" / "pam_ibge" / "pam_1612_soja_uf.json"
    caminho_json.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")

    df_bruto = sidra_para_dataframe(dados)
    caminho_bruto = ROOT / "data" / "raw" / "pam_ibge" / "pam_1612_soja_uf_bruto.csv"
    salvar_csv(df_bruto, caminho_bruto)

    pam = tratar_pam_1612(df_bruto)
    caminho_tratado = ROOT / "data" / "interim" / "pam_soja_uf_tratada.csv"
    salvar_csv(pam, caminho_tratado)

    print(f"Dados brutos salvos em: {caminho_bruto}")
    print(f"Dados tratados salvos em: {caminho_tratado}")
    print(f"Linhas tratadas: {len(pam)}")
    print(f"Municípios com soja: {pam['codigo_municipio'].nunique()}")
    print(f"Período: {pam['ano'].min()}–{pam['ano'].max()}")


if __name__ == "__main__":
    main()
