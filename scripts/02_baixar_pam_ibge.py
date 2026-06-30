from __future__ import annotations

import json
import sys
import time
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


def dividir_em_blocos(lista: list[int], tamanho_bloco: int) -> list[list[int]]:
    """
    Divide uma lista de anos em blocos menores.

    Isso evita erro da API do SIDRA por excesso de valores solicitados.
    """
    return [
        lista[i : i + tamanho_bloco]
        for i in range(0, len(lista), tamanho_bloco)
    ]


def juntar_respostas_sidra(respostas: list[list[dict]]) -> list[dict]:
    """
    Junta múltiplas respostas da API do SIDRA em uma única lista.

    A primeira linha da resposta do SIDRA é o cabeçalho.
    Por isso, mantemos o cabeçalho apenas da primeira resposta
    e removemos o cabeçalho das respostas seguintes.
    """
    dados_unificados: list[dict] = []

    for indice, resposta in enumerate(respostas):
        if not resposta:
            continue

        if indice == 0:
            dados_unificados.extend(resposta)
        else:
            dados_unificados.extend(resposta[1:])

    return dados_unificados


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    ano_inicio = int(config["projeto"]["ano_inicio"])
    ano_fim = int(config["projeto"]["ano_fim"])
    anos = list(range(ano_inicio, ano_fim + 1))

    # Baixar em blocos evita o limite de 50.000 valores do SIDRA.
    tamanho_bloco_anos = 5
    blocos_anos = dividir_em_blocos(anos, tamanho_bloco_anos)

    respostas: list[list[dict]] = []

    print("Baixando dados da PAM/IBGE via SIDRA em blocos...")
    print(f"Período solicitado: {ano_inicio}–{ano_fim}")
    print(f"Total de blocos: {len(blocos_anos)}")

    for indice, bloco in enumerate(blocos_anos, start=1):
        url = montar_url_pam_1612(
            tabela=str(config["sidra"]["tabela_pam_lavouras_temporarias"]),
            uf_codigo=str(config["projeto"]["uf_codigo"]),
            anos=bloco,
            classificacao_produto=str(
                config["sidra"]["classificacao_produto_lavoura_temporaria"]
            ),
            codigo_produto=str(config["sidra"]["codigo_produto_soja"]),
        )

        print("-" * 80)
        print(f"Bloco {indice}/{len(blocos_anos)}")
        print(f"Anos: {min(bloco)}–{max(bloco)}")
        print(f"URL: {url}")

        dados_bloco = baixar_sidra_json(url)
        respostas.append(dados_bloco)

        print(f"Registros recebidos no bloco: {len(dados_bloco)}")

        # Pausa curta para evitar excesso de chamadas consecutivas.
        time.sleep(0.5)

    dados = juntar_respostas_sidra(respostas)

    caminho_json = ROOT / "data" / "raw" / "pam_ibge" / "pam_1612_soja_uf.json"
    caminho_json.parent.mkdir(parents=True, exist_ok=True)
    caminho_json.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    df_bruto = sidra_para_dataframe(dados)
    caminho_bruto = ROOT / "data" / "raw" / "pam_ibge" / "pam_1612_soja_uf_bruto.csv"
    salvar_csv(df_bruto, caminho_bruto)

    pam = tratar_pam_1612(df_bruto)
    caminho_tratado = ROOT / "data" / "interim" / "pam_soja_uf_tratada.csv"
    salvar_csv(pam, caminho_tratado)

    print("=" * 80)
    print("Download e tratamento da PAM finalizados com sucesso.")
    print(f"JSON unificado salvo em: {caminho_json}")
    print(f"Dados brutos salvos em: {caminho_bruto}")
    print(f"Dados tratados salvos em: {caminho_tratado}")
    print(f"Linhas tratadas: {len(pam)}")

    if not pam.empty:
        print(f"Municípios com soja: {pam['codigo_municipio'].nunique()}")
        print(f"Período: {pam['ano'].min()}–{pam['ano'].max()}")
    else:
        print("Atenção: o dataframe tratado ficou vazio.")


if __name__ == "__main__":
    main()