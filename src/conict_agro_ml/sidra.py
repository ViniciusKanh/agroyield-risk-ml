from __future__ import annotations

from typing import Iterable
from urllib.parse import quote

import pandas as pd
import requests

from conict_agro_ml.io import converter_numero_sidra, normalizar_texto


BASE_URL = "https://apisidra.ibge.gov.br/values"


def montar_url_pam_1612(
    tabela: str,
    uf_codigo: str,
    anos: Iterable[int],
    classificacao_produto: str,
    codigo_produto: str,
) -> str:
    """Monta URL direta da API SIDRA para a Tabela 1612.

    A expressão territorial `n6/in n3 UF` significa: municípios contidos na UF.
    """
    periodo = ",".join(str(ano) for ano in anos)
    filtro_territorial = quote(f"in n3 {uf_codigo}")

    return (
        f"{BASE_URL}/t/{tabela}/n6/{filtro_territorial}"
        f"/v/all/p/{periodo}/c{classificacao_produto}/{codigo_produto}"
        "?formato=json"
    )


def baixar_sidra_json(url: str, timeout: int = 120) -> list[dict]:
    """Baixa resposta JSON do SIDRA com mensagem de erro legível."""
    resposta = requests.get(url, timeout=timeout)
    if not resposta.ok:
        raise RuntimeError(
            f"Falha ao consultar SIDRA. Status={resposta.status_code}. URL={url}\n"
            f"Resposta: {resposta.text[:500]}"
        )
    dados = resposta.json()
    if not dados:
        raise RuntimeError("SIDRA retornou resposta vazia.")
    return dados


def sidra_para_dataframe(dados: list[dict]) -> pd.DataFrame:
    """Converte o formato da API SIDRA em DataFrame tabular.

    O primeiro registro da resposta geralmente contém o mapeamento de códigos para nomes
    de colunas. A função trata esse cabeçalho automaticamente.
    """
    df = pd.DataFrame(dados)

    if df.empty:
        raise RuntimeError("DataFrame SIDRA vazio após conversão.")

    primeira_linha = df.iloc[0].to_dict()
    if str(primeira_linha.get("V", "")).lower() == "valor":
        df = df.iloc[1:].copy()
        df = df.rename(columns={codigo: nome for codigo, nome in primeira_linha.items()})

    return df.reset_index(drop=True)


def tratar_pam_1612(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Trata dados brutos da Tabela 1612 e retorna uma tabela ampla.

    A função não depende rigidamente dos códigos das variáveis. Ela usa os nomes das
    variáveis retornados pela própria API, o que deixa o pipeline mais resistente.
    """
    colunas_esperadas = {
        "Município (Código)": "codigo_municipio",
        "Município": "municipio",
        "Ano (Código)": "ano",
        "Ano": "ano_nome",
        "Variável (Código)": "codigo_variavel",
        "Variável": "variavel",
        "Produto das lavouras temporárias (Código)": "codigo_produto",
        "Produto das lavouras temporárias": "produto",
        "Valor": "valor",
        "Unidade de Medida": "unidade_medida",
    }

    # Renomeia apenas colunas que existirem na resposta.
    df = df_bruto.rename(columns={k: v for k, v in colunas_esperadas.items() if k in df_bruto.columns})

    obrigatorias = ["codigo_municipio", "municipio", "ano", "variavel", "valor"]
    faltantes = [col for col in obrigatorias if col not in df.columns]
    if faltantes:
        raise RuntimeError(
            "Não foi possível identificar colunas essenciais do SIDRA: "
            f"{faltantes}. Verifique se a API mudou o cabeçalho."
        )

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["valor"] = df["valor"].apply(converter_numero_sidra)
    df["variavel_norm"] = df["variavel"].apply(normalizar_texto)

    id_cols = ["codigo_municipio", "municipio", "ano"]
    if "produto" in df.columns:
        id_cols.append("produto")

    tabela = (
        df.pivot_table(
            index=id_cols,
            columns="variavel_norm",
            values="valor",
            aggfunc="first",
        )
        .reset_index()
    )

    renomear = {}
    for coluna in tabela.columns:
        col_norm = normalizar_texto(coluna)
        if "area plantada" in col_norm:
            renomear[coluna] = "area_plantada_ha"
        elif "area colhida" in col_norm:
            renomear[coluna] = "area_colhida_ha"
        elif "quantidade produzida" in col_norm:
            renomear[coluna] = "quantidade_produzida_t"
        elif "rendimento medio" in col_norm:
            renomear[coluna] = "rendimento_medio_kg_ha"
        elif "valor da producao" in col_norm:
            renomear[coluna] = "valor_producao_mil_reais"

    tabela = tabela.rename(columns=renomear)

    colunas_finais = [
        "codigo_municipio",
        "municipio",
        "ano",
        "produto",
        "area_plantada_ha",
        "area_colhida_ha",
        "quantidade_produzida_t",
        "rendimento_medio_kg_ha",
        "valor_producao_mil_reais",
    ]

    existentes = [col for col in colunas_finais if col in tabela.columns]
    tabela = tabela[existentes].copy()

    tabela["codigo_municipio"] = tabela["codigo_municipio"].astype(str).str.extract(r"(\d+)")[0]
    tabela = tabela.dropna(subset=["ano", "rendimento_medio_kg_ha"])
    tabela = tabela[tabela["rendimento_medio_kg_ha"] > 0]

    return tabela.sort_values(["codigo_municipio", "ano"]).reset_index(drop=True)
