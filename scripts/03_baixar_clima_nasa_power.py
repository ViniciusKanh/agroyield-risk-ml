from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import ler_csv, salvar_csv  # noqa: E402

NASA_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def selecionar_municipios_relevantes(
    pam: pd.DataFrame,
    municipios: pd.DataFrame,
    limite: int,
    area_minima: float,
) -> pd.DataFrame:
    """Seleciona municípios mais relevantes pela área colhida média de soja."""
    ranking = (
        pam.groupby("codigo_municipio", as_index=False)
        .agg(
            area_colhida_media_ha=("area_colhida_ha", "mean"),
            rendimento_medio_kg_ha=("rendimento_medio_kg_ha", "mean"),
            anos_disponiveis=("ano", "nunique"),
        )
        .sort_values("area_colhida_media_ha", ascending=False)
    )

    ranking = ranking[ranking["area_colhida_media_ha"] >= area_minima]
    ranking = ranking.head(limite)

    selecionados = ranking.merge(municipios, on="codigo_municipio", how="left")
    selecionados = selecionados.dropna(subset=["latitude", "longitude"])
    return selecionados.reset_index(drop=True)


def baixar_clima_municipio(
    codigo_municipio: str,
    latitude: float,
    longitude: float,
    parametros: list[str],
    ano_inicio: int,
    ano_fim: int,
    community: str,
    timeout: int,
) -> dict:
    """Baixa dados diários da NASA POWER para um município."""
    params = {
        "parameters": ",".join(parametros),
        "community": community,
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{ano_inicio}0101",
        "end": f"{ano_fim}1231",
        "format": "JSON",
    }

    resposta = requests.get(NASA_DAILY_URL, params=params, timeout=timeout)
    if not resposta.ok:
        raise RuntimeError(
            f"Falha NASA POWER no município {codigo_municipio}. "
            f"Status={resposta.status_code}. Resposta={resposta.text[:300]}"
        )
    return resposta.json()


def nasa_json_para_anual(dados: dict, codigo_municipio: str) -> pd.DataFrame:
    """Transforma JSON diário da NASA POWER em indicadores anuais."""
    parametros = dados.get("properties", {}).get("parameter", {})
    if not parametros:
        raise RuntimeError(f"NASA POWER retornou dados vazios para {codigo_municipio}")

    series = []
    for parametro, valores in parametros.items():
        serie = pd.Series(valores, name=parametro)
        series.append(serie)

    diario = pd.concat(series, axis=1).reset_index(names="data")
    diario["data"] = pd.to_datetime(diario["data"], format="%Y%m%d")
    diario["ano"] = diario["data"].dt.year
    diario["codigo_municipio"] = str(codigo_municipio)

    # Valores sentinela comuns em bases climáticas são tratados como ausentes.
    for coluna in diario.columns:
        if coluna not in {"data", "ano", "codigo_municipio"}:
            diario[coluna] = pd.to_numeric(diario[coluna], errors="coerce")
            diario.loc[diario[coluna] <= -900, coluna] = pd.NA

    anual = (
        diario.groupby(["codigo_municipio", "ano"], as_index=False)
        .agg(
            chuva_total_mm=("PRECTOTCORR", "sum"),
            chuva_media_diaria_mm=("PRECTOTCORR", "mean"),
            temperatura_media_c=("T2M", "mean"),
            temperatura_maxima_media_c=("T2M_MAX", "mean"),
            temperatura_minima_media_c=("T2M_MIN", "mean"),
            umidade_media_pct=("RH2M", "mean"),
            vento_medio_ms=("WS2M", "mean"),
            dias_observados=("PRECTOTCORR", "count"),
        )
    )

    # Indicadores de estresse climático, calculados separadamente.
    diario["flag_dia_seco"] = (diario["PRECTOTCORR"] < 1).astype(int)
    diario["flag_chuva_extrema"] = (diario["PRECTOTCORR"] >= 50).astype(int)
    diario["flag_calor_extremo"] = (diario["T2M_MAX"] >= 35).astype(int)

    estresse = (
        diario.groupby(["codigo_municipio", "ano"], as_index=False)
        .agg(
            dias_secos=("flag_dia_seco", "sum"),
            dias_chuva_extrema=("flag_chuva_extrema", "sum"),
            dias_calor_extremo=("flag_calor_extremo", "sum"),
        )
    )

    anual = anual.merge(estresse, on=["codigo_municipio", "ano"], how="left")
    return anual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limite-municipios", type=int, default=None)
    args = parser.parse_args()

    garantir_diretorios()
    config = carregar_config()

    pam = ler_csv(ROOT / "data" / "interim" / "pam_soja_uf_tratada.csv")
    municipios = ler_csv(ROOT / "data" / "interim" / "municipios_uf.csv")

    pam["codigo_municipio"] = pam["codigo_municipio"].astype(str)
    municipios["codigo_municipio"] = municipios["codigo_municipio"].astype(str)

    limite = args.limite_municipios or int(config["amostra"]["limite_municipios_por_area_soja"])
    area_minima = float(config["amostra"]["area_colhida_minima_media_ha"])

    selecionados = selecionar_municipios_relevantes(pam, municipios, limite, area_minima)
    caminho_sel = ROOT / "data" / "interim" / "municipios_selecionados_clima.csv"
    salvar_csv(selecionados, caminho_sel)

    print(f"Municípios selecionados para clima: {len(selecionados)}")
    print(f"Lista salva em: {caminho_sel}")

    parametros = list(config["nasa_power"]["parametros"])
    ano_inicio = int(config["projeto"]["ano_inicio"])
    ano_fim = int(config["projeto"]["ano_fim"])
    community = str(config["nasa_power"]["community"])
    sleep_segundos = float(config["nasa_power"]["sleep_segundos"])
    timeout = int(config["nasa_power"]["timeout_segundos"])

    tabelas_anuais = []
    for linha in tqdm(selecionados.itertuples(index=False), total=len(selecionados)):
        codigo = str(linha.codigo_municipio)
        cache = ROOT / "data" / "raw" / "nasa_power" / f"nasa_power_{codigo}.json"

        if cache.exists():
            dados = json.loads(cache.read_text(encoding="utf-8"))
        else:
            dados = baixar_clima_municipio(
                codigo_municipio=codigo,
                latitude=float(linha.latitude),
                longitude=float(linha.longitude),
                parametros=parametros,
                ano_inicio=ano_inicio,
                ano_fim=ano_fim,
                community=community,
                timeout=timeout,
            )
            cache.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
            time.sleep(sleep_segundos)

        anual = nasa_json_para_anual(dados, codigo)
        tabelas_anuais.append(anual)

    clima = pd.concat(tabelas_anuais, ignore_index=True)
    caminho_clima = ROOT / "data" / "interim" / "clima_nasa_power_anual.csv"
    salvar_csv(clima, caminho_clima)

    print(f"Clima anual salvo em: {caminho_clima}")
    print(f"Linhas: {len(clima)}")


if __name__ == "__main__":
    main()
