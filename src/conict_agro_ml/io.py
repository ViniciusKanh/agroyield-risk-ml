from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalizar_texto(texto: object) -> str:
    """Normaliza texto para comparação: minúsculo, sem acento e sem excesso de espaço."""
    if pd.isna(texto):
        return ""
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    texto = re.sub(r"\s+", " ", texto)
    return texto


def converter_numero_sidra(valor: object) -> float:
    """Converte valores textuais do SIDRA para float, tratando símbolos de ausência."""
    if pd.isna(valor):
        return float("nan")

    texto = str(valor).strip()
    if texto in {"", "-", "...", "..", "X", "x"}:
        return float("nan")

    texto = texto.replace(".", "").replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return float("nan")


def salvar_csv(df: pd.DataFrame, caminho: Path | str) -> None:
    """Salva DataFrame em CSV com UTF-8 e separador vírgula."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8")


def ler_csv(caminho: Path | str) -> pd.DataFrame:
    """Lê CSV com erro explícito caso o arquivo não exista."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}. Execute as etapas anteriores do pipeline."
        )
    return pd.read_csv(caminho)
