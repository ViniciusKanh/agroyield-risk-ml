from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def carregar_config(caminho: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Carrega o arquivo YAML de configuração do projeto."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {caminho}")

    with caminho.open("r", encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def garantir_diretorios() -> None:
    """Garante que os diretórios principais existam antes de salvar arquivos."""
    diretorios = [
        ROOT / "data" / "raw" / "pam_ibge",
        ROOT / "data" / "raw" / "nasa_power",
        ROOT / "data" / "raw" / "municipios",
        ROOT / "data" / "interim",
        ROOT / "data" / "processed",
        ROOT / "outputs" / "figures",
        ROOT / "outputs" / "tables",
        ROOT / "models",
        ROOT / "reports",
    ]
    for diretorio in diretorios:
        diretorio.mkdir(parents=True, exist_ok=True)
