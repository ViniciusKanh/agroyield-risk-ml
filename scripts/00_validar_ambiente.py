from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402


PACOTES = [
    "pandas",
    "numpy",
    "requests",
    "yaml",
    "sklearn",
    "matplotlib",
    "joblib",
    "tqdm",
]


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    print("Validando ambiente do projeto...")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Projeto: {config['projeto']['nome']}")
    print(f"Recorte: {config['projeto']['cultura']} — {config['projeto']['estado_nome']}")

    erros = []
    for pacote in PACOTES:
        try:
            importlib.import_module(pacote)
            print(f"OK: {pacote}")
        except ImportError as exc:
            erros.append((pacote, exc))
            print(f"ERRO: {pacote} não instalado")

    if erros:
        print("\nInstale as dependências com:")
        print("pip install -r requirements.txt")
        raise SystemExit(1)

    print("\nAmbiente validado com sucesso.")


if __name__ == "__main__":
    main()
