from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ETAPAS = [
    "scripts/00_validar_ambiente.py",
    "scripts/01_baixar_municipios.py",
    "scripts/02_baixar_pam_ibge.py",
    "scripts/03_baixar_clima_nasa_power.py",
    "scripts/04_montar_dataset.py",
    "scripts/05_modelagem.py",
    "scripts/06_interpretabilidade.py",
    "scripts/07_gerar_resumo_resultados.py",
]


def main() -> None:
    for etapa in ETAPAS:
        print("\n" + "=" * 80)
        print(f"Executando etapa: {etapa}")
        print("=" * 80)
        subprocess.run([sys.executable, str(ROOT / etapa)], check=True)

    print("\nPipeline finalizado com sucesso.")


if __name__ == "__main__":
    main()
