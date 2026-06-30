# Roteiro de execução no VSCode

## 1. Abrir a pasta

Abra a pasta `conict_agro_ml` no VSCode.

## 2. Criar ambiente virtual

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/Mac:

```bash
source .venv/bin/activate
```

## 3. Instalar dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Testar ambiente

```bash
python scripts/00_validar_ambiente.py
```

## 5. Executar pipeline completo

```bash
python run_pipeline.py
```

## 6. Executar com menos municípios no teste inicial

Para validar rápido, rode até a coleta climática com poucos municípios:

```bash
python scripts/01_baixar_municipios.py
python scripts/02_baixar_pam_ibge.py
python scripts/03_baixar_clima_nasa_power.py --limite-municipios 10
python scripts/04_montar_dataset.py
python scripts/05_modelagem.py
python scripts/06_interpretabilidade.py
python scripts/07_gerar_resumo_resultados.py
```

Depois que funcionar, aumente para 80 ou 120 municípios em `configs/config.yaml`.

## 7. Onde olhar os resultados

- Dataset final: `data/processed/dataset_agro_ml.csv`.
- Métricas: `outputs/tables/metricas_modelos.csv`.
- Importância das variáveis: `outputs/tables/importancia_variaveis.csv`.
- Gráficos: `outputs/figures/`.
- Resumo textual: `reports/resumo_resultados.md`.
