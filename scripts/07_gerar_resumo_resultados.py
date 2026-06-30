from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import ler_csv  # noqa: E402


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    dataset = ler_csv(ROOT / "data" / "processed" / "dataset_agro_ml.csv")
    metricas = ler_csv(ROOT / "outputs" / "tables" / "metricas_modelos.csv")
    importancia = ler_csv(ROOT / "outputs" / "tables" / "importancia_variaveis.csv")

    metadata_path = ROOT / "models" / "metadata_modelagem.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    melhor = metricas.iloc[0]
    top_importancia = importancia.head(8)

    texto = f"""# Resumo dos resultados experimentais

## Recorte

- Cultura: {config['projeto']['cultura']}
- Estado: {config['projeto']['estado_nome']}
- Período analisado: {int(dataset['ano'].min())}–{int(dataset['ano'].max())}
- Municípios analisados: {dataset['codigo_municipio'].nunique()}
- Linhas município-ano: {len(dataset)}

## Variável-alvo

A variável `risco_quebra` foi definida como 1 quando o rendimento médio do município no ano ficou abaixo de {float(config['alvo']['limiar_quebra']) * 100:.0f}% da média móvel dos {config['alvo']['janela_media_movel']} anos anteriores.

Distribuição do alvo:

{dataset['risco_quebra'].value_counts().sort_index().to_string()}

## Melhor modelo

- Modelo: {melhor['modelo']}
- F1-score: {melhor['f1']:.4f}
- Recall: {melhor['recall']:.4f}
- Precision: {melhor['precision']:.4f}
- Balanced accuracy: {melhor['balanced_accuracy']:.4f}
- ROC AUC: {melhor.get('roc_auc', float('nan')):.4f}

## Comparação dos modelos

{metricas.to_markdown(index=False)}

## Variáveis mais importantes

{top_importancia.to_markdown(index=False)}

## Interpretação inicial

O resultado deve ser interpretado como uma classificação exploratória de risco, e não como inferência causal. Um bom resultado experimental para o CONICT deve mostrar que variáveis climáticas e histórico produtivo possuem poder informativo para distinguir anos com e sem quebra relevante de produtividade.

## Cuidados para o artigo

- Não afirmar causalidade climática sem desenho causal.
- Descrever a divisão temporal entre treino e teste.
- Explicar a criação da variável-alvo.
- Relatar limitações: agregação anual, uso de coordenada municipal aproximada e ausência de variáveis de solo/manejo.

## Arquivos gerados

- `outputs/tables/metricas_modelos.csv`
- `outputs/tables/importancia_variaveis.csv`
- `outputs/tables/previsoes_teste.csv`
- `outputs/figures/matriz_confusao_*.png`
- `outputs/figures/importancia_permutacao_melhor_modelo.png`
- `models/melhor_modelo.joblib`

## Metadados

```json
{json.dumps(metadata, ensure_ascii=False, indent=2)}
```
"""

    caminho = ROOT / "reports" / "resumo_resultados.md"
    caminho.write_text(texto, encoding="utf-8")
    print(f"Resumo salvo em: {caminho}")
    print(texto)


if __name__ == "__main__":
    main()
