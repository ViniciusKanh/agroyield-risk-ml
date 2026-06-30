from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import ler_csv  # noqa: E402


def carregar_json(caminho: Path) -> dict:
    """Carrega arquivo JSON, retornando dicionário vazio se não existir."""
    if not caminho.exists():
        return {}

    return json.loads(caminho.read_text(encoding="utf-8"))


def obter_valor(linha: pd.Series, coluna: str, padrao: Any = np.nan) -> Any:
    """Obtém valor de uma coluna em uma linha."""
    if coluna in linha.index:
        return linha[coluna]

    return padrao


def formatar_float(valor: Any, casas: int = 4) -> str:
    """Formata número real com segurança."""
    try:
        if pd.isna(valor):
            return "NA"

        return f"{float(valor):.{casas}f}"
    except Exception:
        return "NA"


def formatar_int(valor: Any) -> str:
    """Formata número inteiro com segurança."""
    try:
        if pd.isna(valor):
            return "NA"

        return str(int(valor))
    except Exception:
        return "NA"


def formatar_periodo(valor: Any) -> str:
    """Formata períodos salvos nos metadados."""
    if isinstance(valor, list) and len(valor) == 2:
        return f"{valor[0]}–{valor[1]}"

    return str(valor)


def selecionar_modelo_validacao(metricas: pd.DataFrame) -> pd.Series:
    """
    Seleciona o modelo marcado como modelo_selecionado.

    Compatível com CSV onde booleano pode vir como True, False, 'True' ou 'False'.
    """
    if "modelo_selecionado" in metricas.columns:
        coluna = metricas["modelo_selecionado"].astype(str).str.lower()
        selecionados = metricas[coluna == "true"]

        if not selecionados.empty:
            return selecionados.iloc[0]

    return metricas.iloc[0]


def selecionar_melhor_teste(metricas: pd.DataFrame) -> pd.Series:
    """
    Seleciona o melhor modelo no teste final.

    Prioridade:
    1. Maior F2-score;
    2. Maior recall;
    3. Maior F1-score;
    4. Maior balanced accuracy;
    5. Maior precision.
    """
    colunas_ordenacao = []
    ascendentes = []

    for coluna in ["f2", "recall", "f1", "balanced_accuracy", "precision"]:
        if coluna in metricas.columns:
            colunas_ordenacao.append(coluna)
            ascendentes.append(False)

    if not colunas_ordenacao:
        return metricas.iloc[0]

    return metricas.sort_values(
        colunas_ordenacao,
        ascending=ascendentes,
    ).iloc[0]


def criar_tabela_resumo_modelos(metricas: pd.DataFrame) -> pd.DataFrame:
    """Cria tabela enxuta de comparação dos modelos."""
    colunas_desejadas = [
        "modelo",
        "modelo_selecionado",
        "limiar_decisao",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "average_precision",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    colunas_existentes = [
        coluna for coluna in colunas_desejadas if coluna in metricas.columns
    ]

    return metricas[colunas_existentes].copy()


def criar_tabela_importancia(importancia: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Retorna as principais variáveis por importância.

    Ordena automaticamente pela primeira coluna de importância média encontrada.
    """
    if importancia.empty:
        return pd.DataFrame()

    colunas_prioridade = [
        "importancia_media_f2",
        "importancia_media_f1",
        "importancia_media",
        "importancia",
    ]

    coluna_ordenacao = None
    for coluna in colunas_prioridade:
        if coluna in importancia.columns:
            coluna_ordenacao = coluna
            break

    if coluna_ordenacao is not None:
        importancia = importancia.sort_values(coluna_ordenacao, ascending=False)

    return importancia.head(n).copy()


def identificar_metrica_importancia(importancia: pd.DataFrame) -> str:
    """Identifica qual métrica foi usada na tabela de importância."""
    if "importancia_media_f2" in importancia.columns:
        return "F2-score"

    if "importancia_media_f1" in importancia.columns:
        return "F1-score"

    if "importancia_media" in importancia.columns:
        return "métrica configurada no script de interpretabilidade"

    if "importancia" in importancia.columns:
        return "importância interna do modelo"

    return "não identificada"


def criar_texto_matriz_confusao(modelo: pd.Series) -> str:
    """Cria matriz de confusão textual em Markdown."""
    tn = formatar_int(obter_valor(modelo, "tn"))
    fp = formatar_int(obter_valor(modelo, "fp"))
    fn = formatar_int(obter_valor(modelo, "fn"))
    tp = formatar_int(obter_valor(modelo, "tp"))

    linhas = [
        "| Classe real / predita | Sem quebra | Quebra |",
        "|:--|--:|--:|",
        f"| Sem quebra | {tn} | {fp} |",
        f"| Quebra | {fn} | {tp} |",
    ]

    return "\n".join(linhas)


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    dataset = ler_csv(ROOT / "data" / "processed" / "dataset_agro_ml.csv")
    metricas = ler_csv(ROOT / "outputs" / "tables" / "metricas_modelos.csv")

    caminho_importancia = ROOT / "outputs" / "tables" / "importancia_variaveis.csv"
    if caminho_importancia.exists():
        importancia = ler_csv(caminho_importancia)
    else:
        importancia = pd.DataFrame()

    metadata_path = ROOT / "models" / "metadata_modelagem.json"
    metadata = carregar_json(metadata_path)

    modelo_validacao = selecionar_modelo_validacao(metricas)
    modelo_melhor_teste = selecionar_melhor_teste(metricas)

    tabela_modelos = criar_tabela_resumo_modelos(metricas)
    top_importancia = criar_tabela_importancia(importancia, n=10)
    metrica_importancia = identificar_metrica_importancia(importancia)

    cultura = config["projeto"]["cultura"]
    estado = config["projeto"]["estado_nome"]
    limiar_quebra = float(config["alvo"]["limiar_quebra"]) * 100
    janela = config["alvo"]["janela_media_movel"]

    periodo_dataset = f"{int(dataset['ano'].min())}–{int(dataset['ano'].max())}"
    total_municipios = dataset["codigo_municipio"].nunique()
    total_linhas = len(dataset)
    distribuicao_alvo = dataset["risco_quebra"].value_counts().sort_index()

    modelo_validacao_nome = str(obter_valor(modelo_validacao, "modelo", "NA"))
    modelo_teste_nome = str(obter_valor(modelo_melhor_teste, "modelo", "NA"))

    mesmo_modelo = modelo_validacao_nome == modelo_teste_nome

    if mesmo_modelo:
        observacao_modelos = (
            "O modelo selecionado pela validação temporal também apresentou o "
            "maior F2-score no teste final."
        )
    else:
        observacao_modelos = (
            f"O modelo selecionado pela validação temporal foi `{modelo_validacao_nome}`. "
            f"Entretanto, no teste temporal final, o maior F2-score foi observado em "
            f"`{modelo_teste_nome}`. Essa diferença deve ser relatada como evidência "
            "de instabilidade temporal e como limitação experimental."
        )

    if not top_importancia.empty:
        texto_importancia = top_importancia.to_markdown(index=False)
    else:
        texto_importancia = (
            "Tabela de importância não encontrada. Execute "
            "`python scripts/06_interpretabilidade.py` para gerar "
            "`outputs/tables/importancia_variaveis.csv`."
        )

    linhas: list[str] = []

    linhas.append("# Resumo dos resultados experimentais")
    linhas.append("")
    linhas.append("## 1. Recorte experimental")
    linhas.append("")
    linhas.append(f"- Cultura: {cultura}")
    linhas.append(f"- Estado: {estado}")
    linhas.append(f"- Período analisado após criação do alvo: {periodo_dataset}")
    linhas.append(f"- Municípios analisados: {total_municipios}")
    linhas.append(f"- Linhas município-ano: {total_linhas}")
    linhas.append("")
    linhas.append(
        "A base final representa observações no nível município-ano. "
        "Os anos iniciais anteriores ao período modelável foram utilizados "
        "para calcular o histórico necessário à variável-alvo."
    )
    linhas.append("")

    linhas.append("## 2. Variável-alvo")
    linhas.append("")
    linhas.append(
        f"A variável `risco_quebra` foi definida como 1 quando o rendimento "
        f"médio do município no ano ficou abaixo de {limiar_quebra:.0f}% da "
        f"média móvel dos {janela} anos anteriores."
    )
    linhas.append("")
    linhas.append("Distribuição do alvo:")
    linhas.append("")
    linhas.append("```text")
    linhas.append(distribuicao_alvo.to_string())
    linhas.append("```")
    linhas.append("")
    linhas.append(
        "Essa distribuição indica desbalanceamento entre as classes, com "
        "predominância de anos sem quebra. Por isso, a avaliação não deve "
        "ser baseada apenas em acurácia."
    )
    linhas.append("")

    linhas.append("## 3. Estratégia de validação")
    linhas.append("")
    linhas.append("A modelagem adotou separação temporal:")
    linhas.append("")
    linhas.append(
        f"- Treino interno: {formatar_periodo(metadata.get('periodo_treino_interno', 'NA'))}"
    )
    linhas.append(
        f"- Validação temporal: {formatar_periodo(metadata.get('periodo_validacao', 'NA'))}"
    )
    linhas.append(
        f"- Treino completo: {formatar_periodo(metadata.get('periodo_treino_completo', 'NA'))}"
    )
    linhas.append(
        f"- Teste temporal final: {formatar_periodo(metadata.get('periodo_teste', 'NA'))}"
    )
    linhas.append("")
    linhas.append(
        "O modelo e o limiar de decisão foram escolhidos com base na validação "
        "temporal interna. O teste final foi preservado para avaliação fora da amostra."
    )
    linhas.append("")
    linhas.append("Métrica principal de seleção:")
    linhas.append("")
    linhas.append("```text")
    linhas.append(str(metadata.get("metrica_principal_selecao", "não informado")))
    linhas.append("```")
    linhas.append("")
    linhas.append("Justificativa:")
    linhas.append("")
    linhas.append("```text")
    linhas.append(str(metadata.get("justificativa_metrica_principal", "não informado")))
    linhas.append("```")
    linhas.append("")

    linhas.append("## 4. Modelo selecionado pela validação temporal")
    linhas.append("")
    linhas.append(f"- Modelo: {modelo_validacao_nome}")
    linhas.append(
        f"- Limiar de decisão: {formatar_float(obter_valor(modelo_validacao, 'limiar_decisao'))}"
    )
    linhas.append(
        f"- Precision: {formatar_float(obter_valor(modelo_validacao, 'precision'))}"
    )
    linhas.append(
        f"- Recall: {formatar_float(obter_valor(modelo_validacao, 'recall'))}"
    )
    linhas.append(
        f"- F1-score: {formatar_float(obter_valor(modelo_validacao, 'f1'))}"
    )
    linhas.append(
        f"- F2-score: {formatar_float(obter_valor(modelo_validacao, 'f2'))}"
    )
    linhas.append(
        f"- Balanced accuracy: {formatar_float(obter_valor(modelo_validacao, 'balanced_accuracy'))}"
    )
    linhas.append(
        f"- ROC AUC: {formatar_float(obter_valor(modelo_validacao, 'roc_auc'))}"
    )
    linhas.append(
        f"- Average precision: {formatar_float(obter_valor(modelo_validacao, 'average_precision'))}"
    )
    linhas.append("")
    linhas.append("Matriz de confusão no teste final para o modelo selecionado:")
    linhas.append("")
    linhas.append(criar_texto_matriz_confusao(modelo_validacao))
    linhas.append("")
    linhas.append(
        "Interpretação: o modelo selecionado pela validação temporal deve ser "
        "tratado como o modelo principal do desenho experimental, pois sua escolha "
        "não usou o conjunto de teste."
    )
    linhas.append("")

    linhas.append("## 5. Melhor desempenho observado no teste final")
    linhas.append("")
    linhas.append(f"- Modelo: {modelo_teste_nome}")
    linhas.append(
        f"- Limiar de decisão: {formatar_float(obter_valor(modelo_melhor_teste, 'limiar_decisao'))}"
    )
    linhas.append(
        f"- Precision: {formatar_float(obter_valor(modelo_melhor_teste, 'precision'))}"
    )
    linhas.append(
        f"- Recall: {formatar_float(obter_valor(modelo_melhor_teste, 'recall'))}"
    )
    linhas.append(
        f"- F1-score: {formatar_float(obter_valor(modelo_melhor_teste, 'f1'))}"
    )
    linhas.append(
        f"- F2-score: {formatar_float(obter_valor(modelo_melhor_teste, 'f2'))}"
    )
    linhas.append(
        f"- Balanced accuracy: {formatar_float(obter_valor(modelo_melhor_teste, 'balanced_accuracy'))}"
    )
    linhas.append(
        f"- ROC AUC: {formatar_float(obter_valor(modelo_melhor_teste, 'roc_auc'))}"
    )
    linhas.append(
        f"- Average precision: {formatar_float(obter_valor(modelo_melhor_teste, 'average_precision'))}"
    )
    linhas.append("")
    linhas.append("Matriz de confusão no teste final para o modelo com maior F2-score:")
    linhas.append("")
    linhas.append(criar_texto_matriz_confusao(modelo_melhor_teste))
    linhas.append("")
    linhas.append("Observação metodológica:")
    linhas.append("")
    linhas.append(observacao_modelos)
    linhas.append("")

    linhas.append("## 6. Comparação dos modelos no teste temporal final")
    linhas.append("")
    linhas.append(tabela_modelos.to_markdown(index=False))
    linhas.append("")

    linhas.append("## 7. Variáveis mais importantes")
    linhas.append("")
    linhas.append(f"Métrica usada na importância: **{metrica_importancia}**.")
    linhas.append("")
    linhas.append(texto_importancia)
    linhas.append("")
    linhas.append(
        "A importância das variáveis deve ser interpretada como evidência de "
        "contribuição preditiva para o modelo, não como prova de causalidade."
    )
    linhas.append("")

    linhas.append("## 8. Interpretação científica preliminar")
    linhas.append("")
    linhas.append(
        "Os resultados indicam que variáveis climáticas e histórico produtivo "
        "municipal possuem sinal informativo para classificar anos de risco de "
        "quebra de produtividade da soja no Paraná."
    )
    linhas.append("")
    linhas.append(
        "A hipótese inicial é parcialmente sustentada de forma exploratória, pois "
        "variáveis relacionadas à precipitação, temperatura, extremos climáticos "
        "e produtividade histórica aparecem entre os atributos relevantes para a classificação."
    )
    linhas.append("")
    linhas.append(
        "No entanto, os resultados também indicam limitação importante: os modelos "
        "alcançam recall relativamente alto para eventos de quebra quando o limiar "
        "é ajustado, mas apresentam baixa precision, gerando quantidade elevada de "
        "falsos positivos. Isso caracteriza o pipeline como mais adequado para "
        "triagem e alerta exploratório do que para decisão operacional isolada."
    )
    linhas.append("")

    linhas.append("## 9. Cuidados para o CONICT e para artigo")
    linhas.append("")
    linhas.append("- Não afirmar causalidade direta entre clima e quebra de produtividade.")
    linhas.append("- Relatar os resultados como associações estatísticas e computacionais exploratórias.")
    linhas.append("- Explicar a construção da variável-alvo baseada em média móvel histórica.")
    linhas.append("- Informar que a divisão treino-teste respeitou a ordem temporal.")
    linhas.append("- Destacar que o F2-score foi usado por priorizar recall em um problema de triagem de risco.")
    linhas.append("- Discutir a baixa precision como limitação central.")
    linhas.append("- Relatar que os dados são agregados por município e ano.")
    linhas.append(
        "- Apontar ausência de variáveis de solo, manejo, cultivar, pragas, crédito rural e tecnologia agrícola."
    )
    linhas.append(
        "- Evitar uso de `rendimento_medio_kg_ha` atual como variável explicativa, pois isso causaria vazamento de dados."
    )
    linhas.append("")

    linhas.append("## 10. Arquivos gerados")
    linhas.append("")
    linhas.append("- `data/processed/dataset_agro_ml.csv`")
    linhas.append("- `outputs/tables/metricas_modelos.csv`")
    linhas.append("- `outputs/tables/metricas_modelos_limiar_050.csv`")
    linhas.append("- `outputs/tables/metricas_por_limiar.csv`")
    linhas.append("- `outputs/tables/metricas_por_limiar_validacao.csv`")
    linhas.append("- `outputs/tables/metricas_por_limiar_teste.csv`")
    linhas.append("- `outputs/tables/importancia_variaveis.csv`")
    linhas.append("- `outputs/tables/previsoes_teste.csv`")
    linhas.append("- `outputs/figures/matriz_confusao_*.png`")
    linhas.append("- `outputs/figures/curva_roc_*.png`")
    linhas.append("- `outputs/figures/curva_precision_recall_*.png`")
    linhas.append("- `outputs/figures/importancia_variaveis_*.png`")
    linhas.append("- `models/melhor_modelo.joblib`")
    linhas.append("- `models/modelos_treinados.joblib`")
    linhas.append("- `models/metadata_modelagem.json`")
    linhas.append("")

    linhas.append("## 11. Metadados da modelagem")
    linhas.append("")
    linhas.append("```json")
    linhas.append(json.dumps(metadata, ensure_ascii=False, indent=2))
    linhas.append("```")
    linhas.append("")

    texto = "\n".join(linhas)

    caminho = ROOT / "reports" / "resumo_resultados.md"
    caminho.write_text(texto, encoding="utf-8")

    print(f"Resumo salvo em: {caminho}")
    print(texto)


if __name__ == "__main__":
    main()