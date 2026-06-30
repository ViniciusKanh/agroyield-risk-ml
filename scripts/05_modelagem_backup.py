from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conict_agro_ml.config import carregar_config, garantir_diretorios  # noqa: E402
from conict_agro_ml.io import ler_csv, salvar_csv  # noqa: E402


FEATURES_CONSERVADORAS = [
    "area_plantada_ha",
    "produtividade_lag1",
    "media_movel_3anos",
    "chuva_total_mm",
    "chuva_media_diaria_mm",
    "temperatura_media_c",
    "temperatura_maxima_media_c",
    "temperatura_minima_media_c",
    "umidade_media_pct",
    "vento_medio_ms",
    "dias_secos",
    "dias_chuva_extrema",
    "dias_calor_extremo",
]

FEATURES_EXPANDIDAS = FEATURES_CONSERVADORAS + [
    "area_colhida_ha",
]


def obter_features(df: pd.DataFrame, tipo: str) -> list[str]:
    """Retorna lista de variáveis existentes no dataset."""
    base = FEATURES_EXPANDIDAS if tipo == "expandido" else FEATURES_CONSERVADORAS
    return [col for col in base if col in df.columns]


def criar_modelos(random_state: int) -> dict[str, Pipeline]:
    """Cria modelos comparáveis em pipelines do scikit-learn."""
    return {
        "regressao_logistica": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    GradientBoostingClassifier(random_state=random_state),
                ),
            ]
        ),
    }


def avaliar_modelo(nome: str, modelo: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Avalia um modelo e retorna métricas principais."""
    y_pred = modelo.predict(x_test)

    if hasattr(modelo, "predict_proba"):
        y_score = modelo.predict_proba(x_test)[:, 1]
    else:
        y_score = None

    metricas = {
        "modelo": nome,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_score is not None and len(np.unique(y_test)) == 2:
        metricas["roc_auc"] = roc_auc_score(y_test, y_score)
    else:
        metricas["roc_auc"] = np.nan

    return metricas


def salvar_matriz_confusao(nome: str, modelo: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Salva matriz de confusão do modelo."""
    y_pred = modelo.predict(x_test)
    matriz = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Sem quebra", "Quebra"])
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Matriz de confusão — {nome}")
    fig.tight_layout()

    caminho = ROOT / "outputs" / "figures" / f"matriz_confusao_{nome}.png"
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def salvar_importancia_rf(nome: str, modelo: Pipeline, features: list[str]) -> None:
    """Salva importância interna caso o estimador possua feature_importances_."""
    estimador = modelo.named_steps.get("modelo")
    if not hasattr(estimador, "feature_importances_"):
        return

    importancia = pd.DataFrame(
        {
            "variavel": features,
            "importancia": estimador.feature_importances_,
        }
    ).sort_values("importancia", ascending=False)

    salvar_csv(importancia, ROOT / "outputs" / "tables" / f"importancia_interna_{nome}.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    top = importancia.head(12).sort_values("importancia", ascending=True)
    ax.barh(top["variavel"], top["importancia"])
    ax.set_title(f"Importância das variáveis — {nome}")
    ax.set_xlabel("Importância interna do modelo")
    fig.tight_layout()

    caminho = ROOT / "outputs" / "figures" / f"importancia_variaveis_{nome}.png"
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    dataset = ler_csv(ROOT / "data" / "processed" / "dataset_agro_ml.csv")
    ano_treino_ate = int(config["modelagem"]["ano_treino_ate"])
    random_state = int(config["modelagem"]["random_state"])
    tipo_features = str(config["modelagem"].get("usar_feature_set", "conservador"))

    features = obter_features(dataset, tipo_features)
    if not features:
        raise RuntimeError("Nenhuma feature modelável foi encontrada no dataset final.")

    treino = dataset[dataset["ano"] <= ano_treino_ate].copy()
    teste = dataset[dataset["ano"] > ano_treino_ate].copy()

    if treino.empty or teste.empty:
        raise RuntimeError(
            "Divisão temporal gerou treino ou teste vazio. Ajuste `ano_treino_ate` no config.yaml."
        )

    x_train = treino[features]
    y_train = treino["risco_quebra"].astype(int)
    x_test = teste[features]
    y_test = teste["risco_quebra"].astype(int)

    print(f"Features usadas ({tipo_features}): {features}")
    print(f"Treino: {treino['ano'].min()}–{treino['ano'].max()} | linhas={len(treino)}")
    print(f"Teste: {teste['ano'].min()}–{teste['ano'].max()} | linhas={len(teste)}")
    print("Distribuição do alvo no treino:")
    print(y_train.value_counts().sort_index())
    print("Distribuição do alvo no teste:")
    print(y_test.value_counts().sort_index())

    modelos = criar_modelos(random_state=random_state)
    metricas = []
    previsoes_teste = teste[["codigo_municipio", "municipio", "ano", "risco_quebra"]].copy()

    for nome, modelo in modelos.items():
        print(f"\nTreinando: {nome}")
        modelo.fit(x_train, y_train)
        metricas_modelo = avaliar_modelo(nome, modelo, x_test, y_test)
        metricas.append(metricas_modelo)

        y_pred = modelo.predict(x_test)
        previsoes_teste[f"pred_{nome}"] = y_pred
        if hasattr(modelo, "predict_proba"):
            previsoes_teste[f"prob_quebra_{nome}"] = modelo.predict_proba(x_test)[:, 1]

        salvar_matriz_confusao(nome, modelo, x_test, y_test)
        salvar_importancia_rf(nome, modelo, features)

        relatorio = classification_report(y_test, y_pred, zero_division=0)
        caminho_relatorio = ROOT / "outputs" / "tables" / f"classification_report_{nome}.txt"
        caminho_relatorio.write_text(relatorio, encoding="utf-8")
        print(relatorio)

    metricas_df = pd.DataFrame(metricas).sort_values("f1", ascending=False)
    salvar_csv(metricas_df, ROOT / "outputs" / "tables" / "metricas_modelos.csv")
    salvar_csv(previsoes_teste, ROOT / "outputs" / "tables" / "previsoes_teste.csv")

    melhor_nome = metricas_df.iloc[0]["modelo"]
    melhor_modelo = modelos[melhor_nome]
    joblib.dump(
        {"modelo": melhor_modelo, "features": features, "melhor_nome": melhor_nome},
        ROOT / "models" / "melhor_modelo.joblib",
    )

    metadados = {
        "melhor_modelo": melhor_nome,
        "features": features,
        "ano_treino_ate": ano_treino_ate,
        "tipo_features": tipo_features,
    }
    (ROOT / "models" / "metadata_modelagem.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMétricas finais:")
    print(metricas_df)
    print(f"\nMelhor modelo salvo: {melhor_nome}")


if __name__ == "__main__":
    main()
