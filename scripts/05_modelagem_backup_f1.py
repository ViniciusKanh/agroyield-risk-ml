from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False
    XGBClassifier = None


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


LIMIARES_PADRAO = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


def obter_features(df: pd.DataFrame, tipo: str) -> list[str]:
    """
    Retorna a lista de variáveis explicativas existentes no dataset.

    Observação metodológica:
    - Não inclui rendimento_medio_kg_ha atual.
    - Isso evita vazamento de dados, pois o rendimento atual participa da construção do alvo.
    """
    base = FEATURES_EXPANDIDAS if tipo == "expandido" else FEATURES_CONSERVADORAS
    return [col for col in base if col in df.columns]


def calcular_scale_pos_weight(y: pd.Series) -> float:
    """
    Calcula a razão entre classe negativa e positiva.

    Essa razão é usada no XGBoost para lidar com desbalanceamento.
    """
    qtd_negativos = int((y == 0).sum())
    qtd_positivos = int((y == 1).sum())

    if qtd_positivos == 0:
        return 1.0

    return qtd_negativos / qtd_positivos


def calcular_pesos_amostrais_balanceados(y: pd.Series) -> np.ndarray:
    """
    Calcula pesos amostrais balanceados para modelos que aceitam sample_weight.

    Usado principalmente no Gradient Boosting, pois ele não possui class_weight nativo.
    """
    y_array = y.to_numpy()
    classes, contagens = np.unique(y_array, return_counts=True)

    pesos_por_classe = {
        classe: len(y_array) / (len(classes) * contagem)
        for classe, contagem in zip(classes, contagens)
    }

    return np.array([pesos_por_classe[classe] for classe in y_array])


def criar_modelos(random_state: int, scale_pos_weight: float) -> dict[str, Pipeline]:
    """
    Cria modelos comparáveis em pipelines do scikit-learn.

    Os modelos foram configurados para classe desbalanceada e validação temporal.
    """
    modelos: dict[str, Pipeline] = {
        "regressao_logistica": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    LogisticRegression(
                        max_iter=5000,
                        C=0.75,
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
                        n_estimators=800,
                        max_depth=8,
                        min_samples_split=20,
                        min_samples_leaf=8,
                        max_features="sqrt",
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
                    GradientBoostingClassifier(
                        n_estimators=350,
                        learning_rate=0.03,
                        max_depth=3,
                        min_samples_leaf=15,
                        subsample=0.85,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if XGBOOST_DISPONIVEL and XGBClassifier is not None:
        modelos["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=3,
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return modelos


def ajustar_modelo(
    nome: str,
    modelo: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Ajusta o modelo.

    Para Gradient Boosting, aplica sample_weight balanceado.
    Para os demais, usa class_weight ou scale_pos_weight configurado no estimador.
    """
    if nome == "gradient_boosting":
        pesos = calcular_pesos_amostrais_balanceados(y_train)
        modelo.fit(x_train, y_train, modelo__sample_weight=pesos)
    else:
        modelo.fit(x_train, y_train)

    return modelo


def obter_probabilidade_classe_positiva(
    modelo: Pipeline,
    x: pd.DataFrame,
) -> np.ndarray:
    """Obtém a probabilidade prevista para risco_quebra = 1."""
    if not hasattr(modelo, "predict_proba"):
        raise RuntimeError("O modelo não possui predict_proba.")

    return modelo.predict_proba(x)[:, 1]


def aplicar_limiar(y_score: np.ndarray, limiar: float) -> np.ndarray:
    """Converte probabilidades em classes usando um limiar definido."""
    return (y_score >= limiar).astype(int)


def calcular_metricas_binarias(
    nome: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    limiar: float,
    conjunto: str,
) -> dict:
    """Calcula métricas binárias para um dado modelo, conjunto e limiar."""
    y_pred = aplicar_limiar(y_score, limiar)

    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
    else:
        roc_auc = np.nan
        average_precision = np.nan

    return {
        "conjunto": conjunto,
        "modelo": nome,
        "limiar": float(limiar),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def avaliar_limiares(
    nome: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    limiares: list[float],
    conjunto: str,
) -> pd.DataFrame:
    """Avalia vários limiares para um modelo."""
    resultados = [
        calcular_metricas_binarias(
            nome=nome,
            y_true=y_true,
            y_score=y_score,
            limiar=limiar,
            conjunto=conjunto,
        )
        for limiar in limiares
    ]

    return pd.DataFrame(resultados)


def escolher_limiar_validacao(
    metricas_validacao: pd.DataFrame,
    recall_minimo: float,
    precision_minima: float,
) -> tuple[pd.Series, str]:
    """
    Escolhe o limiar com base apenas no conjunto de validação.

    Critério:
    1. Preferir limiares com recall >= recall_minimo e precision >= precision_minima.
    2. Entre eles, escolher maior F1.
    3. Se nenhum atender aos dois critérios, relaxar precision e manter recall mínimo.
    4. Se ainda não houver candidato, escolher maior F1 geral.

    Isso evita escolher limiar usando o conjunto de teste.
    """
    tabela = metricas_validacao.copy()

    tabela["atende_recall_minimo"] = tabela["recall"] >= recall_minimo
    tabela["atende_precision_minima"] = tabela["precision"] >= precision_minima
    tabela["atende_criterio_completo"] = (
        tabela["atende_recall_minimo"] & tabela["atende_precision_minima"]
    )

    candidatos = tabela[tabela["atende_criterio_completo"]].copy()
    criterio = "recall_minimo_e_precision_minima"

    if candidatos.empty:
        candidatos = tabela[tabela["atende_recall_minimo"]].copy()
        criterio = "recall_minimo"

    if candidatos.empty:
        candidatos = tabela.copy()
        criterio = "maior_f1_geral"

    candidatos = candidatos.sort_values(
        ["f1", "balanced_accuracy", "precision", "recall"],
        ascending=[False, False, False, False],
    )

    return candidatos.iloc[0], criterio


def salvar_matriz_confusao(
    nome: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    limiar: float,
) -> None:
    """Salva matriz de confusão usando o limiar selecionado."""
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=["Sem quebra", "Quebra"],
    )
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Matriz de confusão — {nome} | limiar={limiar:.2f}")
    fig.tight_layout()

    caminho = ROOT / "outputs" / "figures" / f"matriz_confusao_{nome}.png"
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def salvar_curvas_modelo(
    nome: str,
    y_true: pd.Series,
    y_score: np.ndarray,
) -> None:
    """Salva curvas ROC e Precision-Recall."""
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
    ax.set_title(f"Curva ROC — {nome}")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.legend()
    fig.tight_layout()

    caminho_roc = ROOT / "outputs" / "figures" / f"curva_roc_{nome}.png"
    fig.savefig(caminho_roc, dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_title(f"Curva Precision-Recall — {nome}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()

    caminho_pr = ROOT / "outputs" / "figures" / f"curva_precision_recall_{nome}.png"
    fig.savefig(caminho_pr, dpi=180)
    plt.close(fig)


def salvar_importancia_interna(
    nome: str,
    modelo: Pipeline,
    features: list[str],
) -> None:
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

    salvar_csv(
        importancia,
        ROOT / "outputs" / "tables" / f"importancia_interna_{nome}.csv",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    top = importancia.head(12).sort_values("importancia", ascending=True)
    ax.barh(top["variavel"], top["importancia"])
    ax.set_title(f"Importância das variáveis — {nome}")
    ax.set_xlabel("Importância interna do modelo")
    fig.tight_layout()

    caminho = ROOT / "outputs" / "figures" / f"importancia_variaveis_{nome}.png"
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def salvar_coeficientes_regressao_logistica(
    nome: str,
    modelo: Pipeline,
    features: list[str],
) -> None:
    """
    Salva coeficientes da regressão logística.

    Como o modelo usa StandardScaler, os coeficientes ficam comparáveis
    em escala padronizada.
    """
    estimador = modelo.named_steps.get("modelo")

    if not isinstance(estimador, LogisticRegression):
        return

    coeficientes = estimador.coef_[0]

    tabela = pd.DataFrame(
        {
            "variavel": features,
            "coeficiente_padronizado": coeficientes,
            "coeficiente_abs": np.abs(coeficientes),
            "direcao_associacao_modelo": np.where(
                coeficientes > 0,
                "aumenta_probabilidade_modelada_de_quebra",
                "reduz_probabilidade_modelada_de_quebra",
            ),
        }
    ).sort_values("coeficiente_abs", ascending=False)

    salvar_csv(
        tabela,
        ROOT / "outputs" / "tables" / f"coeficientes_{nome}.csv",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    top = tabela.head(12).sort_values("coeficiente_padronizado", ascending=True)
    ax.barh(top["variavel"], top["coeficiente_padronizado"])
    ax.axvline(0, linewidth=1)
    ax.set_title(f"Coeficientes padronizados — {nome}")
    ax.set_xlabel("Coeficiente padronizado")
    fig.tight_layout()

    caminho = ROOT / "outputs" / "figures" / f"coeficientes_{nome}.png"
    fig.savefig(caminho, dpi=180)
    plt.close(fig)


def separar_treino_validacao_temporal(
    treino_completo: pd.DataFrame,
    anos_validacao: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa o conjunto de treino em treino interno e validação temporal.

    Exemplo:
    - Treino completo: 2009–2020
    - anos_validacao = 2
    - Treino interno: 2009–2018
    - Validação: 2019–2020
    """
    anos_treino = sorted(treino_completo["ano"].dropna().unique())

    if len(anos_treino) <= anos_validacao + 1:
        raise RuntimeError(
            "Poucos anos disponíveis para criar validação temporal interna. "
            "Reduza anos_validacao ou revise o recorte temporal."
        )

    ano_inicio_validacao = anos_treino[-anos_validacao]

    treino_interno = treino_completo[
        treino_completo["ano"] < ano_inicio_validacao
    ].copy()

    validacao = treino_completo[
        treino_completo["ano"] >= ano_inicio_validacao
    ].copy()

    return treino_interno, validacao


def main() -> None:
    garantir_diretorios()
    config = carregar_config()

    dataset = ler_csv(ROOT / "data" / "processed" / "dataset_agro_ml.csv")

    ano_treino_ate = int(config["modelagem"]["ano_treino_ate"])
    random_state = int(config["modelagem"]["random_state"])
    tipo_features = str(config["modelagem"].get("usar_feature_set", "conservador"))

    anos_validacao = int(config["modelagem"].get("anos_validacao", 2))
    recall_minimo = float(config["modelagem"].get("recall_minimo_selecao", 0.60))
    precision_minima = float(config["modelagem"].get("precision_minima_selecao", 0.20))

    features = obter_features(dataset, tipo_features)

    if not features:
        raise RuntimeError("Nenhuma feature modelável foi encontrada no dataset final.")

    dataset = dataset.sort_values(["ano", "codigo_municipio"]).copy()

    treino_completo = dataset[dataset["ano"] <= ano_treino_ate].copy()
    teste = dataset[dataset["ano"] > ano_treino_ate].copy()

    if treino_completo.empty or teste.empty:
        raise RuntimeError(
            "Divisão temporal gerou treino ou teste vazio. "
            "Ajuste `ano_treino_ate` no config.yaml."
        )

    treino_interno, validacao = separar_treino_validacao_temporal(
        treino_completo=treino_completo,
        anos_validacao=anos_validacao,
    )

    x_train_inner = treino_interno[features]
    y_train_inner = treino_interno["risco_quebra"].astype(int)

    x_val = validacao[features]
    y_val = validacao["risco_quebra"].astype(int)

    x_train_full = treino_completo[features]
    y_train_full = treino_completo["risco_quebra"].astype(int)

    x_test = teste[features]
    y_test = teste["risco_quebra"].astype(int)

    scale_pos_weight = calcular_scale_pos_weight(y_train_full)

    print("=" * 80)
    print("Modelagem temporal de risco de quebra de produtividade")
    print(f"Features usadas ({tipo_features}): {features}")
    print(f"Treino interno: {treino_interno['ano'].min()}–{treino_interno['ano'].max()} | linhas={len(treino_interno)}")
    print(f"Validação: {validacao['ano'].min()}–{validacao['ano'].max()} | linhas={len(validacao)}")
    print(f"Treino completo: {treino_completo['ano'].min()}–{treino_completo['ano'].max()} | linhas={len(treino_completo)}")
    print(f"Teste final: {teste['ano'].min()}–{teste['ano'].max()} | linhas={len(teste)}")
    print(f"Recall mínimo para seleção de limiar: {recall_minimo:.2f}")
    print(f"Precision mínima para seleção de limiar: {precision_minima:.2f}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print("=" * 80)

    if not XGBOOST_DISPONIVEL:
        print("Aviso: XGBoost não está instalado. O modelo xgboost será ignorado.")
        print("Para incluir XGBoost futuramente, rode: pip install -r requirements-optional.txt")

    print("\nDistribuição do alvo no treino interno:")
    print(y_train_inner.value_counts().sort_index())

    print("\nDistribuição do alvo na validação:")
    print(y_val.value_counts().sort_index())

    print("\nDistribuição do alvo no teste final:")
    print(y_test.value_counts().sort_index())

    modelos_base = criar_modelos(
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )

    todas_metricas_validacao = []
    todas_metricas_teste = []
    metricas_finais = []
    modelos_finais = {}
    limiares_escolhidos = {}

    colunas_identificacao = [
        col
        for col in ["codigo_municipio", "municipio", "ano", "risco_quebra"]
        if col in teste.columns
    ]
    previsoes_teste = teste[colunas_identificacao].copy()

    for nome, modelo_base in modelos_base.items():
        print("\n" + "=" * 80)
        print(f"Treinando e validando modelo: {nome}")

        modelo_validacao = clone(modelo_base)
        modelo_validacao = ajustar_modelo(
            nome=nome,
            modelo=modelo_validacao,
            x_train=x_train_inner,
            y_train=y_train_inner,
        )

        y_score_val = obter_probabilidade_classe_positiva(modelo_validacao, x_val)

        metricas_validacao = avaliar_limiares(
            nome=nome,
            y_true=y_val,
            y_score=y_score_val,
            limiares=LIMIARES_PADRAO,
            conjunto="validacao",
        )

        melhor_limiar_validacao, criterio_limiar = escolher_limiar_validacao(
            metricas_validacao=metricas_validacao,
            recall_minimo=recall_minimo,
            precision_minima=precision_minima,
        )

        limiar_escolhido = float(melhor_limiar_validacao["limiar"])

        print(f"Limiar escolhido por validação: {limiar_escolhido:.2f}")
        print(f"Critério de escolha: {criterio_limiar}")
        print("Métricas na validação com limiar escolhido:")
        print(
            melhor_limiar_validacao[
                ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
            ]
        )

        todas_metricas_validacao.append(metricas_validacao)

        modelo_final = clone(modelo_base)
        modelo_final = ajustar_modelo(
            nome=nome,
            modelo=modelo_final,
            x_train=x_train_full,
            y_train=y_train_full,
        )

        modelos_finais[nome] = modelo_final
        limiares_escolhidos[nome] = limiar_escolhido

        y_score_test = obter_probabilidade_classe_positiva(modelo_final, x_test)

        metricas_teste_limiares = avaliar_limiares(
            nome=nome,
            y_true=y_test,
            y_score=y_score_test,
            limiares=LIMIARES_PADRAO,
            conjunto="teste",
        )

        todas_metricas_teste.append(metricas_teste_limiares)

        metricas_teste_limiar_escolhido = metricas_teste_limiares[
            np.isclose(metricas_teste_limiares["limiar"], limiar_escolhido)
        ].iloc[0].to_dict()

        metricas_teste_limiar_escolhido.update(
            {
                "limiar_decisao": limiar_escolhido,
                "criterio_limiar": criterio_limiar,
                "accuracy_validacao": float(melhor_limiar_validacao["accuracy"]),
                "balanced_accuracy_validacao": float(
                    melhor_limiar_validacao["balanced_accuracy"]
                ),
                "precision_validacao": float(melhor_limiar_validacao["precision"]),
                "recall_validacao": float(melhor_limiar_validacao["recall"]),
                "f1_validacao": float(melhor_limiar_validacao["f1"]),
                "roc_auc_validacao": float(melhor_limiar_validacao["roc_auc"]),
                "average_precision_validacao": float(
                    melhor_limiar_validacao["average_precision"]
                ),
            }
        )

        metricas_finais.append(metricas_teste_limiar_escolhido)

        y_pred_test = aplicar_limiar(y_score_test, limiar_escolhido)
        y_pred_test_050 = aplicar_limiar(y_score_test, 0.50)

        previsoes_teste[f"prob_quebra_{nome}"] = y_score_test
        previsoes_teste[f"pred_{nome}"] = y_pred_test
        previsoes_teste[f"pred_limiar_050_{nome}"] = y_pred_test_050

        salvar_matriz_confusao(
            nome=nome,
            y_true=y_test,
            y_pred=y_pred_test,
            limiar=limiar_escolhido,
        )

        salvar_curvas_modelo(
            nome=nome,
            y_true=y_test,
            y_score=y_score_test,
        )

        salvar_importancia_interna(
            nome=nome,
            modelo=modelo_final,
            features=features,
        )

        salvar_coeficientes_regressao_logistica(
            nome=nome,
            modelo=modelo_final,
            features=features,
        )

        relatorio = classification_report(
            y_test,
            y_pred_test,
            labels=[0, 1],
            target_names=["Sem quebra", "Quebra"],
            zero_division=0,
        )

        caminho_relatorio = ROOT / "outputs" / "tables" / f"classification_report_{nome}.txt"
        caminho_relatorio.write_text(relatorio, encoding="utf-8")

        print("\nRelatório no teste final com limiar escolhido:")
        print(relatorio)

    metricas_validacao_df = pd.concat(todas_metricas_validacao, ignore_index=True)
    metricas_teste_df = pd.concat(todas_metricas_teste, ignore_index=True)

    metricas_por_limiar = pd.concat(
        [metricas_validacao_df, metricas_teste_df],
        ignore_index=True,
    )

    salvar_csv(
        metricas_validacao_df,
        ROOT / "outputs" / "tables" / "metricas_por_limiar_validacao.csv",
    )
    salvar_csv(
        metricas_teste_df,
        ROOT / "outputs" / "tables" / "metricas_por_limiar_teste.csv",
    )
    salvar_csv(
        metricas_por_limiar,
        ROOT / "outputs" / "tables" / "metricas_por_limiar.csv",
    )

    metricas_df = pd.DataFrame(metricas_finais)

    ranking_validacao = metricas_df.sort_values(
        [
            "f1_validacao",
            "balanced_accuracy_validacao",
            "recall_validacao",
            "precision_validacao",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    melhor_nome = str(ranking_validacao.iloc[0]["modelo"])
    melhor_modelo = modelos_finais[melhor_nome]
    melhor_limiar = float(limiares_escolhidos[melhor_nome])

    metricas_df["modelo_selecionado"] = metricas_df["modelo"] == melhor_nome

    metricas_df = metricas_df.sort_values(
        [
            "modelo_selecionado",
            "f1",
            "balanced_accuracy",
            "recall",
            "precision",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    salvar_csv(
        metricas_df,
        ROOT / "outputs" / "tables" / "metricas_modelos.csv",
    )

    metricas_limiar_050 = metricas_teste_df[
        np.isclose(metricas_teste_df["limiar"], 0.50)
    ].copy()

    salvar_csv(
        metricas_limiar_050,
        ROOT / "outputs" / "tables" / "metricas_modelos_limiar_050.csv",
    )

    salvar_csv(
        previsoes_teste,
        ROOT / "outputs" / "tables" / "previsoes_teste.csv",
    )

    joblib.dump(
        {
            "modelo": melhor_modelo,
            "features": features,
            "melhor_nome": melhor_nome,
            "limiar_decisao": melhor_limiar,
            "tipo_features": tipo_features,
        },
        ROOT / "models" / "melhor_modelo.joblib",
    )

    joblib.dump(
        {
            "modelos": modelos_finais,
            "features": features,
            "limiares_escolhidos": limiares_escolhidos,
            "tipo_features": tipo_features,
        },
        ROOT / "models" / "modelos_treinados.joblib",
    )

    metadados = {
        "melhor_modelo": melhor_nome,
        "limiar_decisao": melhor_limiar,
        "features": features,
        "ano_treino_ate": ano_treino_ate,
        "anos_validacao": anos_validacao,
        "periodo_treino_interno": [
            int(treino_interno["ano"].min()),
            int(treino_interno["ano"].max()),
        ],
        "periodo_validacao": [
            int(validacao["ano"].min()),
            int(validacao["ano"].max()),
        ],
        "periodo_treino_completo": [
            int(treino_completo["ano"].min()),
            int(treino_completo["ano"].max()),
        ],
        "periodo_teste": [
            int(teste["ano"].min()),
            int(teste["ano"].max()),
        ],
        "tipo_features": tipo_features,
        "recall_minimo_selecao": recall_minimo,
        "precision_minima_selecao": precision_minima,
        "limiares_avaliados": LIMIARES_PADRAO,
        "xgboost_disponivel": XGBOOST_DISPONIVEL,
        "scale_pos_weight": scale_pos_weight,
        "observacao_metodologica": (
            "Modelo e limiar selecionados por validação temporal interna. "
            "O teste final foi mantido separado para avaliação temporal fora da amostra."
        ),
    }

    (ROOT / "models" / "metadata_modelagem.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Métricas finais no teste, usando limiar escolhido por validação:")
    print(
        metricas_df[
            [
                "modelo",
                "modelo_selecionado",
                "limiar_decisao",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ]
    )

    print("\nMelhor modelo selecionado por validação temporal:")
    print(f"Modelo: {melhor_nome}")
    print(f"Limiar de decisão: {melhor_limiar:.2f}")
    print(f"Modelo salvo em: {ROOT / 'models' / 'melhor_modelo.joblib'}")
    print("=" * 80)


if __name__ == "__main__":
    main()