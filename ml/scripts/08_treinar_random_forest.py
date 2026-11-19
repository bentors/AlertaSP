
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline


BASE = Path(__file__).resolve().parents[1]

DADOS = BASE / "dados" / "modelagem"
RESULTADOS = BASE / "resultados"
MODELOS = BASE / "modelos"

RESULTADOS.mkdir(parents=True, exist_ok=True)
MODELOS.mkdir(parents=True, exist_ok=True)

ALVO = "alvo_registro_amanha"

FEATURES = [
    "chuva_mm_a701",
    "temperatura_media_c_a701",
    "umidade_media_pct_a701",
    "chuva_acumulada_3d_a701",
    "chuva_acumulada_7d_a701",
    "chuva_mm_a771",
    "temperatura_media_c_a771",
    "umidade_media_pct_a771",
    "chuva_acumulada_3d_a771",
    "chuva_acumulada_7d_a771",
    "dia_ano_seno",
    "dia_ano_cosseno",
]


def carregar_dados():
    treino = pd.read_csv(
        DADOS / "treino_2022_2024.csv",
        parse_dates=["data"],
    )

    validacao = pd.read_csv(
        DADOS / "teste_2025.csv",
        parse_dates=["data"],
    )

    treino = treino.sort_values("data")
    validacao = validacao.sort_values("data")

    if treino["data"].max() >= validacao["data"].min():
        raise ValueError(
            "Treino e teste possuem períodos sobrepostos."
        )

    for nome, df in [
        ("treino", treino),
        ("teste", validacao),
    ]:
        faltantes = set(FEATURES + [ALVO]) - set(df.columns)

        if faltantes:
            raise ValueError(
                f"Colunas ausentes em {nome}: {faltantes}"
            )

        if df[ALVO].isna().any():
            raise ValueError(
                f"Existem valores ausentes no alvo de {nome}."
            )

    X_treino = treino[FEATURES]
    y_treino = treino[ALVO].astype(int)

    X_validacao = validacao[FEATURES]
    y_validacao = validacao[ALVO].astype(int)

    return X_treino, y_treino, X_validacao, y_validacao


def construir_pipeline():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "modelo",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=1,
            ),
        ),
    ])


def treinar_modelo(X, y):
    print("\n=== CONFIGURAÇÃO DO TREINAMENTO ===")

    pipeline = construir_pipeline()

    # 3 x 2 x 2 = 12 combinações
    parametros = {
        "modelo__n_estimators": [100, 200, 300],
        "modelo__max_depth": [5, 10],
        "modelo__min_samples_leaf": [2, 5],
    }

    print(
        "Combinações:",
        len(ParameterGrid(parametros)),
    )

    # Validação temporal somente no treinamento
    validacao_temporal = TimeSeriesSplit(
        n_splits=4,
        gap=1,
    )

    busca = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring="average_precision",
        cv=validacao_temporal,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        refit=True,
    )

    busca.fit(X, y)

    print("\n=== MELHORES HIPERPARÂMETROS ===")
    print(busca.best_params_)

    print(
        "\nMelhor PR-AUC médio na validação cruzada:",
        round(busca.best_score_, 4),
    )

    resultados = pd.DataFrame(busca.cv_results_)

    colunas = [
        "params",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]

    resultados[colunas].sort_values(
        "rank_test_score"
    ).to_csv(
        RESULTADOS / "08_comparacao_hiperparametros.csv",
        index=False,
    )

    return busca.best_estimator_


def avaliar_modelo(modelo, X, y, nome):
    probabilidades = modelo.predict_proba(X)[:, 1]

    # Limiar inicial, sem ajuste na validação
    previsoes = (probabilidades >= 0.5).astype(int)

    metricas = {
        "Conjunto": nome,
        "Accuracy": accuracy_score(y, previsoes),
        "Precision": precision_score(
            y, previsoes, zero_division=0
        ),
        "Recall": recall_score(
            y, previsoes, zero_division=0
        ),
        "F1": f1_score(
            y, previsoes, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y, probabilidades
        ),
        "ROC-AUC": roc_auc_score(
            y, probabilidades
        ),
    }

    print(f"\n=== RESULTADOS: {nome.upper()} ===")

    print(
        classification_report(
            y,
            previsoes,
            labels=[0, 1],
            zero_division=0,
        )
    )

    return metricas, previsoes


def gerar_graficos(modelo, y_validacao, previsoes):
    print("\n=== GERANDO GRÁFICOS ===")

    # Matriz de confusão
    fig, ax = plt.subplots(figsize=(7, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_validacao,
        previsoes,
        labels=[0, 1],
        display_labels=[
            "Sem registro",
            "Com registro",
        ],
        ax=ax,
        cmap="Blues",
    )

    ax.set_title("Random Forest — Teste 2025")

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "08_matriz_confusao.png",
        dpi=160,
    )
    plt.close(fig)

    # Importância das variáveis
    importancias = pd.Series(
        modelo.named_steps["modelo"].feature_importances_,
        index=FEATURES,
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))

    importancias.plot.barh(ax=ax)

    ax.set_title("Importância das variáveis — Random Forest")
    ax.set_xlabel("Importância relativa")

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "08_importancia_features.png",
        dpi=160,
    )
    plt.close(fig)

    print("\nImportância das variáveis:")
    print(importancias.sort_values(ascending=False))


def main():
    print("=== ALERTASP — RANDOM FOREST ===")

    X_treino, y_treino, X_val, y_val = carregar_dados()

    print("\nTreinamento:", len(X_treino), "amostras")
    print("Teste 2025:", len(X_val), "amostras")
    print("Features:", len(FEATURES))

    modelo = treinar_modelo(X_treino, y_treino)

    metricas_treino, _ = avaliar_modelo(
        modelo, X_treino, y_treino, "Treino"
    )

    metricas_val, previsoes_val = avaliar_modelo(
        modelo, X_val, y_val, "Teste 2025"
    )

    comparacao = pd.DataFrame([
        metricas_treino,
        metricas_val,
    ])

    print("\n=== COMPARAÇÃO TREINO X TESTE ===")
    print(comparacao.round(4).to_string(index=False))

    comparacao.to_csv(
        RESULTADOS / "08_metricas_modelo.csv",
        index=False,
    )

    print("\n=== BASELINE ===")
    print(
        "PR-AUC da referência sem discriminação:",
        round(y_val.mean(), 4),
    )

    gerar_graficos(
        modelo,
        y_val,
        previsoes_val,
    )

    caminho_modelo = (
        MODELOS / "random_forest_alagamentos.joblib"
    )

    joblib.dump(modelo, caminho_modelo)

    print("\nModelo salvo em:", caminho_modelo)
    print("Resultados salvos em:", RESULTADOS)

    print("\nTreinamento concluído!")


if __name__ == "__main__":
    main()
