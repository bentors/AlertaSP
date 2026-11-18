
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


BASE = Path(__file__).resolve().parents[1]
DADOS = BASE / "dados" / "modelagem"
RESULTADOS = BASE / "resultados"
MODELOS = BASE / "modelos"

RESULTADOS.mkdir(parents=True, exist_ok=True)

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

SAZONALIDADE = ["dia_ano_seno", "dia_ano_cosseno"]

METEOROLOGIA = [
    coluna for coluna in FEATURES
    if coluna not in SAZONALIDADE
]


def carregar_dados():
    treino = pd.read_csv(
        DADOS / "treino_2022_2024.csv",
        parse_dates=["data"]
    )

    validacao = pd.read_csv(
        DADOS / "validacao_2025.csv",
        parse_dates=["data"]
    )

    treino = treino.sort_values("data")
    validacao = validacao.sort_values("data")

    if treino["data"].max() >= validacao["data"].min():
        raise ValueError("Períodos sobrepostos.")

    for df in (treino, validacao):
        if df[ALVO].isna().any():
            raise ValueError("Alvo contém valores ausentes.")

    return treino, validacao


def calcular_metricas(nome, y, probabilidades):
    previsoes = (probabilidades >= 0.5).astype(int)

    return {
        "Modelo": nome,
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


def main():
    print("=== ALERTASP — ANÁLISE DO MODELO ===")

    treino, validacao = carregar_dados()

    X_treino = treino[FEATURES]
    y_treino = treino[ALVO].astype(int)

    X_val = validacao[FEATURES]
    y_val = validacao[ALVO].astype(int)

    # Carregar modelo completo treinado no script 08.
    caminho = MODELOS / "random_forest_alagamentos.joblib"
    modelo_completo = joblib.load(caminho)

    modelos = {
        "Completo": (modelo_completo, FEATURES),
        "Sazonalidade": (
            clone(modelo_completo),
            SAZONALIDADE,
        ),
        "Meteorologia": (
            clone(modelo_completo),
            METEOROLOGIA,
        ),
    }

    resultados = []
    probabilidades_modelos = {}

    for nome, (modelo, colunas) in modelos.items():
        print(f"\n=== MODELO: {nome.upper()} ===")

        # O modelo completo já está treinado.
        if nome != "Completo":
            modelo.fit(X_treino[colunas], y_treino)

        probabilidades = modelo.predict_proba(
            X_val[colunas]
        )[:, 1]

        probabilidades_modelos[nome] = probabilidades

        metricas = calcular_metricas(
            nome, y_val, probabilidades
        )

        resultados.append(metricas)

        print(
            pd.DataFrame([metricas])
            .round(4)
            .to_string(index=False)
        )

    # Referência: probabilidade constante,
    # igual à frequência positiva do treinamento.
    referencia = y_treino.mean()

    print("\n=== REFERÊNCIA SEM DISCRIMINAÇÃO ===")
    print(f"Frequência positiva treino: {referencia:.4f}")
    print(f"PR-AUC referência 2025: {y_val.mean():.4f}")

    comparacao = pd.DataFrame(resultados)

    comparacao.to_csv(
        RESULTADOS / "09_comparacao_modelos.csv",
        index=False
    )

    print("\n=== COMPARAÇÃO FINAL ===")
    print(
        comparacao.sort_values(
            "PR-AUC", ascending=False
        ).round(4).to_string(index=False)
    )

    # Gráfico 1: comparação da PR-AUC.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        comparacao["Modelo"],
        comparacao["PR-AUC"]
    )

    ax.axhline(
        y_val.mean(),
        linestyle="--",
        label="Referência sem discriminação"
    )

    ax.set_title("Comparação de modelos — 2025")
    ax.set_ylabel("Average Precision")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "09_comparacao_pr_auc.png",
        dpi=160
    )
    plt.close(fig)

    # Gráfico 2: curvas Precision-Recall.
    fig, ax = plt.subplots(figsize=(9, 6))

    for nome, probabilidades in probabilidades_modelos.items():
        precision, recall, _ = precision_recall_curve(
            y_val, probabilidades
        )

        ap = average_precision_score(
            y_val, probabilidades
        )

        ax.plot(
            recall,
            precision,
            label=f"{nome} (AP={ap:.3f})"
        )

    ax.axhline(
        y_val.mean(),
        linestyle="--",
        label="Referência"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — 2025")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "09_curvas_precision_recall.png",
        dpi=160
    )
    plt.close(fig)

    # Importância por permutação: mede quanto a
    # PR-AUC cai quando embaralhamos uma variável.
    print("\n=== IMPORTÂNCIA POR PERMUTAÇÃO ===")

    importancia = permutation_importance(
        modelo_completo,
        X_val[FEATURES],
        y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    tabela_importancia = pd.DataFrame({
        "feature": FEATURES,
        "importancia_media": importancia.importances_mean,
        "desvio_padrao": importancia.importances_std,
    }).sort_values(
        "importancia_media",
        ascending=False
    )

    print(tabela_importancia.to_string(index=False))

    tabela_importancia.to_csv(
        RESULTADOS / "09_importancia_permutacao.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    tabela_importancia.sort_values(
        "importancia_media"
    ).plot.barh(
        x="feature",
        y="importancia_media",
        legend=False,
        ax=ax
    )

    ax.axvline(0, linewidth=1)
    ax.set_title("Importância por permutação — 2025")
    ax.set_xlabel("Queda média da Average Precision")

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "09_importancia_permutacao.png",
        dpi=160
    )
    plt.close(fig)

    # Salvar previsões para auditoria.
    previsoes = pd.DataFrame({
        "data": validacao["data"].to_numpy(),
        "real": y_val.to_numpy(),
        "prob_completo": probabilidades_modelos["Completo"],
        "prob_sazonalidade": probabilidades_modelos[
            "Sazonalidade"
        ],
        "prob_meteorologia": probabilidades_modelos[
            "Meteorologia"
        ],
    })

    previsoes.to_csv(
        RESULTADOS / "09_previsoes_2025.csv",
        index=False
    )

    print("\n=== ARQUIVOS GERADOS ===")
    print("09_comparacao_modelos.csv")
    print("09_comparacao_pr_auc.png")
    print("09_curvas_precision_recall.png")
    print("09_importancia_permutacao.csv")
    print("09_importancia_permutacao.png")
    print("09_previsoes_2025.csv")

    print("\nAnálise concluída!")


if __name__ == "__main__":
    main()
