
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DADOS = BASE / "dados" / "processados"
RESULTADOS = BASE / "resultados"

RESULTADOS.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(
        DADOS / "base_diaria_2025.csv",
        parse_dates=["data"]
    ).sort_values("data").reset_index(drop=True)

    ocorrencias = pd.read_csv(
        DADOS / "alagamentos_sp_2025.csv"
    )

    print("=== AUDITORIA DOS DADOS ===")
    print("Total de dias:", len(df))
    print("Período:", df["data"].min(), "até", df["data"].max())
    print("Datas duplicadas:", df["data"].duplicated().sum())

    print("\nValores ausentes por coluna:")
    print(df.isna().sum().to_string())

    print("\n=== DISTRIBUIÇÃO DA VARIÁVEL-ALVO ===")
    print(
        df["houve_registro_dia"]
        .value_counts()
        .sort_index()
    )

    print("\n=== VERIFICAÇÃO DO ALVO FUTURO ===")
    esperado = df["houve_registro_dia"].shift(-1)

    divergencias = (
        df["alvo_registro_amanha"]
        .fillna(-1)
        .ne(esperado.fillna(-1))
        .sum()
    )

    print("Divergências no target:", divergencias)
    print(
        "Linhas sem target futuro:",
        df["alvo_registro_amanha"].isna().sum()
    )

    print("\n=== OCORRÊNCIAS POR MÊS ===")
    df["mes"] = df["data"].dt.month

    mensal = df.groupby("mes").agg(
        dias_com_registro=("houve_registro_dia", "sum"),
        total_ocorrencias=("registros_dia", "sum")
    )

    print(mensal.to_string())

    print("\n=== REGISTROS REPETIDOS ===")
    repetidos = ocorrencias.duplicated(
        subset=[
            "data_ocorrencia",
            "tipo_ocorrencia",
            "x_utm",
            "y_utm"
        ]
    ).sum()

    print("Mesmo dia, tipo e coordenadas:", repetidos)

    print("\n=== PRECIPITAÇÃO ===")

    colunas_chuva = [
        "chuva_mm_a701",
        "chuva_mm_a771"
    ]

    # Maior total entre as duas estações disponíveis.
    # Não representa a chuva em toda a cidade.
    df["chuva_max_estacoes_mm"] = df[
        colunas_chuva
    ].max(axis=1, skipna=True)

    for coluna in colunas_chuva:
        print(f"\n{coluna}")
        print(df[coluna].describe().to_string())

    # Gráfico 1: quantidade de dias com registros por mês.
    fig, ax = plt.subplots(figsize=(10, 5))

    mensal["dias_com_registro"].plot.bar(ax=ax)

    ax.set_title("Dias com registros de alagamento — 2025")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Dias com registros")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "01_ocorrencias_por_mes.png",
        dpi=160
    )
    plt.close(fig)

    # Gráfico 2: chuva e quantidade de ocorrências
    # registradas no mesmo dia.
    dados_grafico = df.dropna(
        subset=["chuva_max_estacoes_mm"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        dados_grafico["chuva_max_estacoes_mm"],
        dados_grafico["registros_dia"],
        alpha=0.6
    )

    ax.set_title("Chuva observada x ocorrências registradas")
    ax.set_xlabel("Maior chuva diária entre A701/A771 (mm)")
    ax.set_ylabel("Ocorrências no mesmo dia")

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "02_chuva_vs_ocorrencias.png",
        dpi=160
    )
    plt.close(fig)

    # Gráfico 3: chuva de hoje x registro amanhã.
    amostras = df.dropna(
        subset=[
            "chuva_max_estacoes_mm",
            "alvo_registro_amanha"
        ]
    )

    sem_registro = amostras.loc[
        amostras["alvo_registro_amanha"] == 0,
        "chuva_max_estacoes_mm"
    ]

    com_registro = amostras.loc[
        amostras["alvo_registro_amanha"] == 1,
        "chuva_max_estacoes_mm"
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(
        [sem_registro, com_registro],
        tick_labels=[
            "Sem registro amanhã",
            "Com registro amanhã"
        ]
    )

    ax.set_title("Chuva de hoje e registros no dia seguinte")
    ax.set_ylabel("Maior chuva entre A701/A771 (mm)")

    fig.tight_layout()
    fig.savefig(
        RESULTADOS / "03_chuva_vs_alvo_futuro.png",
        dpi=160
    )
    plt.close(fig)

    print("\n=== CONFERÊNCIA FINAL ===")
    print(
        "Ocorrências no CSV:",
        len(ocorrencias)
    )
    print(
        "Ocorrências na base diária:",
        df["registros_dia"].sum()
    )

    print("\nGráficos salvos em:", RESULTADOS)
    print("\nAnálise exploratória concluída!")


if __name__ == "__main__":
    main()
    