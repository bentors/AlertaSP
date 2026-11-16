
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DADOS = BASE / "dados" / "processados"

ARQUIVO_INMET = DADOS / "inmet_sp_2025.csv"
ARQUIVO_ALAGAMENTOS = DADOS / "alagamentos_sp_2025.csv"
ARQUIVO_FINAL = DADOS / "base_diaria_2025.csv"

ESTACOES = ["A701", "A771"]


def preparar_inmet():
    print("=== PREPARANDO INMET ===")

    df = pd.read_csv(ARQUIVO_INMET)

    horarios = pd.to_datetime(
        df["data_hora_utc"],
        utc=True,
        errors="coerce"
    )

    if horarios.isna().any():
        raise ValueError("Existem horários inválidos no INMET.")

    df["data"] = pd.to_datetime(
        horarios.dt.tz_convert(
            "America/Sao_Paulo"
        ).dt.strftime("%Y-%m-%d")
    )

    diario = (
        df.groupby(["data", "estacao_codigo"])
        .agg(
            horas_csv=("precipitacao_mm", "size"),
            horas_validas=("precipitacao_mm", "count"),
            chuva_mm=(
                "precipitacao_mm",
                lambda x: x.sum(min_count=24)
            ),
            temperatura_media_c=(
                "temperatura_c", "mean"
            ),
            umidade_media_pct=(
                "umidade_pct", "mean"
            )
        )
        .reset_index()
    )

    base = pd.DataFrame({
        "data": pd.date_range(
            "2025-01-01", "2025-12-31"
        )
    })

    for codigo in ESTACOES:
        estacao = diario[
            diario["estacao_codigo"] == codigo
        ].copy()

        estacao = estacao.drop(
            columns=["estacao_codigo"]
        )

        estacao = estacao.rename(
            columns={
                coluna: f"{coluna}_{codigo.lower()}"
                for coluna in estacao.columns
                if coluna != "data"
            }
        )

        base = base.merge(
            estacao,
            on="data",
            how="left",
            validate="one_to_one"
        )

    return base


def preparar_alagamentos():
    print("\n=== PREPARANDO ALAGAMENTOS ===")

    df = pd.read_csv(ARQUIVO_ALAGAMENTOS)

    df["data"] = pd.to_datetime(
        df["data_ocorrencia"],
        errors="coerce"
    ).dt.normalize()

    if df["data"].isna().any():
        raise ValueError(
            "Existem datas inválidas nos alagamentos de 2025."
        )


    diario = (
        df.groupby("data")
        .size()
        .rename("registros_dia")
        .reset_index()
    )

    return diario


def main():
    inmet = preparar_inmet()
    alagamentos = preparar_alagamentos()

    print("\n=== CRUZANDO BASES ===")

    base = inmet.merge(
        alagamentos,
        on="data",
        how="left",
        validate="one_to_one"
    )

    # Zero significa nenhum registro na fonte,
    # não ausência comprovada de alagamento.
    base["registros_dia"] = (
        base["registros_dia"]
        .fillna(0)
        .astype(int)
    )

    base["houve_registro_dia"] = (
        base["registros_dia"] > 0
    ).astype(int)

    # Alvo experimental: ocorrência no dia seguinte.
    # Não utiliza a chuva do próprio dia previsto.
    base["alvo_registro_amanha"] = (
        base["houve_registro_dia"]
        .shift(-1)
        .astype("Int64")
    )

    base.to_csv(ARQUIVO_FINAL, index=False)

    print("Total de dias:", len(base))

    print(
        "Dias com registros:",
        base["houve_registro_dia"].sum()
    )

    print(
        "Total de ocorrências:",
        base["registros_dia"].sum()
    )

    print("\nDias sem total completo de chuva:")
    for codigo in ESTACOES:
        coluna = f"chuva_mm_{codigo.lower()}"
        print(
            f" {codigo}:",
            base[coluna].isna().sum()
        )

    print("\nPrimeiros registros:")
    print(base.head(8).to_string(index=False))

    print("\nArquivo salvo em:")
    print(ARQUIVO_FINAL)


if __name__ == "__main__":
    main()
