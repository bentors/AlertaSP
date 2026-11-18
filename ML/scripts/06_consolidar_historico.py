
from pathlib import Path
from zipfile import ZipFile, is_zipfile
import unicodedata

import pandas as pd
import requests


BASE = Path(__file__).resolve().parents[1]
BRUTOS = BASE / "dados" / "brutos"
PROCESSADOS = BASE / "dados" / "processados"

BRUTOS.mkdir(parents=True, exist_ok=True)
PROCESSADOS.mkdir(parents=True, exist_ok=True)

ANOS_BASE = range(2022, 2027)
ANOS_DOWNLOAD = range(2021, 2027)
ESTACOES = ["A701", "A771"]

SAIDA = PROCESSADOS / "base_diaria_2022_2025.csv"


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(
        c for c in texto if not unicodedata.combining(c)
    ).upper()


def encontrar_coluna(df, prefixo):
    for coluna in df.columns:
        if normalizar(coluna).startswith(prefixo):
            return coluna
    raise ValueError(f"Coluna não encontrada: {prefixo}")


def converter_numero(serie):
    valores = pd.to_numeric(
        serie.astype("string").str.replace(
            ",", ".", regex=False
        ),
        errors="coerce"
    )
    return valores.mask(valores.abs() >= 9999)


def baixar_inmet(ano):
    destino = BRUTOS / f"inmet_{ano}.zip"

    if destino.exists() and is_zipfile(destino):
        print(f"{ano}: arquivo existente")
        return destino

    url = (
        f"https://portal.inmet.gov.br/"
        f"uploads/dadoshistoricos/{ano}.zip"
    )

    print(f"Baixando INMET {ano}...")
    temporario = destino.with_suffix(".part")

    try:
        with requests.get(
            url, stream=True, timeout=(20, 120)
        ) as resposta:
            resposta.raise_for_status()

            with open(temporario, "wb") as arquivo:
                for bloco in resposta.iter_content(1024 * 1024):
                    if bloco:
                        arquivo.write(bloco)

        if not is_zipfile(temporario):
            raise ValueError("Download não é um ZIP válido.")

        temporario.replace(destino)
        return destino

    except (requests.RequestException, ValueError) as erro:
        temporario.unlink(missing_ok=True)

        # Arquivos das bordas são complementares.
        if ano in (2021, 2026):
            print(f"Aviso: {ano} indisponível: {erro}")
            return None

        raise


def carregar_estacao(zip_file, codigo):
    encontrados = [
        nome for nome in zip_file.namelist()
        if f"_{codigo}_" in nome.upper()
        and nome.upper().endswith(".CSV")
    ]

    if not encontrados:
        return None

    with zip_file.open(encontrados[0]) as arquivo:
        df = pd.read_csv(
            arquivo,
            sep=";",
            encoding="latin-1",
            skiprows=8,
            low_memory=False
        )

    df.columns = df.columns.str.strip()

    chuva = encontrar_coluna(df, "PRECIPITACAO TOTAL")
    temp = encontrar_coluna(
        df, "TEMPERATURA DO AR - BULBO SECO"
    )
    umidade = encontrar_coluna(
        df, "UMIDADE RELATIVA DO AR"
    )

    horas = df["Hora UTC"].astype("string").str.extract(
        r"(\d{4})", expand=False
    )

    datas = (
        df["Data"].astype("string")
        .str.strip()
        .str.replace("-", "/", regex=False)
    )

    timestamp = pd.to_datetime(
        datas + " " + horas,
        format="%Y/%m/%d %H%M",
        utc=True,
        errors="coerce"
    )

    resultado = pd.DataFrame({
        "data_hora_utc": timestamp,
        "estacao_codigo": codigo,
        "chuva_mm": converter_numero(df[chuva]),
        "temperatura_c": converter_numero(df[temp]),
        "umidade_pct": converter_numero(df[umidade])
    })

    resultado["chuva_mm"] = resultado[
        "chuva_mm"
    ].mask(resultado["chuva_mm"] < 0)

    resultado["umidade_pct"] = resultado[
        "umidade_pct"
    ].where(
        resultado["umidade_pct"].between(0, 100)
    )

    return resultado.dropna(subset=["data_hora_utc"])


def preparar_inmet():
    datasets = []

    for ano in ANOS_DOWNLOAD:
        caminho = baixar_inmet(ano)

        if caminho is None:
            continue

        with ZipFile(caminho) as zip_file:
            for codigo in ESTACOES:
                df = carregar_estacao(zip_file, codigo)

                if df is None:
                    if ano in ANOS_BASE:
                        raise RuntimeError(
                            f"Estação {codigo} ausente em {ano}"
                        )
                    continue

                datasets.append(df)

        print(f"{ano}: processamento concluído")

    df = pd.concat(datasets, ignore_index=True)

    df = df.drop_duplicates(
        subset=["estacao_codigo", "data_hora_utc"]
    )

    # Primeiro converter UTC para o horário da cidade.
    df["data"] = (
        df["data_hora_utc"]
        .dt.tz_convert("America/Sao_Paulo")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Manter somente o período do experimento.
    df = df[
        df["data"].dt.year.isin(ANOS_BASE)
    ].copy()

    calendario = pd.DataFrame({
        "data": pd.date_range(
            "2022-01-01", "2025-12-31"
        )
    })

    for codigo in ESTACOES:
        estacao = df[
            df["estacao_codigo"] == codigo
        ]

        diario = estacao.groupby("data").agg(
            horas_validas=("chuva_mm", "count"),
            chuva_mm=(
                "chuva_mm",
                lambda x: x.sum(min_count=24)
            ),
            temperatura_media_c=(
                "temperatura_c", "mean"
            ),
            umidade_media_pct=(
                "umidade_pct", "mean"
            )
        ).reset_index()

        diario = diario.rename(columns={
            coluna: f"{coluna}_{codigo.lower()}"
            for coluna in diario.columns
            if coluna != "data"
        })

        calendario = calendario.merge(
            diario,
            on="data",
            how="left",
            validate="one_to_one"
        )

    return calendario


def preparar_alagamentos():
    caminho = BRUTOS / "alagamentos_consolidado.csv"

    df = pd.read_csv(caminho, low_memory=False)

    datas = (
        df["data"].astype("string")
        .str.strip()
        .str.replace("-", "/", regex=False)
    )

    df["data"] = pd.to_datetime(
        datas,
        format="%Y/%m/%d",
        errors="coerce"
    )

    if df["data"].isna().any():
        raise ValueError("Há datas inválidas nos registros.")

    df["tipo"] = df["ocorrencia"].map(normalizar)

    df = df[
        df["data"].dt.year.isin(ANOS_BASE)
        & df["tipo"].isin(["ALAGAMENTO", "INUNDACAO"])
    ]

    diario = (
        df.groupby("data")
        .size()
        .rename("registros_dia")
        .reset_index()
    )

    return diario


def main():
    print("=== CONSOLIDANDO INMET ===")
    inmet = preparar_inmet()

    print("\n=== CONSOLIDANDO ALAGAMENTOS ===")
    alagamentos = preparar_alagamentos()

    base = inmet.merge(
        alagamentos,
        on="data",
        how="left",
        validate="one_to_one"
    )

    base["registros_dia"] = (
        base["registros_dia"]
        .fillna(0)
        .astype(int)
    )

    base["houve_registro_dia"] = (
        base["registros_dia"] > 0
    ).astype(int)

    # Prever registro no dia seguinte usando
    # informações disponíveis até o fim do dia atual.
    base["alvo_registro_amanha"] = (
        base["houve_registro_dia"]
        .shift(-1)
        .astype("Int64")
    )

    base.to_csv(SAIDA, index=False)

    print("\n=== RESULTADO POR ANO ===")

    base["ano"] = base["data"].dt.year

    resumo = base.groupby("ano").agg(
        dias=("data", "count"),
        dias_com_registro=("houve_registro_dia", "sum"),
        ocorrencias=("registros_dia", "sum"),
        chuva_ausente_a701=("chuva_mm_a701",
                             lambda x: x.isna().sum()),
        chuva_ausente_a771=("chuva_mm_a771",
                             lambda x: x.isna().sum())
    )

    print(resumo.to_string())

    print("\n=== AUDITORIA ===")
    print("Total de dias:", len(base))
    print("Datas duplicadas:", base["data"].duplicated().sum())
    print("Alvos ausentes:",
          base["alvo_registro_amanha"].isna().sum())

    print("\nArquivo salvo:", SAIDA)
    print("Consolidação concluída!")


if __name__ == "__main__":
    main()
