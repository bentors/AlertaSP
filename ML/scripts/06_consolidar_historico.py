
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

# 2021 é usado para completar a virada para 2022
ANOS_DOWNLOAD = range(2021, 2027)
ANOS_BASE = range(2022, 2027)

ESTACOES = ("A701", "A771")

SAIDA = PROCESSADOS / "base_diaria_2022_2026.csv"


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    ).upper().strip()


def encontrar_coluna(df, prefixo):
    for coluna in df.columns:
        if normalizar(coluna).startswith(prefixo):
            return coluna

    raise ValueError(f"Coluna não encontrada: {prefixo}")


def converter_numero(serie):
    numeros = pd.to_numeric(
        serie.astype("string").str.replace(
            ",", ".", regex=False
        ),
        errors="coerce",
    )

    return numeros.mask(numeros.abs() >= 9999)


def baixar_inmet(ano):
    destino = BRUTOS / f"inmet_{ano}.zip"

    if destino.exists() and is_zipfile(destino):
        print(f"{ano}: arquivo existente")
        return destino

    url = (
        f"https://portal.inmet.gov.br/"
        f"uploads/dadoshistoricos/{ano}.zip"
    )

    temporario = destino.with_suffix(".part")

    print(f"Baixando INMET {ano}...")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 120),
        ) as resposta:

            resposta.raise_for_status()

            with temporario.open("wb") as arquivo:
                for bloco in resposta.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if bloco:
                        arquivo.write(bloco)

        if not is_zipfile(temporario):
            raise ValueError(
                f"Arquivo ZIP inválido: {ano}"
            )

        temporario.replace(destino)
        return destino

    except (requests.RequestException, ValueError) as erro:
        temporario.unlink(missing_ok=True)

        if ano == 2021:
            print(
                f"Aviso: arquivo complementar "
                f"de 2021 indisponível: {erro}"
            )
            return None

        raise


def carregar_estacao(zip_file, codigo):
    encontrados = [
        nome
        for nome in zip_file.namelist()
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
            low_memory=False,
        )

    df.columns = df.columns.str.strip()

    coluna_chuva = encontrar_coluna(
        df, "PRECIPITACAO TOTAL"
    )

    coluna_temp = encontrar_coluna(
        df, "TEMPERATURA DO AR - BULBO SECO"
    )

    coluna_umidade = encontrar_coluna(
        df, "UMIDADE RELATIVA DO AR"
    )

    horas = (
        df["Hora UTC"]
        .astype("string")
        .str.extract(r"(\d{4})", expand=False)
    )

    datas = (
        df["Data"]
        .astype("string")
        .str.strip()
        .str.replace("-", "/", regex=False)
    )

    horarios_utc = pd.to_datetime(
        datas + " " + horas,
        format="%Y/%m/%d %H%M",
        errors="coerce",
        utc=True,
    )

    resultado = pd.DataFrame({
        "data_hora_utc": horarios_utc,
        "estacao_codigo": codigo,
        "chuva_mm": converter_numero(df[coluna_chuva]),
        "temperatura_c": converter_numero(df[coluna_temp]),
        "umidade_pct": converter_numero(df[coluna_umidade]),
    })

    resultado["chuva_mm"] = (
        resultado["chuva_mm"]
        .where(resultado["chuva_mm"] >= 0)
    )

    resultado["umidade_pct"] = (
        resultado["umidade_pct"]
        .where(resultado["umidade_pct"].between(0, 100))
    )

    return resultado.dropna(
        subset=["data_hora_utc"]
    )


def preparar_inmet():
    conjuntos = []

    for ano in ANOS_DOWNLOAD:
        caminho = baixar_inmet(ano)

        if caminho is None:
            continue

        with ZipFile(caminho) as zip_file:
            for codigo in ESTACOES:

                df = carregar_estacao(
                    zip_file, codigo
                )

                if df is None:
                    if ano in ANOS_BASE:
                        raise RuntimeError(
                            f"Estação {codigo} não "
                            f"encontrada no INMET {ano}."
                        )
                    continue

                conjuntos.append(df)

        print(f"{ano}: processamento concluído")

    if not conjuntos:
        raise RuntimeError(
            "Não foi possível carregar dados do INMET."
        )

    df = pd.concat(
        conjuntos,
        ignore_index=True
    )

    df = (
        df.sort_values("data_hora_utc")
        .drop_duplicates(
            subset=[
                "estacao_codigo",
                "data_hora_utc"
            ],
            keep="last",
        )
    )

    # Converter UTC para horário de São Paulo
    df["data"] = (
        df["data_hora_utc"]
        .dt.tz_convert("America/Sao_Paulo")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df = df[
        df["data"].dt.year.isin(ANOS_BASE)
    ].copy()

    calendario = pd.DataFrame({
        "data": pd.date_range(
            "2022-01-01",
            "2026-12-31",
            freq="D"
        )
    })

    for codigo in ESTACOES:
        estacao = df[
            df["estacao_codigo"] == codigo
        ]

        diario = (
            estacao.groupby("data")
            .agg(
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
                ),
            )
            .reset_index()
        )

        diario = diario.rename(
            columns={
                coluna: f"{coluna}_{codigo.lower()}"
                for coluna in diario.columns
                if coluna != "data"
            }
        )

        calendario = calendario.merge(
            diario,
            on="data",
            how="left",
            validate="one_to_one"
        )

    return calendario


def preparar_alagamentos():
    caminho = (
        BRUTOS / "alagamentos_consolidado.csv"
    )

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    df = pd.read_csv(
        caminho,
        low_memory=False
    )

    datas = (
        df["data"]
        .astype("string")
        .str.strip()
        .str.replace("-", "/", regex=False)
    )

    df["data"] = pd.to_datetime(
        datas,
        format="%Y/%m/%d",
        errors="coerce"
    )

    if df["data"].isna().any():
        raise ValueError(
            "Há datas inválidas no CSV de alagamentos."
        )

    df["tipo"] = df["ocorrencia"].map(normalizar)

    df = df[
        df["data"].dt.year.isin(ANOS_BASE)
        & df["tipo"].isin([
            "ALAGAMENTO",
            "INUNDACAO"
        ])
    ].copy()

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

    print("\n=== VERIFICANDO COBERTURA DE 2026 ===")

    inmet_2026 = inmet[
        inmet["data"].dt.year == 2026
    ]

    pelo_menos_uma_estacao_completa = (
        (inmet_2026["horas_validas_a701"] >= 24)
        | (inmet_2026["horas_validas_a771"] >= 24)
    )

    ultima_data_inmet = inmet_2026.loc[
        pelo_menos_uma_estacao_completa,
        "data"
    ].max()

    ultima_ocorrencia_2026 = alagamentos.loc[
        alagamentos["data"].dt.year == 2026,
        "data"
    ].max()

    print(
        "Último dia com chuva completa:",
        ultima_data_inmet
    )

    print(
        "Último registro de alagamento:",
        ultima_ocorrencia_2026
    )

    if (
        pd.isna(ultima_data_inmet)
        or pd.isna(ultima_ocorrencia_2026)
    ):
        raise ValueError(
            "Não há cobertura suficiente de 2026 "
            "para estabelecer o corte."
        )

    # Limite provisório para não incluir dias futuros.
    # A última ocorrência não comprova a cobertura
    # completa da fonte de alagamentos.
    data_limite = min(
        ultima_data_inmet,
        ultima_ocorrencia_2026
    )

    print(
        "Data limite provisória:",
        data_limite
    )

    # Cruzar medições meteorológicas com ocorrências
    base = inmet.merge(
        alagamentos,
        on="data",
        how="left",
        validate="one_to_one"
    )

    base = base.loc[
        base["data"] <= data_limite
    ].copy()

    # Zero significa ausência de registro no CSV,
    # não prova de que a rua não alagou.
    base["registros_dia"] = (
        base["registros_dia"]
        .fillna(0)
        .astype(int)
    )

    base["houve_registro_dia"] = (
        base["registros_dia"] > 0
    ).astype(int)

    # Alvo: ocorrência registrada no dia seguinte
    base["alvo_registro_amanha"] = (
        base["houve_registro_dia"]
        .shift(-1)
        .astype("Int64")
    )

    # Salvar a base consolidada
    base.to_csv(
        SAIDA,
        index=False
    )

    print("\n=== RESULTADO POR ANO ===")

    base["ano"] = base["data"].dt.year

    resumo = base.groupby("ano").agg(
        dias=("data", "count"),
        dias_com_registro=(
            "houve_registro_dia", "sum"
        ),
        ocorrencias=("registros_dia", "sum"),
        chuva_ausente_a701=(
            "chuva_mm_a701",
            lambda x: x.isna().sum()
        ),
        chuva_ausente_a771=(
            "chuva_mm_a771",
            lambda x: x.isna().sum()
        ),
    )

    print(resumo.to_string())

    print("\n=== AUDITORIA ===")

    print(
        "Total de dias:",
        len(base)
    )

    print(
        "Datas duplicadas:",
        base["data"].duplicated().sum()
    )

    print(
        "Alvos ausentes:",
        base["alvo_registro_amanha"].isna().sum()
    )

    print(
        "\nAVISO: 2026 é provisório até confirmar "
        "a cobertura da fonte de ocorrências."
    )

    print("\nArquivo salvo:", SAIDA)
    print("Consolidação concluída!")


if __name__ == "__main__":
    main()
