
from pathlib import Path
from zipfile import ZipFile
import unicodedata
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ARQUIVO_ZIP = BASE / "dados/brutos/inmet_2025.zip"
PASTA_SAIDA = BASE / "dados/processados"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ESTACOES = {
    "A701": "São Paulo - Mirante",
    "A771": "São Paulo - Interlagos",
}


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(
        c for c in texto if not unicodedata.combining(c)
    ).upper()


def localizar_coluna(df, prefixo):
    for coluna in df.columns:
        if normalizar(coluna).startswith(prefixo):
            return coluna

    raise ValueError(f"Coluna não encontrada: {prefixo}")


def converter_numero(serie):
    valores = pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    return valores.mask(valores == -9999)


def preparar_estacao(zip_file, arquivo, codigo):
    print(f"\nProcessando estação {codigo}...")

    with zip_file.open(arquivo) as f:
        df = pd.read_csv(
            f,
            sep=";",
            encoding="latin-1",
            skiprows=8,
            low_memory=False
        )

    df.columns = df.columns.str.strip()

    coluna_chuva = localizar_coluna(df, "PRECIPITACAO TOTAL")
    coluna_temp = localizar_coluna(df, "TEMPERATURA DO AR - BULBO SECO")
    coluna_umidade = localizar_coluna(df, "UMIDADE RELATIVA DO AR")

    # Horários originais do INMET estão em UTC
    horas = df["Hora UTC"].astype(str).str.extract(
        r"(\d{4})", expand=False
    )

    datas_utc = pd.to_datetime(
        df["Data"].astype(str).str.strip() + " " + horas,
        format="%Y/%m/%d %H%M",
        errors="coerce",
        utc=True
    )

    resultado = pd.DataFrame({
        "data_hora_utc": datas_utc,
        "estacao_codigo": codigo,
        "estacao_nome": ESTACOES[codigo],
        "precipitacao_mm": converter_numero(df[coluna_chuva]),
        "temperatura_c": converter_numero(df[coluna_temp]),
        "umidade_pct": converter_numero(df[coluna_umidade])
    })

    resultado = resultado.dropna(subset=["data_hora_utc"])

    print(f"Registros: {len(resultado)}")
    print("Dados ausentes:")
    print(resultado.isna().sum())

    return resultado


def main():
    datasets = []

    with ZipFile(ARQUIVO_ZIP) as zip_file:
        arquivos = zip_file.namelist()

        for codigo in ESTACOES:
            encontrados = [
                nome for nome in arquivos
                if f"_{codigo}_" in nome.upper()
                and nome.upper().endswith(".CSV")
            ]

            if not encontrados:
                print(f"Estação {codigo} não encontrada!")
                continue

            df = preparar_estacao(
                zip_file, encontrados[0], codigo
            )

            datasets.append(df)

    if not datasets:
        raise RuntimeError("Nenhuma estação foi processada.")

    df_final = pd.concat(datasets, ignore_index=True)
    df_final = df_final.sort_values(
        ["estacao_codigo", "data_hora_utc"]
    )

    destino = PASTA_SAIDA / "inmet_sp_2025.csv"
    df_final.to_csv(destino, index=False)

    print("\n=== RESULTADO FINAL ===")
    print(f"Total de registros: {len(df_final)}")
    print(f"Estações: {df_final['estacao_codigo'].nunique()}")
    print("\nPrimeiros registros:")
    print(df_final.head())
    print(f"\nArquivo salvo em: {destino}")


if __name__ == "__main__":
    main()
