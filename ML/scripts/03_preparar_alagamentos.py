
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import unicodedata

import pandas as pd
import requests


BASE = Path(__file__).resolve().parents[1]
BRUTOS = BASE / "dados" / "brutos"
PROCESSADOS = BASE / "dados" / "processados"

BRUTOS.mkdir(parents=True, exist_ok=True)
PROCESSADOS.mkdir(parents=True, exist_ok=True)

URL = (
    "https://raw.githubusercontent.com/"
    "23216886alunounivesp/"
    "sp-alagamentos-inundacoes-defesa-civil/"
    "main/alagamentos_e_inundacoes_sp.csv"
)

ARQUIVO_BRUTO = BRUTOS / "alagamentos_consolidado.csv"
ARQUIVO_2025 = PROCESSADOS / "alagamentos_sp_2025.csv"
ARQUIVO_DIARIO = PROCESSADOS / "ocorrencias_diarias_2025.csv"


def normalizar(texto):
    if pd.isna(texto):
        return None

    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(
        letra for letra in texto
        if not unicodedata.combining(letra)
    )
    return texto.strip().upper()


def baixar_dados():
    if ARQUIVO_BRUTO.exists():
        print("Arquivo existente encontrado.")
        return

    print("Baixando registros consolidados...")

    resposta = requests.get(URL, timeout=60)
    resposta.raise_for_status()

    ARQUIVO_BRUTO.write_bytes(resposta.content)

    print("Download concluído!")


def preparar_dados():
    df = pd.read_csv(
        ARQUIVO_BRUTO,
        dtype={"name": "string"},
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    obrigatorias = {
        "X", "Y", "data",
        "ocorrencia", "subprefeit"
    }

    ausentes = obrigatorias - set(df.columns)

    if ausentes:
        raise ValueError(
            f"Colunas ausentes: {sorted(ausentes)}"
        )

    print("\n=== BASE ORIGINAL ===")
    print("Total de registros:", len(df))
    print("Colunas:", df.columns.tolist())

    # Datas do CSV no formato AAAA/MM/DD
    df["data_ocorrencia"] = pd.to_datetime(
    df["data"]
    .astype("string")
    .str.strip()
    .str.replace("-", "/", regex=False),
    format="%Y/%m/%d",
    errors="coerce"
    )

    df["tipo_ocorrencia"] = (
        df["ocorrencia"].map(normalizar)
    )

    df["subprefeitura"] = df["subprefeit"].map(normalizar)

    # Coordenadas em SIRGAS 2000 / UTM 23S
    df["x_utm"] = pd.to_numeric(
        df["X"], errors="coerce"
    )
    df["y_utm"] = pd.to_numeric(
        df["Y"], errors="coerce"
    )

    print("\nDatas inválidas:",
          df["data_ocorrencia"].isna().sum())

    print("Coordenadas ausentes:")
    print(df[["x_utm", "y_utm"]].isna().sum())

    print("\nTipos encontrados:")
    print(df["tipo_ocorrencia"].value_counts(dropna=False))

    print("\nRegistros por ano:")
    print(
        df["data_ocorrencia"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    # Selecionar somente 2025
    df_2025 = df[
        (df["data_ocorrencia"].dt.year == 2025)
        & df["tipo_ocorrencia"].isin(
            ["ALAGAMENTO", "INUNDACAO"]
        )
    ].copy()

    if df_2025.empty:
        raise ValueError(
            "Nenhuma ocorrência válida encontrada em 2025."
        )

    df_2025 = df_2025[
        [
            "data_ocorrencia",
            "tipo_ocorrencia",
            "subprefeitura",
            "x_utm",
            "y_utm",
            "name"
        ]
    ]

    df_2025 = df_2025.sort_values(
        "data_ocorrencia"
    )

    # Preservamos registros repetidos para auditoria.
    # Não eliminamos relatos diferentes do mesmo evento.
    duplicados = df_2025.duplicated().sum()

    df_2025.to_csv(
        ARQUIVO_2025,
        index=False
    )

    # Contagem apenas dos dias COM registros
    diario = (
        df_2025
        .groupby(
            ["data_ocorrencia", "tipo_ocorrencia"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    diario.to_csv(ARQUIVO_DIARIO, index=False)

    print("\n=== OCORRÊNCIAS DE 2025 ===")
    print("Total:", len(df_2025))
    print("Dias com registros:", len(diario))
    print("Linhas idênticas:", duplicados)

    print("\nPor tipo:")
    print(df_2025["tipo_ocorrencia"].value_counts())

    print("\nTop 10 subprefeituras:")
    print(
        df_2025["subprefeitura"]
        .value_counts()
        .head(10)
    )

    print("\nPrimeiros registros:")
    print(df_2025.head(5).to_string(index=False))

    # Registrar a origem do arquivo para rastreabilidade
    metadata = {
        "fonte_download": URL,
        "origem_declarada": "GeoSampa / Defesa Civil",
        "tipo_fonte": "CSV consolidado por terceiro",
        "baixado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "sha256": hashlib.sha256(
            ARQUIVO_BRUTO.read_bytes()
        ).hexdigest(),
        "periodo_analisado": 2025
    }

    destino_metadata = BRUTOS / "fonte_alagamentos.json"

    destino_metadata.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== ARQUIVOS GERADOS ===")
    print(ARQUIVO_2025)
    print(ARQUIVO_DIARIO)
    print(destino_metadata)


if __name__ == "__main__":
    baixar_dados()
    preparar_dados()
