
from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ENTRADA = BASE / "dados" / "processados" / "base_diaria_2022_2026.csv"
SAIDA = BASE / "dados" / "modelagem"

ESTACOES = ("a701", "a771")
ALVO = "alvo_registro_amanha"


def carregar_base():
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"Base do script 06 não encontrada: {ENTRADA}"
        )

    df = pd.read_csv(ENTRADA, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)

    colunas_necessarias = {
        "data",
        "houve_registro_dia",
        ALVO,
        *(f"chuva_mm_{e}" for e in ESTACOES),
        *(f"temperatura_media_c_{e}" for e in ESTACOES),
        *(f"umidade_media_pct_{e}" for e in ESTACOES),
    }

    faltando = colunas_necessarias - set(df.columns)

    if faltando:
        raise ValueError(
            f"Colunas ausentes: {sorted(faltando)}"
        )

    if df["data"].isna().any() or df["data"].duplicated().any():
        raise ValueError("Datas inválidas ou duplicadas.")

    intervalo = df["data"].diff().dropna()

    if not intervalo.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Existem dias faltantes na base.")

    alvo_calculado = df["houve_registro_dia"].shift(-1)

    divergente = (
        pd.to_numeric(df[ALVO], errors="coerce")
        .fillna(-1)
        .ne(alvo_calculado.fillna(-1))
    )

    if divergente.any():
        raise ValueError(
            "O alvo não corresponde ao dia seguinte."
        )

    return df


def criar_features(df):
    df = df.copy()
    features = []

    for estacao in ESTACOES:
        chuva = f"chuva_mm_{estacao}"
        temperatura = f"temperatura_media_c_{estacao}"
        umidade = f"umidade_media_pct_{estacao}"

        features.extend([chuva, temperatura, umidade])

        # Chuva acumulada dos últimos 3 e 7 dias
        for janela in (3, 7):
            nome = f"chuva_acumulada_{janela}d_{estacao}"

            df[nome] = df[chuva].rolling(
                window=janela,
                min_periods=janela
            ).sum()

            features.append(nome)

    # Sazonalidade
    dia_ano = df["data"].dt.dayofyear

    df["dia_ano_seno"] = np.sin(
        2 * np.pi * dia_ano / 365.25
    )

    df["dia_ano_cosseno"] = np.cos(
        2 * np.pi * dia_ano / 365.25
    )

    features.extend([
        "dia_ano_seno",
        "dia_ano_cosseno"
    ])

    return df, features


def preparar_amostras(df, features):
    proxima_data = df["data"].shift(-1)

    proximo_dia_existe = proxima_data.eq(
        df["data"] + pd.Timedelta(days=1)
    )

    mesmo_ano = df["data"].dt.year.eq(
        proxima_data.dt.year
    )

    alvo_presente = df[ALVO].notna()

    amostras = df.loc[
        proximo_dia_existe & mesmo_ano & alvo_presente,
        ["data", *features, ALVO]
    ].copy()

    amostras[ALVO] = amostras[ALVO].astype(int)

    return amostras


def salvar_e_resumir(nome, df, features):
    caminho = SAIDA / nome
    df.to_csv(caminho, index=False)

    positivos = int(df[ALVO].sum()) if not df.empty else 0

    proporcao = (
        100 * positivos / len(df)
        if len(df) else 0
    )

    faltantes = int(
        df[features].isna().sum().sum()
    )

    print(f"\n{nome}")
    print(f"  Amostras: {len(df)}")
    print(
        f"  Com registro amanhã: "
        f"{positivos} ({proporcao:.1f}%)"
    )
    print(f"  Valores ausentes: {faltantes}")

    if not df.empty:
        print(
            f"  Período: "
            f"{df['data'].min().date()} até "
            f"{df['data'].max().date()}"
        )

    print(f"  Arquivo: {caminho}")


def main():
    print("=== PREPARAÇÃO DAS FEATURES ===")

    base = carregar_base()
    base, features = criar_features(base)

    amostras = preparar_amostras(base, features)

    SAIDA.mkdir(parents=True, exist_ok=True)

    treino = amostras[
        amostras["data"].dt.year.between(2022, 2024)
    ].copy()

    validacao = amostras[
        amostras["data"].dt.year == 2025
    ].copy()

    teste_provisorio = amostras[
        amostras["data"].dt.year == 2026
    ].copy()

    print(f"Total de dias: {len(base)}")
    print(f"Amostras preparadas: {len(amostras)}")
    print(f"Quantidade de features: {len(features)}")

    salvar_e_resumir(
        "treino_2022_2024.csv", treino, features
    )

    salvar_e_resumir(
        "validacao_2025.csv", validacao, features
    )

    salvar_e_resumir(
        "teste_2026_PROVISORIO.csv",
        teste_provisorio,
        features
    )

    print("\n=== VALORES AUSENTES NO TREINO ===")
    print(treino[features].isna().sum().to_string())

    salvar_e_resumir(
        "dataset_features_2022_2026.csv",
        amostras,
        features
    )

    print("\nOBSERVAÇÕES:")
    print("- Valores ausentes preservados como NaN.")
    print("- O tratamento será realizado no treinamento.")
    print("- Dados de 2026 ainda são provisórios.")
    print("- Nenhum dado do dia seguinte foi usado como feature.")

    print("\nPreparação concluída!")


if __name__ == "__main__":
    main()
