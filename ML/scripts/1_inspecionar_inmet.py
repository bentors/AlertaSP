
from pathlib import Path
from zipfile import ZipFile, is_zipfile
import unicodedata
import requests

# Dados históricos oficiais do INMET
ANO = 2025
URL = f"https://portal.inmet.gov.br/uploads/dadoshistoricos/{ANO}.zip"

PASTA = Path(__file__).resolve().parents[1] / "dados" / "brutos"
PASTA.mkdir(parents=True, exist_ok=True)
ARQUIVO = PASTA / f"inmet_{ANO}.zip"


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(
        c for c in texto if not unicodedata.combining(c)
    ).upper()


def baixar_dados():
    if ARQUIVO.exists() and is_zipfile(ARQUIVO):
        print("Arquivo já baixado:", ARQUIVO)
        return

    print(f"Baixando dados oficiais do INMET ({ANO})...")

    temporario = ARQUIVO.with_suffix(".part")

    with requests.get(URL, stream=True, timeout=(20, 120)) as resposta:
        resposta.raise_for_status()

        with open(temporario, "wb") as arquivo:
            for bloco in resposta.iter_content(chunk_size=1024 * 1024):
                if bloco:
                    arquivo.write(bloco)

    if not is_zipfile(temporario):
        temporario.unlink(missing_ok=True)
        raise ValueError("O download não retornou um ZIP válido.")

    temporario.replace(ARQUIVO)
    print("Download concluído!")


def inspecionar_dados():
    with ZipFile(ARQUIVO) as zip_file:
        arquivos = [
            nome for nome in zip_file.namelist()
            if nome.upper().endswith(".CSV")
        ]

        estacoes_sp = [
            nome for nome in arquivos
            if "_SP_" in normalizar(nome)
        ]

        print(f"\nTotal de arquivos CSV: {len(arquivos)}")
        print(f"Estações do estado de SP: {len(estacoes_sp)}")

        print("\nEstações próximas ao nosso interesse:")
        encontrados = []

        for nome in estacoes_sp:
            nome_normalizado = normalizar(nome)

            if any(
                termo in nome_normalizado
                for termo in ["SAO PAULO", "MIRANTE", "INTERLAGOS"]
            ):
                print(" -", nome)
                encontrados.append(nome)

        if not encontrados:
            print("Nenhuma estação da capital identificada pelo nome.")
            print("Exemplos de estações de SP:")
            for nome in estacoes_sp[:10]:
                print(" -", nome)

        if encontrados:
            print("\nPrévia da estrutura do primeiro arquivo:")
            with zip_file.open(encontrados[0]) as arquivo:
                for _ in range(10):
                    linha = arquivo.readline().decode(
                        "latin-1", errors="replace"
                    ).strip()
                    print(linha)


if __name__ == "__main__":
    baixar_dados()
    inspecionar_dados()
