import shutil
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

RAW_DIR = Path("data/raw")
CSV_PATH = RAW_DIR / "bank-additional-full.csv"
ZIP_PATH = RAW_DIR / "bank_marketing.zip"


def download_data() -> pd.DataFrame:
    """
    Realiza o download do dataset do UCI, extrai o CSV e o salva em data/raw/bank-additional-full.csv.
    Se o CSV já existir, ele não será baixado novamente.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Se o CSV já existe, não baixa novamente
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH, sep=";")

    print("Baixando dataset do UCI...")

    # Download do ZIP principal
    urlretrieve(URL, ZIP_PATH)

    # Diretório temporário para extração
    temp_dir = RAW_DIR / "temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir()

    try:
        # Extrai bank_marketing.zip
        with ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Localiza bank-additional.zip
        additional_zip = next(temp_dir.rglob("bank-additional.zip"))

        # Extrai bank-additional.zip
        additional_dir = temp_dir / "bank-additional"

        with ZipFile(additional_zip, "r") as zip_ref:
            zip_ref.extractall(additional_dir)

        # Localiza o CSV principal
        source_csv = next(additional_dir.rglob("bank-additional-full.csv"))

        # Move o CSV para data/raw/
        shutil.move(str(source_csv), str(CSV_PATH))

        print("Dataset preparado com sucesso.")

    finally:
        # Remove ZIP e todos os arquivos temporários
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # Carrega o CSV
    return pd.read_csv(CSV_PATH, sep=";")
