from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "bank_marketing_processed.csv"


def prepare_data(df):
    """Prepara e organiza os dados para treinamento."""
    df = df.copy()

    if "y" in df.columns:
        df["reward"] = (df["y"] == "yes").astype(int)
        
    df = df.drop_duplicates()
    df = df.drop(columns=["duration"], errors="ignore")

    
    bins = [0, 25, 35, 45, 55, 65, 75, 100]
    labels = [
        "Até 25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "66-75",
        "76+",
    ]

    df["age_group"] = pd.cut(
        df["age"],
        bins=bins,
        labels=labels,
    )

    return df

def save_processed_data(df):
    """Salva os dados processados em CSV."""

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, sep=";")