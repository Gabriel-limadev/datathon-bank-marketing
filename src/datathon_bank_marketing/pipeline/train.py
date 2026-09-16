from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.datathon_bank_marketing.bandit.thompson_sampling import ThompsonSampling
from src.datathon_bank_marketing.data.preprocess import (
    prepare_data,
    save_processed_data,
)

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "bank-additional-full.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "bank_marketing_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "bootstrap"
ARMS = ["cellular", "telephone"]

context_features = [
    "age_group",
    "poutcome",
    "campaign",
    "month"
]

categorical_features = [
    "age_group",
    "poutcome",
    "campaign",
    "month"
]
 
numeric_features = [
]

N_BOOTSTRAPS = 100


def load_data():
    """Carrega a base de dados bruta."""

    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    return df


def split_data(df):
    """Divide os dados em treinamento e teste."""

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["reward"],
    )

    return train, test


def train_bandit(train):
    """Treina o Thompson Sampling com modelos bootstrap."""

    bandit = ThompsonSampling(
        arms=ARMS,
        n_bootstraps=N_BOOTSTRAPS,
    )
    
    bandit.fit(
        train=train,
        context_features=context_features,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    return bandit


if __name__ == "__main__":
    df = load_data()

    df = prepare_data(df)
    save_processed_data(df)

    train, test = split_data(df)

    bandit = train_bandit(train)

    bandit.save(MODEL_PATH)
