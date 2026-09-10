from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.datathon_bank_marketing.bandit.thompson_sampling import ThompsonSampling

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "raw" / "bank-additional-full.csv"
MODEL_PATH = BASE_DIR / "models" / "bootstrap"
ARMS = ["cellular", "telephone"]

context_features = [
    "age_group",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "poutcome",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "euribor3m",
    "nr.employed",
]

categorical_features = [
    "age_group",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "poutcome",
]

numeric_features = [
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "euribor3m",
    "nr.employed",
]

N_BOOTSTRAPS = 20


def load_data():
    """Carrega a base de dados de treinamento."""

    df = pd.read_csv(DATA_PATH, sep=";")

    return df


def prepare_data(df):
    """Prepara as variáveis necessárias para o treinamento."""

    df = df.copy()

    df["reward"] = (df["y"] == "yes").astype(int)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 75, float("inf")],
        labels=[
            "Até 25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "66-75",
            "76+",
        ],
    )

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

    train, test = split_data(df)

    bandit = train_bandit(train)

    bandit.save(MODEL_PATH)
