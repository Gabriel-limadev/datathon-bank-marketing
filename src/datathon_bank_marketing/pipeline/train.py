import os
from pathlib import Path

import mlflow
import numpy as np
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
    "month",
]

categorical_features = [
    "age_group",
    "poutcome",
    "campaign",
    "month",
]

numeric_features = []

N_BOOTSTRAPS = 50


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

def evaluate_bandit(bandit, test):
    """Avalia o Thompson Sampling usando Replay Method."""

    ts_decisions = []
    diffs = []

    for _, row in test.iterrows():
        customer = row[context_features].to_frame().T

        selected_arm, sampled_predictions = bandit.recommend(customer)

        diff = abs(
            sampled_predictions["cellular"]
            - sampled_predictions["telephone"]
        )

        diffs.append(diff)
        ts_decisions.append(selected_arm)

    test_bootstrap = test.copy()
    test_bootstrap["selected_arm"] = ts_decisions

    test_bootstrap["evaluated_reward"] = test_bootstrap.apply(
        lambda row: (
            row["reward"]
            if row["selected_arm"] == row["contact"]
            else None
        ),
        axis=1,
    )

    evaluated_bts = test_bootstrap.dropna(
        subset=["evaluated_reward"]
    )

    bts_conversion = evaluated_bts["evaluated_reward"].mean()

    feedback_rate = len(evaluated_bts) / len(test)

    arm_distribution = (
        test_bootstrap["selected_arm"]
        .value_counts(normalize=True)
    )

    cellular_pct = arm_distribution.get("cellular", 0)
    telephone_pct = arm_distribution.get("telephone", 0)

    mean_diff = np.mean(diffs)
    median_diff = np.median(diffs)

    print(
        f"Casos avaliados no Bootstrap TS: {len(evaluated_bts)}"
    )
    print(f"Taxa de feedback: {feedback_rate:.2%}")
    print(
        f"Conversão Bootstrap TS (Replay): "
        f"{bts_conversion:.2%}"
    )
    print(f"Diferença média: {mean_diff}")
    print(f"Diferença mediana: {median_diff}")

    print("\n--- Distribuição de Seleção dos Braços ---")
    print(
        pd.DataFrame(
            {"Bootstrap TS": arm_distribution}
        ).fillna(0)
    )

    return {
        "conversion": bts_conversion,
        "feedback_rate": feedback_rate,
        "evaluated_cases": len(evaluated_bts),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "cellular_pct": cellular_pct,
        "telephone_pct": telephone_pct,
    }


def log_mlflow():
    """Registra os parâmetros do treinamento no MLflow."""

    mlflow.log_param(
        "algorithm",
        "Bootstrap Thompson Sampling",
    )

    mlflow.log_param(
        "n_bootstraps",
        N_BOOTSTRAPS,
    )

    mlflow.log_param(
        "context_features",
        ", ".join(context_features),
    )

    mlflow.log_param(
        "categorical_features",
        ", ".join(categorical_features),
    )



if __name__ == "__main__":

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI")
    )
    mlflow.set_experiment("bank_marketing")

    with mlflow.start_run(
        run_name="bootstrap_thompson_sampling"
    ):
        df = load_data()

        df = prepare_data(df)
        save_processed_data(df)

        train, test = split_data(df)

        bandit = train_bandit(train)

        metrics = evaluate_bandit(bandit, test)

        bandit.save(MODEL_PATH)

        log_mlflow()

        mlflow.log_metrics(metrics)