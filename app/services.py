from pathlib import Path

import pandas as pd

from src.datathon_bank_marketing.bandit.thompson_sampling import ThompsonSampling
from src.datathon_bank_marketing.data.download import download_data
from src.datathon_bank_marketing.data.preprocess import (
    prepare_data,
    save_processed_data,
)
from src.datathon_bank_marketing.pipeline.train import (
    load_data,
    split_data,
    train_bandit,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "bootstrap"
ARMS = ["cellular", "telephone"]

bandit = ThompsonSampling.load(
    MODEL_PATH,
    arms=ARMS,
)


def download_service():
    """Baixa e prepara o dataset do UCI, retornando um DataFrame."""

    try:
        df = download_data()
        return {
            "message": "Dataset baixado e preparado com sucesso.",
            "rows": len(df),
            "columns": len(df.columns),
        }
    except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Erro ao baixar dados: {e!s}") from e


def recommend_service(customer):
    """Gera uma recomendação de canal para um cliente."""

    try:
        
        customer_df = pd.DataFrame([customer])
        
        customer_df = prepare_data(customer_df)
        # Ajustar nomes das colunas para os nomes usados no treinamento
        customer_df = customer_df.rename(
            columns={
                "emp_var_rate": "emp.var.rate",
                "cons_price_idx": "cons.price.idx",
                "nr_employed": "nr.employed",
                "cons_conf_idx": "cons.conf.idx",
            }
        )
        
        selected_arm, probabilities = bandit.recommend(customer_df)
        
        
        return {
            "recommended_arm": selected_arm,
            "estimated_probability": probabilities[selected_arm],
        }
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Erro ao gerar recomendação: {e!s}") from e

def train_service():
    """Retreina o Thompson Sampling e salva os novos modelos."""

    df = load_data()
    df = prepare_data(df)
    save_processed_data(df)

    train, _ = split_data(df)

    new_bandit = train_bandit(train)
    new_bandit.save(MODEL_PATH)

    global bandit
    bandit = ThompsonSampling.load(
        MODEL_PATH,
        arms=ARMS,
    )

    return {
        "status": "success",
        "message": "Modelos retreinados e salvos com sucesso.",
    }