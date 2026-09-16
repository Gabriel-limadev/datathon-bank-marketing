import pandas as pd

from src.datathon_bank_marketing.bandit.thompson_sampling import ThompsonSampling
from src.datathon_bank_marketing.pipeline.train import (
    load_data,
    prepare_data,
    split_data,
    train_bandit,
)


def test_prepare_data_creates_reward_and_age_group():
    df = pd.DataFrame(
        {
            "age": [25, 40],
            "y": ["yes", "no"],
        }
    )

    result = prepare_data(df)

    assert "reward" in result.columns
    assert "age_group" in result.columns
    assert result["reward"].tolist() == [1, 0]

def test_split_data():
    df = pd.DataFrame(
        {
            "age": range(100),
            "reward": [0] * 90 + [1] * 10,
        }
    )

    train, test = split_data(df)

    assert len(train) == 80
    assert len(test) == 20
    assert len(train) + len(test) == len(df)

def test_train_bandit():
    df = pd.DataFrame(
        {
            "contact": ["cellular"] * 20 + ["telephone"] * 20,
            "reward": [0, 1] * 10 + [0, 1] * 10,
            "age_group": ["26-35"] * 40,
            "job": ["admin."] * 40,
            "marital": ["single"] * 40,
            "education": ["university.degree"] * 40,
            "default": ["no"] * 40,
            "housing": ["no"] * 40,
            "loan": ["no"] * 40,
            "poutcome": ["nonexistent"] * 40,
            "campaign": [1] * 40,
            "previous": [0] * 40,
            "emp.var.rate": [1.0] * 40,
            "cons.price.idx": [93.0] * 40,
            "euribor3m": [4.0] * 40,
            "nr.employed": [5000.0] * 40,
        }
    )

    bandit = train_bandit(df)

    assert set(bandit.bootstrap_models.keys()) == {"cellular", "telephone"}
    assert len(bandit.bootstrap_models["cellular"]) == 20
    assert len(bandit.bootstrap_models["telephone"]) == 20

def test_recommend():
    df = pd.DataFrame(
        {
            "contact": ["cellular"] * 20 + ["telephone"] * 20,
            "reward": [0, 1] * 10 + [0, 1] * 10,
            "age_group": ["26-35"] * 40,
            "job": ["admin."] * 40,
            "marital": ["single"] * 40,
            "education": ["university.degree"] * 40,
            "default": ["no"] * 40,
            "housing": ["no"] * 40,
            "loan": ["no"] * 40,
            "poutcome": ["nonexistent"] * 40,
            "campaign": [1] * 40,
            "previous": [0] * 40,
            "emp.var.rate": [1.0] * 40,
            "cons.price.idx": [93.0] * 40,
            "euribor3m": [4.0] * 40,
            "nr.employed": [5000.0] * 40,
        }
    )

    bandit = train_bandit(df)

    customer = df.iloc[[0]][
        [
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
    ]

    selected_arm, probabilities = bandit.recommend(customer)

    assert selected_arm in {"cellular", "telephone"}
    assert set(probabilities.keys()) == {"cellular", "telephone"}
    assert all(0 <= probability <= 1 for probability in probabilities.values())

def test_load_data():

    df = load_data()

    assert not df.empty
    assert "age" in df.columns
    assert "contact" in df.columns
    assert "y" in df.columns

def test_save_and_load(tmp_path):
    df = pd.DataFrame(
        {
            "contact": ["cellular"] * 20 + ["telephone"] * 20,
            "reward": [0, 1] * 10 + [0, 1] * 10,
            "age_group": ["26-35"] * 40,
            "job": ["admin."] * 40,
            "marital": ["single"] * 40,
            "education": ["university.degree"] * 40,
            "default": ["no"] * 40,
            "housing": ["no"] * 40,
            "loan": ["no"] * 40,
            "poutcome": ["nonexistent"] * 40,
            "campaign": [1] * 40,
            "previous": [0] * 40,
            "emp.var.rate": [1.0] * 40,
            "cons.price.idx": [93.0] * 40,
            "euribor3m": [4.0] * 40,
            "nr.employed": [5000.0] * 40,
        }
    )

    bandit = train_bandit(df)

    bandit.save(tmp_path)

    loaded_bandit = ThompsonSampling.load(
        tmp_path,
        arms=["cellular", "telephone"],
    )

    assert len(loaded_bandit.bootstrap_models["cellular"]) == 20
    assert len(loaded_bandit.bootstrap_models["telephone"]) == 20