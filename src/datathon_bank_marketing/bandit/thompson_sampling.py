from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ThompsonSampling:
    def __init__(self, arms, n_bootstraps=20):
        self.arms = arms
        self.n_bootstraps = n_bootstraps
        self.bootstrap_models = {arm: [] for arm in arms}
        self.rng = np.random.default_rng(42)

    def fit(self, train, context_features, categorical_features, numeric_features):
        """Treina modelos bootstrap independentes para cada braço."""

        for arm in self.arms:
            print(f"FIT: iniciando braço {arm}", flush=True)

            arm_data = train[train["contact"] == arm].copy()

            print(
                f"FIT: {arm} possui {len(arm_data)} registros",
                flush=True,
            )

            for i in range(self.n_bootstraps):
                if i % 10 == 0:
                    print(
                        f"FIT: {arm} bootstrap {i}/{self.n_bootstraps}",
                        flush=True,
                    )

                bootstrap_sample = arm_data.sample(
                    n=len(arm_data),
                    replace=True,
                    random_state=42 + i,
                )

                X_bootstrap = bootstrap_sample[context_features]
                y_bootstrap = bootstrap_sample["reward"]

                preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical_features,
                        ),
                        (
                            "numeric",
                            StandardScaler(),
                            numeric_features,
                        ),
                    ]
                )

                model = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=100,
                                random_state=42,
                            ),
                        ),
                    ]
                )

                print(f"FIT: {arm} bootstrap {i} - iniciando model.fit()", flush=True)

            model.fit(X_bootstrap, y_bootstrap)

            print(f"FIT: {arm} bootstrap {i} - model.fit() concluído", flush=True)

            self.bootstrap_models[arm].append(model)

            print(f"FIT: braço {arm} concluído", flush=True)

    def recommend(self, customer):
        """Seleciona um braço usando Thompson Sampling contextual."""
        sampled_predictions = {}

        for arm in self.arms:
            models = self.bootstrap_models[arm]

            sampled_index = self.rng.integers(len(models))
            sampled_model = models[sampled_index]

            sampled_probability = sampled_model.predict_proba(customer)[0, 1]

            sampled_predictions[arm] = sampled_probability

        selected_arm = max(
            sampled_predictions,
            key=sampled_predictions.get,
        )

        return selected_arm, sampled_predictions

    def save(self, path):
        """Salva os modelos bootstrap de cada braço em disco."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for arm, models in self.bootstrap_models.items():
            arm_path = path / arm
            arm_path.mkdir(exist_ok=True)

            for i, model in enumerate(models):
                joblib.dump(model, arm_path / f"model_{i}.pkl")

    @classmethod
    def load(cls, path, arms):
        """Carrega os modelos bootstrap previamente salvos."""
        path = Path(path)

        bandit = cls(arms=arms)

        for arm in arms:
            arm_path = path / arm

            for model_file in sorted(arm_path.glob("model_*.pkl")):
                model = joblib.load(model_file)
                bandit.bootstrap_models[arm].append(model)

        return bandit
