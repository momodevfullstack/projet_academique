"""Train and compare sales forecasting models on the Rossmann dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TEST_DAYS = 42
MAX_TRAIN_ROWS = 100_000

NUMERIC_FEATURES = [
    "DayOfWeek",
    "Day",
    "Month",
    "Year",
    "WeekOfYear",
    "IsWeekend",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
]
CATEGORICAL_FEATURES = [
    "Store",
    "StoreType",
    "Assortment",
    "StateHoliday",
    "PromoInterval",
]


def load_data(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load, validate, and join the competition's training and store files."""
    train_path = data_dir / "train.csv"
    stores_path = data_dir / "store.csv"
    missing = [path.name for path in (train_path, stores_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Fichier(s) manquant(s) dans {data_dir}: {', '.join(missing)}. "
            "Consultez la section « Données » du README.md."
        )

    sales = pd.read_csv(
        train_path,
        parse_dates=["Date"],
        dtype={"StateHoliday": "string"},
        low_memory=False,
    )
    stores = pd.read_csv(stores_path)
    sales = sales.drop_duplicates()
    sales = sales.dropna(subset=["Date", "Store", "Sales"])
    sales["StateHoliday"] = sales["StateHoliday"].fillna("0").replace("0", "none")

    data = sales.merge(stores, on="Store", how="left", validate="many_to_one")
    if data["StoreType"].isna().any():
        raise ValueError("Certaines lignes de ventes n'ont pas de magasin correspondant.")

    # Closed stores are not demand observations; customer counts are excluded
    # because they are only known after the sales day and would leak the target.
    data = data.loc[data["Open"].eq(1)].copy()
    data["CompetitionDistance"] = data["CompetitionDistance"].fillna(
        data["CompetitionDistance"].median()
    )
    return data.sort_values("Date").reset_index(drop=True)


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features using only information available on the forecast date."""
    featured = data.copy()
    featured["Day"] = featured["Date"].dt.day
    featured["Month"] = featured["Date"].dt.month
    featured["Year"] = featured["Date"].dt.year
    featured["WeekOfYear"] = featured["Date"].dt.isocalendar().week.astype("int16")
    featured["IsWeekend"] = featured["DayOfWeek"].isin([6, 7]).astype("int8")
    return featured


def temporal_split(
    data: pd.DataFrame, test_days: int = TEST_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the final calendar days so evaluation does not mix future and past."""
    if test_days < 1:
        raise ValueError("test_days doit être supérieur ou égal à 1.")
    if data.empty:
        raise ValueError("Impossible de séparer un jeu de données vide.")

    cutoff = data["Date"].max() - pd.Timedelta(days=test_days - 1)
    train = data.loc[data["Date"] < cutoff].copy()
    test = data.loc[data["Date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Le découpage temporel doit produire deux ensembles non vides.")
    return train, test


def build_preprocessor() -> ColumnTransformer:
    """Encode categorical features and pass numerical features through unchanged."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    """Return comparable tree-based regressors with one shared preprocessing pipeline."""
    return {
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=60,
                        max_depth=16,
                        min_samples_leaf=3,
                        max_features=0.8,
                        n_jobs=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=160,
                        max_depth=7,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=5.0,
                        tree_method="hist",
                        n_jobs=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate errors in the original unit used by the source dataset."""
    predicted = np.maximum(predicted, 0)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def save_reports(
    results: list[dict[str, float | str]],
    predictions: pd.DataFrame,
    data: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).sort_values("MAE").to_csv(
        REPORTS_DIR / "metrics.csv", index=False
    )
    predictions.to_csv(REPORTS_DIR / "predictions.csv", index=False)

    sns.set_theme(style="whitegrid")
    monthly_sales = data.set_index("Date")["Sales"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    monthly_sales.plot(ax=ax, color="#2563eb", linewidth=2)
    ax.set(title="Ventes quotidiennes moyennes par mois", xlabel="Date", ylabel="Ventes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_sales.png", dpi=150)
    plt.close(fig)

    daily = predictions.groupby("Date", as_index=False).agg(
        Actual=("Actual", "mean"),
        **{
            name: ("Prediction", "mean")
            for name in predictions["Model"].unique()
        },
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["Date"], daily["Actual"], label="Réel", color="#111827", linewidth=2)
    for model_name in predictions["Model"].unique():
        ax.plot(daily["Date"], daily[model_name], label=model_name, alpha=0.85)
    ax.set(title="Ventes réelles et prédites — période de test", xlabel="Date", ylabel="Ventes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_predictions.png", dpi=150)
    plt.close(fig)


def run(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    data = add_features(load_data(data_dir))
    train, test = temporal_split(data)
    if len(train) > MAX_TRAIN_ROWS:
        train = train.sample(n=MAX_TRAIN_ROWS, random_state=42).sort_values("Date")

    x_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = np.log1p(train["Sales"].to_numpy())
    y_test = test["Sales"].to_numpy()

    results: list[dict[str, float | str]] = []
    prediction_frames = []
    for name, model in build_models().items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        predicted = np.expm1(model.predict(x_test))
        metrics = calculate_metrics(y_test, predicted)
        elapsed = time.perf_counter() - started

        results.append({"Model": name, **metrics, "TrainingSeconds": elapsed})
        prediction_frames.append(
            pd.DataFrame(
                {
                    "Date": test["Date"].to_numpy(),
                    "Store": test["Store"].to_numpy(),
                    "Actual": y_test,
                    "Prediction": np.maximum(predicted, 0),
                    "Model": name,
                }
            )
        )
        print(
            f"{name}: MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} "
            f"| entraînement={elapsed:.1f}s"
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    save_reports(results, predictions, data)
    report = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    print(
        f"\nPériode d'entraînement : {train['Date'].min().date()} "
        f"au {train['Date'].max().date()}"
    )
    print(f"Lignes d'entraînement utilisées : {len(train):,}")
    print(
        f"Période de test : {test['Date'].min().date()} "
        f"au {test['Date'].max().date()} ({test['Date'].nunique()} jours)"
    )
    print(f"\nMeilleur modèle sur le test (MAE) : {report.loc[0, 'Model']}")
    print(f"Résultats sauvegardés dans : {REPORTS_DIR}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Dossier contenant train.csv et store.csv (défaut : data/raw).",
    )
    args = parser.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
