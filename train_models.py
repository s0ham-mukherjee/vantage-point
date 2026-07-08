from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research_platform import FEATURE_COLUMNS, engineer_features

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "vantage_point_models.joblib"


def fetch_symbol(symbol: str, period: str) -> pd.DataFrame:
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data["Symbol"] = symbol
    return data.dropna()


def training_frame(symbols: list[str], period: str, horizon_days: int) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        try:
            raw = fetch_symbol(symbol, period)
            featured = engineer_features(raw)
            future_return = featured["Close"].shift(-horizon_days) / featured["Close"] - 1
            featured["Target"] = (future_return > 0).astype(int)
            featured = featured.dropna(subset=FEATURE_COLUMNS + ["Target"])
            featured["Symbol"] = symbol
            frames.append(featured)
            print(f"Loaded {symbol}: {len(featured)} rows")
        except Exception as exc:  # noqa: BLE001
            print(f"Skipped {symbol}: {exc}")
    if not frames:
        raise RuntimeError("No market data was available for training.")
    return pd.concat(frames, axis=0)


def model_candidates(include_neural: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1200, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=260,
            max_depth=9,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=260,
            learning_rate=0.04,
            max_depth=-1,
            random_state=42,
            verbose=-1,
        )
    if include_neural:
        models["Simple Neural Network"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)),
            ]
        )
    return models


def train_all(data: pd.DataFrame, symbols: list[str], horizon_days: int, include_neural: bool) -> dict[str, Any]:
    x = data[FEATURE_COLUMNS]
    y = data["Target"]
    split = int(len(data) * 0.78)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    trained = {}
    leaderboard = []

    for name, model in model_candidates(include_neural).items():
        print(f"\nTraining {name}")
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities) if y_test.nunique() > 1 else 0.5
        print(f"{name} accuracy={accuracy:.3f} auc={auc:.3f}")
        print(classification_report(y_test, predictions, target_names=["Down/Flat", "Up"]))
        trained[name] = model
        leaderboard.append({"name": name, "accuracy": round(float(accuracy), 4), "roc_auc": round(float(auc), 4)})

    leaderboard = sorted(leaderboard, key=lambda item: (item["roc_auc"], item["accuracy"]), reverse=True)
    return {
        "models": trained,
        "leaderboard": leaderboard,
        "best_model": leaderboard[0]["name"],
        "feature_names": FEATURE_COLUMNS,
        "symbols": symbols,
        "horizon_days": horizon_days,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": int(len(data)),
        "optional_models": {
            "xgboost_available": XGBClassifier is not None,
            "lightgbm_available": LGBMClassifier is not None,
            "neural_network_included": include_neural,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vantage Point Research Engine model suite on real market data.")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "SPY", "QQQ"])
    parser.add_argument("--period", default="10y")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--include-neural", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    data = training_frame(args.symbols, args.period, args.horizon_days)
    bundle = train_all(data, args.symbols, args.horizon_days, args.include_neural)
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nBest model: {bundle['best_model']}")
    print(f"Saved model suite to {MODEL_FILE}")


if __name__ == "__main__":
    main()
