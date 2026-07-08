from __future__ import annotations

import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request

from research_platform import (
    FEATURE_COLUMNS,
    build_research_payload,
    compare_strategies,
    engineer_features,
    risk_management,
    series_value,
    volatility_adjusted_signal,
)


APP_NAME = "Vantage Point Research Engine"
STATIC_VERSION = "2"

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / ".app_state"
DEMO_DIR = STATE_DIR / "demo_history"
ENTRY_FILE = STATE_DIR / "last_entry.json"
MODEL_FILE = BASE_DIR / "models" / "vantage_point_models.joblib"
LEGACY_MODEL_FILE = BASE_DIR / "models" / "pulse_v3_models.joblib"  # pre-rename fallback
STATE_DIR.mkdir(exist_ok=True)
DEMO_DIR.mkdir(exist_ok=True)

YF_CACHE_DIR = STATE_DIR / "yf_cache"
YF_CACHE_DIR.mkdir(exist_ok=True)
try:
    yf.set_tz_cache_location(str(YF_CACHE_DIR))
except Exception:
    pass

MODEL_BUNDLE: dict[str, Any] | None = None
HISTORY_CACHE: dict[tuple[str, str], tuple[datetime, pd.DataFrame]] = {}
CACHE_TTL_SECONDS = 300
PROVIDER_TIMEOUT_SECONDS = 5

# Cache of ticker-validity lookups so a repeatedly-requested bad (or good)
# symbol doesn't re-hit the search endpoint on every call within the TTL.
TICKER_VALIDITY_CACHE: dict[str, tuple[datetime, bool]] = {}
TICKER_VALIDITY_TTL_SECONDS = 3600


PERIODS = {
    "1d": "5m",
    "5d": "15m",
    "3mo": "1h",
    "6mo": "1d",
    "1y": "1d",
    "2y": "1d",
    "5y": "1wk",
}


def safe_key(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "default"


def clone_history(data: pd.DataFrame) -> pd.DataFrame:
    cloned = data.copy()
    cloned.attrs.update(data.attrs)
    return cloned


def load_models() -> dict[str, Any] | None:
    global MODEL_BUNDLE
    if MODEL_BUNDLE is not None:
        return MODEL_BUNDLE
    model_path = MODEL_FILE if MODEL_FILE.exists() else LEGACY_MODEL_FILE if LEGACY_MODEL_FILE.exists() else None
    if model_path is None:
        return None
    MODEL_BUNDLE = joblib.load(model_path)
    for model in MODEL_BUNDLE.get("models", {}).values():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    return MODEL_BUNDLE


def ticker_exists(symbol: str) -> bool:
    """Verify a symbol is a real, tradeable ticker via Yahoo Finance's search endpoint.

    This exists purely to gate the synthetic demo-data fallback: demo data should
    stand in for a REAL ticker when live providers are unreachable, not silently
    fabricate a chart for a typo'd or nonexistent symbol. Results are cached for
    an hour since ticker validity almost never changes within a session.

    If the validation call itself fails (e.g. same outage that took down the real
    data providers), we fail OPEN (return True) so a full network outage doesn't
    also break the offline demo fallback, which exists specifically for that case.
    """
    symbol_key = symbol.upper()
    cached = TICKER_VALIDITY_CACHE.get(symbol_key)
    if cached and (datetime.now() - cached[0]).total_seconds() < TICKER_VALIDITY_TTL_SECONDS:
        return cached[1]

    valid_quote_types = {"EQUITY", "ETF", "INDEX", "MUTUALFUND", "CRYPTOCURRENCY", "CURRENCY"}
    try:
        query = urlencode({"q": symbol, "quotesCount": 5, "newsCount": 0})
        req = Request(
            f"https://query1.finance.yahoo.com/v1/finance/search?{query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(req, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        quotes = payload.get("quotes", [])
        is_valid = any(
            quote.get("symbol", "").upper() == symbol_key
            and quote.get("quoteType") in valid_quote_types
            for quote in quotes
        )
        TICKER_VALIDITY_CACHE[symbol_key] = (datetime.now(), is_valid)
        return is_valid
    except Exception:
        # Fail open when ticker validation itself encounters an outage.
        # This allows offline demo fallback to continue working.
        TICKER_VALIDITY_CACHE[symbol_key] = (datetime.now(), True)
        return True


def download_history(symbol: str, period: str) -> pd.DataFrame:
    key = (symbol.upper(), period)
    cached = HISTORY_CACHE.get(key)
    if cached and (datetime.now() - cached[0]).total_seconds() < CACHE_TTL_SECONDS:
        data = clone_history(cached[1])
        data.attrs["cache"] = "memory"
        return data

    errors = []
    try:
        data = download_yahoo_chart(symbol, period)
        if not data.empty:
            data.attrs["source"] = "Yahoo Finance chart API"
            data.attrs["source_kind"] = "live"
            HISTORY_CACHE[key] = (datetime.now(), clone_history(data))
            return clone_history(data)
        errors.append("Yahoo chart API returned no rows")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Yahoo chart API failed: {exc}")

    try:
        data = yf.download(
            symbol,
            period=period,
            interval=PERIODS.get(period, "1d"),
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna()
            data.attrs["source"] = "Yahoo Finance"
            data.attrs["source_kind"] = "live"
            HISTORY_CACHE[key] = (datetime.now(), clone_history(data))
            return clone_history(data)
        errors.append("Yahoo returned no rows")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Yahoo failed: {exc}")

    try:
        data = download_stooq(symbol, period)
        data.attrs["source_kind"] = "live"
        HISTORY_CACHE[key] = (datetime.now(), clone_history(data))
        return clone_history(data)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Stooq failed: {exc}")

    # Never fabricate charts for invalid tickers.
    if not ticker_exists(symbol):
        raise ValueError(
            f"Invalid ticker '{symbol.upper()}'. Please enter a valid stock symbol."
        )

    # Only real tickers are allowed to fall back to demo mode.
    demo = demo_history(symbol, period)
    demo.attrs["source"] = "offline persistent demo"
    demo.attrs["source_kind"] = "demo"
    demo.attrs["provider_errors"] = errors
    HISTORY_CACHE[key] = (datetime.now(), clone_history(demo))
    return clone_history(demo)


def download_yahoo_chart(symbol: str, period: str) -> pd.DataFrame:
    query = urlencode({"range": period, "interval": PERIODS.get(period, "1d")})
    request = Request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result", [None])[0]
    if not result or not result.get("timestamp"):
        return pd.DataFrame()

    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        },
        index=pd.to_datetime(result["timestamp"], unit="s"),
    )
    return frame.apply(pd.to_numeric, errors="coerce").dropna()


def download_stooq(symbol: str, period: str) -> pd.DataFrame:
    ticker = symbol.lower() if "." in symbol else f"{symbol.lower()}.us"
    days = {"1d": 10, "5d": 20, "3mo": 330, "6mo": 520, "1y": 720, "2y": 1100, "5y": 2200}.get(period, 720)
    today = date.today()
    query = urlencode({"s": ticker, "d1": (today - timedelta(days=days)).strftime("%Y%m%d"), "d2": today.strftime("%Y%m%d"), "i": "d"})
    request = Request(f"https://stooq.com/q/d/l/?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
        csv_text = response.read().decode("utf-8")
    data = pd.read_csv(io.StringIO(csv_text))
    if data.empty or "Date" not in data.columns:
        raise RuntimeError("no rows")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    data.attrs["source"] = "Stooq"
    return data


def demo_history(symbol: str, period: str) -> pd.DataFrame:
    path = DEMO_DIR / f"{safe_key(symbol)}_{safe_key(period)}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    rows = 96 if period == "1d" else 160 if period == "5d" else 260 if period in {"3mo", "6mo", "1y"} else 620
    seed = sum((idx + 1) * ord(char) for idx, char in enumerate(symbol.upper()))
    rng = np.random.default_rng(seed)
    if period == "1d":
        dates = pd.date_range(end=pd.Timestamp.now().floor("5min"), periods=rows, freq="5min")
    elif period == "5d":
        dates = pd.date_range(end=pd.Timestamp.now().floor("15min"), periods=rows, freq="15min")
    else:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    close = (120 + seed % 80) * np.exp(np.cumsum(rng.normal(((seed % 17) - 7) / 10000, 0.018, rows)))
    open_price = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, rows))
    spread = np.abs(rng.normal(0.012, 0.005, rows))
    data = pd.DataFrame(
        {
            "Open": open_price,
            "High": np.maximum(open_price, close) * (1 + spread),
            "Low": np.minimum(open_price, close) * (1 - spread),
            "Close": close,
            "Volume": rng.integers(1_200_000, 12_000_000, rows),
        },
        index=dates,
    )
    data.index.name = "Date"
    data.to_csv(path)
    return data


def model_prediction(data: pd.DataFrame) -> dict[str, Any] | None:
    bundle = load_models()
    if not bundle:
        return None
    featured = data.dropna(subset=FEATURE_COLUMNS)
    if featured.empty:
        return None
    latest = featured.iloc[[-1]][FEATURE_COLUMNS]
    predictions = []
    for name, model in bundle["models"].items():
        probability_up = float(model.predict_proba(latest)[0][1])
        predictions.append({"name": name, "probability_up": round(probability_up, 4)})
    best_name = bundle["best_model"]
    best = next(item for item in predictions if item["name"] == best_name)
    adjusted = volatility_adjusted_signal(best["probability_up"], float(featured.iloc[-1]["ATRPct"]))
    return {
        "best_model": best_name,
        "probability_up": adjusted["adjusted_probability_up"],
        "raw_probability_up": adjusted["raw_probability_up"],
        "volatility_penalty": adjusted["volatility_penalty"],
        "model_predictions": predictions,
        "leaderboard": bundle.get("leaderboard", []),
        "trained_at": bundle.get("trained_at"),
        "horizon_days": bundle.get("horizon_days", 5),
        "optional_models": bundle.get("optional_models", {}),
    }


def technical_prediction(data: pd.DataFrame) -> dict[str, Any]:
    latest = data.iloc[-1]
    probability_up = float(latest["BayesianUpProbability"])
    adjusted = volatility_adjusted_signal(probability_up, float(latest["ATRPct"]))
    return {
        "best_model": "Weighted Bayesian Technical Model",
        "probability_up": adjusted["adjusted_probability_up"],
        "raw_probability_up": adjusted["raw_probability_up"],
        "volatility_penalty": adjusted["volatility_penalty"],
        "model_predictions": [{"name": "Weighted Bayesian", "probability_up": adjusted["adjusted_probability_up"]}],
        "leaderboard": [],
        "trained_at": None,
        "horizon_days": 5,
        "optional_models": {},
    }


def direction_from_probability(probability_up: float) -> str:
    if probability_up >= 0.55:
        return "Bullish"
    if probability_up <= 0.45:
        return "Bearish"
    return "Sideways"


def build_forecast(close: float, atr_value: float, probability_up: float, risk: dict[str, Any]) -> dict[str, Any]:
    horizon = 12
    directional_strength = (probability_up - 0.5) * 2
    expected_move = directional_strength * atr_value * 2.4
    if risk["side"] == "Watch":
        expected_move = directional_strength * atr_value

    points = []
    for step in range(1, horizon + 1):
        progress = step / horizon
        curve = 1 - (1 - progress) ** 1.7
        noise_band = atr_value * (0.35 + progress * 0.9)
        mid = close + expected_move * curve
        points.append(
            {
                "step": step,
                "price": round(mid, 2),
                "upper": round(mid + noise_band, 2),
                "lower": round(mid - noise_band, 2),
            }
        )

    return {
        "horizon_bars": horizon,
        "expected_price": points[-1]["price"],
        "trend": "Up" if expected_move > atr_value * 0.25 else "Down" if expected_move < -atr_value * 0.25 else "Flat",
        "confidence": round(max(probability_up, 1 - probability_up), 4),
        "points": points,
        "target": risk["target"],
        "stop_loss": risk["stop_loss"],
    }


def prediction_payload(symbol: str, period: str, record_entry: bool = True) -> dict[str, Any]:
    raw = download_history(symbol, period)
    source = raw.attrs.get("source", "market data")
    source_kind = raw.attrs.get("source_kind", "live")
    provider_errors = raw.attrs.get("provider_errors", [])
    data = engineer_features(raw).dropna()
    minimum_rows = 35 if period == "1d" else 50 if period == "5d" else 80
    if len(data) < minimum_rows:
        raise RuntimeError("Not enough usable history.")

    ml = model_prediction(data) or technical_prediction(data)
    probability_up = ml["probability_up"]
    direction = direction_from_probability(probability_up)
    confidence = int(round(max(probability_up, 1 - probability_up) * 100))
    latest = data.iloc[-1]
    close = series_value(latest["Close"])
    atr_value = max(series_value(latest["ATR"]), close * 0.01)
    risk = risk_management(close, atr_value, probability_up)
    forecast = build_forecast(close, atr_value, probability_up, risk)
    backtests = compare_strategies(data)

    recent = data.tail(140)
    candles = [
        {
            "time": idx.strftime("%Y-%m-%d %H:%M"),
            "open": round(series_value(row["Open"]), 2),
            "high": round(series_value(row["High"]), 2),
            "low": round(series_value(row["Low"]), 2),
            "close": round(series_value(row["Close"]), 2),
            "volume": int(series_value(row["Volume"])),
            "sma20": round(series_value(row["SMA20"]), 2),
            "sma50": round(series_value(row["SMA50"]), 2),
        }
        for idx, row in recent.iterrows()
    ]

    payload = {
        "symbol": symbol.upper(),
        "period": period,
        "source": source,
        "source_kind": source_kind,
        "provider_errors": provider_errors,
        "direction": direction,
        "confidence": confidence,
        "last_price": round(close, 2),
        "target_peak": risk["target"],
        "stop_loss": risk["stop_loss"],
        "risk_reward": risk["risk_reward"],
        "rsi": round(series_value(latest["RSI"]), 2),
        "macd": round(series_value(latest["MACD"]), 2),
        "atr": round(atr_value, 2),
        "weighted_score": round(series_value(latest["WeightedScore"]), 4),
        "bayesian_probability_up": round(series_value(latest["BayesianUpProbability"]), 4),
        "ml_prediction": ml,
        "risk_management": risk,
        "forecast": forecast,
        "backtests": backtests,
        "best_strategy": max(backtests, key=lambda item: item["sharpe_ratio"]),
        "candles": candles,
        "notice": f"{'Live data active' if source_kind == 'live' else 'Demo data active'}: {source}. Educational research estimate only.",
    }
    entry = save_entry(payload) if record_entry else read_entry()
    payload["monitor"] = monitor_payload(payload, entry)
    return payload


def read_entry() -> dict[str, Any] | None:
    if not ENTRY_FILE.exists():
        return None
    try:
        return json.loads(ENTRY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_entry(payload: dict[str, Any]) -> dict[str, Any]:
    side = "Long" if payload["direction"] == "Bullish" else "Short" if payload["direction"] == "Bearish" else "Watch"
    entry = {
        "symbol": payload["symbol"],
        "period": payload["period"],
        "side": side,
        "entry_price": payload["last_price"],
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_peak": payload["target_peak"],
        "stop_loss": payload["stop_loss"],
        "confidence": payload["confidence"],
        "direction": payload["direction"],
    }
    ENTRY_FILE.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return entry


def monitor_payload(payload: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry:
        return None
    side = entry.get("side", "Watch")
    multiplier = 1 if side == "Long" else -1 if side == "Short" else 0
    current = payload["last_price"]
    pnl = (current - float(entry["entry_price"])) * multiplier
    return {
        **entry,
        "current_price": current,
        "pnl": round(pnl, 2),
        "pnl_percent": round((pnl / float(entry["entry_price"])) * 100, 2) if entry["entry_price"] else 0,
        "status": "Monitoring",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.after_request
def disable_browser_cache(response):
    if response.content_type and "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index() -> str:
    return render_template("index.html", app_name=APP_NAME, static_version=STATIC_VERSION)


@app.route("/api/predict")
def predict() -> Any:
    symbol = request.args.get("symbol", "AAPL").strip().upper()
    period = request.args.get("period", "1y")
    record = request.args.get("record", "1") != "0"
    try:
        return jsonify(prediction_payload(symbol, period, record))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.route("/api/backtest")
def backtest() -> Any:
    symbol = request.args.get("symbol", "AAPL").strip().upper()
    period = request.args.get("period", "2y")
    try:
        data = engineer_features(download_history(symbol, period)).dropna()
        return jsonify({"symbol": symbol, "period": period, "strategies": compare_strategies(data)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.route("/api/monitor")
def monitor() -> Any:
    entry = read_entry()
    if not entry:
        return jsonify({"error": "No saved entry yet."}), 404
    try:
        payload = prediction_payload(entry["symbol"], entry["period"], record_entry=False)
        return jsonify(payload["monitor"])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.route("/api/last-entry")
def last_entry() -> Any:
    entry = read_entry()
    if not entry:
        return jsonify({"error": "No saved entry yet."}), 404
    return jsonify(entry)


@app.route("/api/research")
def research() -> Any:
    symbol = request.args.get("symbol", "AAPL").strip().upper()
    period = request.args.get("period", "2y")
    try:
        data = download_history(symbol, period)
        return jsonify(build_research_payload(symbol, data))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print(f"Starting {APP_NAME} at http://127.0.0.1:5003")
    app.run(debug=False, host="127.0.0.1", port=5003)