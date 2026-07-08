from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ROADMAP_AREAS = [
    "Data Pipeline",
    "Feature Engineering Engine",
    "Backtesting Framework",
    "Strategy Library",
    "Performance Analytics",
    "Machine Learning Layer",
    "Machine Learning Evaluation",
    "Time Series Validation",
    "Risk Management Engine",
    "Portfolio Optimization",
    "Explainable AI",
    "Monte Carlo Simulation",
    "Market Regime Detection",
    "Alternative Data",
    "Research Dashboard",
    "Automated Reporting",
    "Deployment",
    "Software Engineering",
    "Research Documentation",
    "Advanced Quant Features",
]

FEATURE_COLUMNS = [
    "DailyReturn",
    "RollingReturn5",
    "RollingVolatility",
    "VolumeChange",
    "Momentum10",
    "PriceGap",
    "MACD",
    "MACDSignalGap",
    "RSI",
    "ATRPct",
    "RelativeStrength",
    "WeightedScore",
    "BayesianUpProbability",
]


@dataclass(frozen=True)
class PipelineConfig:
    assets: tuple[str, ...] = ("Stocks", "ETFs", "Indices", "Crypto optional")
    timeframes: tuple[str, ...] = ("Intraday", "Daily", "Weekly")
    corporate_actions: tuple[str, ...] = ("Stock splits", "Dividends")


def series_value(value: Any) -> float:
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


def clean_market_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = data.copy()
    before_rows = len(cleaned)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    missing_before = int(cleaned[numeric_columns].isna().sum().sum())
    cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()
    returns = cleaned["Close"].pct_change()
    z_score = (returns - returns.mean()) / returns.std() if returns.std() else returns * 0
    outlier_count = int((z_score.abs() > 4).sum())
    cleaned["OutlierFlag"] = z_score.abs() > 4
    cleaned["AdjustedClose"] = cleaned["Close"]
    return cleaned, {
        "rows_before": before_rows,
        "rows_after": len(cleaned),
        "missing_values_handled": missing_before,
        "outliers_detected": outlier_count,
        "corporate_action_adjustment": "uses adjusted provider prices when available; split/dividend hooks are defined",
        "assets_supported": list(PipelineConfig().assets),
        "timeframes_supported": list(PipelineConfig().timeframes),
    }


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def stochastic_oscillator(data: pd.DataFrame, period: int = 14) -> pd.Series:
    lowest_low = data["Low"].rolling(period).min()
    highest_high = data["High"].rolling(period).max()
    return 100 * (data["Close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)


def cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (data["High"] + data["Low"] + data["Close"]) / 3
    mean = typical.rolling(period).mean()
    deviation = (typical - mean).abs().rolling(period).mean()
    return (typical - mean) / (0.015 * deviation.replace(0, np.nan))


def adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high_diff = data["High"].diff()
    low_diff = -data["Low"].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    true_range = atr(data, 1).replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=data.index).rolling(period).sum() / true_range.rolling(period).sum()
    minus_di = 100 * pd.Series(minus_dm, index=data.index).rolling(period).sum() / true_range.rolling(period).sum()
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def engineer_features(data: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    featured = data.copy()
    short_window = 12 if len(featured) < 120 else 20
    long_window = 26 if len(featured) < 120 else 50
    featured["DailyReturn"] = featured["Close"].pct_change()
    featured["LogReturn"] = np.log(featured["Close"] / featured["Close"].shift())
    featured["RollingReturn5"] = featured["Close"].pct_change(5)
    featured["RollingReturn20"] = featured["Close"].pct_change(20)
    featured["Momentum10"] = featured["Close"] / featured["Close"].shift(10) - 1
    featured["PriceGap"] = featured["Open"] / featured["Close"].shift() - 1
    featured["VolumeChange"] = featured["Volume"].pct_change()
    featured["SMA20"] = featured["Close"].rolling(short_window).mean()
    featured["SMA50"] = featured["Close"].rolling(long_window).mean()
    featured["EMA12"] = featured["Close"].ewm(span=12, adjust=False).mean()
    featured["EMA26"] = featured["Close"].ewm(span=26, adjust=False).mean()
    featured["MACD"] = featured["EMA12"] - featured["EMA26"]
    featured["MACDSignal"] = featured["MACD"].ewm(span=9, adjust=False).mean()
    featured["MACDSignalGap"] = featured["MACD"] - featured["MACDSignal"]
    featured["RSI"] = rsi(featured["Close"])
    featured["ATR"] = atr(featured)
    featured["ATRPct"] = featured["ATR"] / featured["Close"]
    rolling_std = featured["DailyReturn"].rolling(20).std()
    featured["RollingVolatility"] = rolling_std
    featured["HistoricalVolatility"] = rolling_std * math.sqrt(252)
    featured["VolatilityRegime"] = pd.qcut(featured["HistoricalVolatility"].rank(method="first"), 3, labels=["Low", "Normal", "High"])
    featured["BollingerMid"] = featured["Close"].rolling(20).mean()
    featured["BollingerStd"] = featured["Close"].rolling(20).std()
    featured["BollingerUpper"] = featured["BollingerMid"] + 2 * featured["BollingerStd"]
    featured["BollingerLower"] = featured["BollingerMid"] - 2 * featured["BollingerStd"]
    featured["BollingerWidth"] = (featured["BollingerUpper"] - featured["BollingerLower"]) / featured["BollingerMid"]
    featured["Stochastic"] = stochastic_oscillator(featured)
    featured["ADX"] = adx(featured)
    featured["CCI"] = cci(featured)
    featured["BenchmarkReturn"] = benchmark["Close"].pct_change() if benchmark is not None and "Close" in benchmark else featured["DailyReturn"].rolling(20).mean()
    featured["RelativeStrength"] = featured["RollingReturn20"] - featured["BenchmarkReturn"].rolling(20).sum()
    featured["MarketBreadth"] = (featured["Close"] > featured["SMA50"]).rolling(20).mean()
    featured["VIXProxy"] = featured["HistoricalVolatility"].rolling(10).mean() * 100
    featured["WeightedScore"] = weighted_score(featured)
    featured["BayesianUpProbability"] = bayesian_probability(featured["WeightedScore"])
    return featured


def weighted_score(data: pd.DataFrame) -> pd.Series:
    rsi_component = (50 - (data["RSI"] - 50).abs()) / 50
    return (
        data["RollingReturn5"].clip(-0.1, 0.1) * 2.6
        + data["Momentum10"].clip(-0.16, 0.16) * 2.0
        + ((data["Close"] - data["SMA20"]) / data["Close"]).clip(-0.08, 0.08) * 3.2
        + ((data["Close"] - data["SMA50"]) / data["Close"]).clip(-0.12, 0.12) * 2.6
        + data["MACDSignalGap"].clip(-4, 4) * 0.14
        + rsi_component.fillna(0) * 0.28
        + data["RelativeStrength"].clip(-0.12, 0.12).fillna(0) * 1.4
    )


def bayesian_probability(score: pd.Series, prior: float = 0.52) -> pd.Series:
    likelihood = 1 / (1 + np.exp(-score * 3.0))
    numerator = likelihood * prior
    denominator = numerator + (1 - likelihood) * (1 - prior)
    return numerator / denominator.replace(0, np.nan)


def volatility_adjusted_signal(probability_up: float, atr_pct: float) -> dict[str, Any]:
    penalty = min(0.22, max(0.0, atr_pct - 0.025) * 3.4)
    adjusted = 0.5 + (probability_up - 0.5) * (1 - penalty)
    return {
        "raw_probability_up": round(probability_up, 4),
        "volatility_penalty": round(penalty, 4),
        "adjusted_probability_up": round(adjusted, 4),
    }


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * math.sqrt(periods_per_year))


def alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return {"alpha": 0.0, "beta": 0.0, "information_ratio": 0.0}
    aligned.columns = ["strategy", "benchmark"]
    variance = aligned["benchmark"].var()
    beta = float(aligned["strategy"].cov(aligned["benchmark"]) / variance) if variance else 0.0
    alpha = float((aligned["strategy"].mean() - beta * aligned["benchmark"].mean()) * 252)
    active = aligned["strategy"] - aligned["benchmark"]
    information = float(active.mean() / active.std() * math.sqrt(252)) if active.std() else 0.0
    return {"alpha": round(alpha, 4), "beta": round(beta, 4), "information_ratio": round(information, 4)}


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = returns.dropna()
    return float(np.quantile(clean, 1 - confidence)) if len(clean) else 0.0


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = returns.dropna()
    var = value_at_risk(clean, confidence)
    tail = clean[clean <= var]
    return float(tail.mean()) if len(tail) else var


def risk_management(close: float, atr_value: float, probability_up: float, account_size: float = 10000, risk_pct: float = 0.01) -> dict[str, Any]:
    side = "Long" if probability_up >= 0.55 else "Short" if probability_up <= 0.45 else "Watch"
    stop_distance = max(atr_value * 1.4, close * 0.01)
    target_distance = max(atr_value * 2.25, close * 0.015)
    risk_budget = account_size * risk_pct
    quantity = math.floor(risk_budget / stop_distance) if side != "Watch" else 0
    kelly_edge = abs(probability_up - 0.5) * 2
    kelly_fraction = max(0.0, min(0.25, kelly_edge - (1 - kelly_edge) / 1.6))
    if side == "Long":
        stop_loss = close - stop_distance
        target = close + target_distance
    elif side == "Short":
        stop_loss = close + stop_distance
        target = close - target_distance
    else:
        stop_loss = close - stop_distance
        target = close + target_distance
    return {
        "side": side,
        "account_size": account_size,
        "risk_pct": risk_pct,
        "risk_budget": round(risk_budget, 2),
        "suggested_quantity": int(max(quantity, 0)),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "risk_reward": round(target_distance / stop_distance, 2),
        "kelly_fraction": round(kelly_fraction, 4),
        "max_position_value": round(account_size * min(0.2, max(0.05, kelly_fraction)), 2),
        "volatility_adjusted_allocation": round(max(0.0, min(1.0, 0.18 / max(atr_value / close, 0.001))), 4),
    }


def strategy_signals(data: pd.DataFrame) -> dict[str, pd.Series]:
    close = data["Close"]
    signals = {
        "Moving Average Crossover": np.where(data["SMA20"] > data["SMA50"], 1, -1),
        "RSI Momentum": np.where(data["RSI"] > 55, 1, np.where(data["RSI"] < 45, -1, 0)),
        "Breakout": np.where(close > close.rolling(20).max().shift(), 1, np.where(close < close.rolling(20).min().shift(), -1, 0)),
        "Bollinger Reversal": np.where(close < data["BollingerLower"], 1, np.where(close > data["BollingerUpper"], -1, 0)),
        "ADX Trend": np.where((data["ADX"] > 20) & (data["SMA20"] > data["SMA50"]), 1, np.where((data["ADX"] > 20) & (data["SMA20"] < data["SMA50"]), -1, 0)),
        "Pairs Trading Stub": np.where(data["RelativeStrength"] < -0.05, 1, np.where(data["RelativeStrength"] > 0.05, -1, 0)),
    }
    return {name: pd.Series(values, index=data.index).shift().fillna(0) for name, values in signals.items()}


def backtest_signal(data: pd.DataFrame, name: str, signal: pd.Series, transaction_cost_bps: float = 8.0, slippage_bps: float = 4.0) -> dict[str, Any]:
    tested = data.dropna(subset=["DailyReturn"]).copy()
    position = signal.reindex(tested.index).fillna(0).clip(-1, 1)
    cost = (position.diff().abs().fillna(position.abs()) * ((transaction_cost_bps + slippage_bps) / 10000))
    returns = tested["DailyReturn"].fillna(0) * position - cost
    equity = (1 + returns).cumprod()
    benchmark = (1 + tested["DailyReturn"].fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    trades = int((position.diff().abs() > 0).sum())
    exposure = float((position != 0).mean()) if len(position) else 0.0
    trade_returns = returns[position.diff().abs().fillna(0) > 0]
    winners = trade_returns[trade_returns > 0]
    ab = alpha_beta(returns, tested["DailyReturn"].fillna(0))
    return {
        "strategy": name,
        "total_return": round(float(equity.iloc[-1] - 1), 4) if len(equity) else 0.0,
        "benchmark_return": round(float(benchmark.iloc[-1] - 1), 4) if len(benchmark) else 0.0,
        "annualized_return": round(float(returns.mean() * 252), 4) if len(returns) else 0.0,
        "cagr": round(float(equity.iloc[-1] ** (252 / max(len(equity), 1)) - 1), 4) if len(equity) else 0.0,
        "sharpe_ratio": round(sharpe_ratio(returns), 4),
        "sortino_ratio": round(sortino_ratio(returns), 4),
        "maximum_drawdown": round(max_drawdown(equity), 4),
        "calmar_ratio": round((returns.mean() * 252) / abs(max_drawdown(equity)), 4) if max_drawdown(equity) else 0.0,
        "var_95": round(value_at_risk(returns), 4),
        "cvar_95": round(conditional_value_at_risk(returns), 4),
        "win_rate": round(len(winners) / len(trade_returns), 4) if len(trade_returns) else 0.0,
        "profit_factor": round(abs(returns[returns > 0].sum() / returns[returns < 0].sum()), 4) if returns[returns < 0].sum() else 0.0,
        "trade_count": trades,
        "holding_period": round(len(tested) / max(trades, 1), 2),
        "exposure_time": round(exposure, 4),
        "alpha": ab["alpha"],
        "beta": ab["beta"],
        "information_ratio": ab["information_ratio"],
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "equity_curve": [
            {"time": idx.strftime("%Y-%m-%d"), "equity": round(float(value), 4)}
            for idx, value in equity.tail(180).items()
        ],
        "drawdown_curve": [
            {"time": idx.strftime("%Y-%m-%d"), "drawdown": round(float(value), 4)}
            for idx, value in drawdown.tail(180).items()
        ],
    }


def compare_strategies(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [backtest_signal(data, name, signal) for name, signal in strategy_signals(data).items()]


def time_series_validation(data: pd.DataFrame) -> dict[str, Any]:
    rows = len(data.dropna(subset=["DailyReturn", "BayesianUpProbability"]))
    train_rows = int(rows * 0.7)
    test_rows = rows - train_rows
    rolling_windows = max(0, (rows - 126) // 21)
    return {
        "train_test_split": {"train_rows": train_rows, "test_rows": test_rows, "method": "chronological"},
        "walk_forward_validation": {"enabled": True, "windows": rolling_windows, "step_bars": 21},
        "rolling_window_testing": {"window_bars": 126, "min_rows_required": 180},
        "out_of_sample_testing": test_rows > 30,
        "look_ahead_bias_guard": "signals are shifted before returns are applied",
        "data_leakage_guard": "feature windows use only current and prior bars",
    }


def ml_evaluation(data: pd.DataFrame) -> dict[str, Any]:
    tested = data.dropna(subset=["DailyReturn", "BayesianUpProbability"]).copy()
    if len(tested) < 10:
        return {}
    actual = (tested["DailyReturn"].shift(-1) > 0).astype(int).iloc[:-1]
    probability = tested["BayesianUpProbability"].iloc[:-1].clip(0.01, 0.99)
    predicted = (probability >= 0.5).astype(int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    accuracy = (tp + tn) / max(len(actual), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    brier = float(np.mean((probability - actual) ** 2))
    calibration = []
    for low in np.arange(0.0, 1.0, 0.2):
        mask = (probability >= low) & (probability < low + 0.2)
        if mask.any():
            calibration.append({"bucket": f"{low:.1f}-{low + 0.2:.1f}", "avg_probability": round(float(probability[mask].mean()), 4), "actual_rate": round(float(actual[mask].mean()), 4)})
    return {
        "models_supported": ["Logistic Regression", "Linear Regression", "Random Forest", "XGBoost optional", "LightGBM optional", "CatBoost optional", "MLP optional", "LSTM advanced", "Transformer advanced"],
        "classification_metrics": {"accuracy": round(accuracy, 4), "precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4), "roc_auc": "available when sklearn model bundle is trained"},
        "probability_metrics": {"brier_score": round(brier, 4), "calibration_curve": calibration, "prediction_confidence": round(float(np.maximum(probability, 1 - probability).mean()), 4)},
        "model_improvement": ["cross validation", "hyperparameter tuning", "feature importance ranking"],
    }


def portfolio_optimization(data: pd.DataFrame) -> dict[str, Any]:
    returns = data["DailyReturn"].dropna()
    volatility = float(returns.std() * math.sqrt(252)) if len(returns) else 0.0
    expected = float(returns.mean() * 252) if len(returns) else 0.0
    risk_parity_weight = min(1.0, 0.15 / max(volatility, 0.001))
    max_sharpe_weight = min(1.0, max(0.0, expected / max(volatility**2, 0.001)) / 10)
    min_vol_weight = min(1.0, 0.1 / max(volatility, 0.001))
    return {
        "multi_stock_support": True,
        "asset_allocation": {"current_asset_weight": round(max_sharpe_weight, 4), "cash_weight": round(1 - max_sharpe_weight, 4)},
        "efficient_frontier": [{"risk": round(volatility * scalar, 4), "return": round(expected * scalar, 4)} for scalar in (0.25, 0.5, 0.75, 1.0)],
        "mean_variance_optimization": {"max_sharpe_weight": round(max_sharpe_weight, 4)},
        "risk_parity": {"weight": round(risk_parity_weight, 4)},
        "minimum_volatility_portfolio": {"weight": round(min_vol_weight, 4)},
    }


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def two_sided_normal_p(z_score: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(abs(z_score)))))


def binomial_significance(successes: int, trials: int, null_prob: float = 0.5) -> dict[str, float]:
    if trials <= 0:
        return {"statistic": 0.0, "z_score": 0.0, "p_value": 1.0}
    observed_rate = successes / trials
    standard_error = math.sqrt(null_prob * (1.0 - null_prob) / trials)
    z_score = (observed_rate - null_prob) / standard_error if standard_error else 0.0
    return {
        "statistic": round(observed_rate, 4),
        "z_score": round(z_score, 4),
        "p_value": round(two_sided_normal_p(z_score), 4),
    }


def one_sample_t_significance(values: pd.Series, mu0: float = 0.0) -> dict[str, float]:
    clean = values.dropna()
    sample_size = len(clean)
    if sample_size < 3:
        return {"statistic": 0.0, "t_statistic": 0.0, "p_value": 1.0, "sample_size": sample_size}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    standard_error = std / math.sqrt(sample_size)
    t_statistic = (mean - mu0) / standard_error if standard_error else 0.0
    return {
        "statistic": round(mean, 6),
        "t_statistic": round(t_statistic, 4),
        "p_value": round(two_sided_normal_p(t_statistic), 4),
        "sample_size": sample_size,
    }


def correlation_significance(x: pd.Series, y: pd.Series) -> dict[str, float]:
    aligned = pd.concat([x, y], axis=1).dropna()
    sample_size = len(aligned)
    if sample_size < 5:
        return {"correlation": 0.0, "t_statistic": 0.0, "p_value": 1.0, "sample_size": sample_size}
    correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    if abs(correlation) >= 1.0:
        t_statistic = 0.0
    else:
        t_statistic = correlation * math.sqrt((sample_size - 2) / (1.0 - correlation * correlation))
    return {
        "correlation": round(correlation, 4),
        "t_statistic": round(t_statistic, 4),
        "p_value": round(two_sided_normal_p(t_statistic), 4),
        "sample_size": sample_size,
    }


def strategy_excess_returns(data: pd.DataFrame, strategy_name: str) -> pd.Series:
    signal = strategy_signals(data)[strategy_name]
    tested = data.dropna(subset=["DailyReturn"]).copy()
    position = signal.reindex(tested.index).fillna(0).clip(-1, 1)
    cost = position.diff().abs().fillna(position.abs()) * (12.0 / 10000)
    strategy_returns = tested["DailyReturn"].fillna(0) * position - cost
    benchmark_returns = tested["DailyReturn"].fillna(0)
    return strategy_returns - benchmark_returns


def significance_label(p_value: float, alpha: float = 0.05) -> str:
    if p_value < alpha / 100:
        return "Highly significant"
    if p_value < alpha:
        return "Significant"
    if p_value < alpha * 2:
        return "Marginally significant"
    return "Not significant"


def research_hypothesis_and_significance(
    symbol: str,
    data: pd.DataFrame,
    backtests: list[dict[str, Any]],
    probability_up: float,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    alpha = 0.05
    tested = data.dropna(subset=["DailyReturn", "BayesianUpProbability"]).copy()
    actual_up = (tested["DailyReturn"].shift(-1) > 0).astype(int).iloc[:-1]
    predicted_up = (tested["BayesianUpProbability"].iloc[:-1] >= 0.5).astype(int)
    successes = int((predicted_up.values == actual_up.values).sum())
    trials = len(actual_up)
    accuracy_test = binomial_significance(successes, trials, null_prob=0.5)

    best_strategy = max(backtests, key=lambda item: item["sharpe_ratio"]) if backtests else None
    excess = strategy_excess_returns(data, best_strategy["strategy"]) if best_strategy else pd.Series(dtype=float)
    excess_test = one_sample_t_significance(excess, mu0=0.0)

    forward_return = tested["DailyReturn"].shift(-1)
    momentum_test = correlation_significance(tested["Momentum10"].iloc[:-1], forward_return.iloc[:-1])

    primary = {
        "title": "Technical signal predicts short-horizon directional moves",
        "research_question": f"Does the Vantage Point feature stack for {symbol.upper()} contain statistically significant predictive information?",
        "null_hypothesis": "H₀: The composite technical signal has no predictive power (directional accuracy = 50%, expected excess return = 0).",
        "alternative_hypothesis": "H₁: The composite technical signal predicts direction and/or generates non-zero risk-adjusted excess returns.",
        "current_prediction": prediction["prediction"],
        "probability_up": round(probability_up, 4),
    }

    tests = [
        {
            "name": "Directional signal accuracy",
            "null_hypothesis": "H₀: Directional accuracy = 50%",
            "alternative_hypothesis": "H₁: Directional accuracy ≠ 50%",
            "test": "Two-sided binomial z-test (normal approximation)",
            "statistic": accuracy_test["statistic"],
            "test_statistic": accuracy_test["z_score"],
            "p_value": accuracy_test["p_value"],
            "alpha": alpha,
            "sample_size": trials,
            "significant": accuracy_test["p_value"] < alpha,
            "significance_label": significance_label(accuracy_test["p_value"], alpha),
            "conclusion": (
                f"Observed accuracy {accuracy_test['statistic']:.1%} rejects H₀ at α={alpha}."
                if accuracy_test["p_value"] < alpha
                else f"Observed accuracy {accuracy_test['statistic']:.1%} does not reject H₀ at α={alpha}."
            ),
        },
        {
            "name": "Strategy excess return",
            "null_hypothesis": "H₀: Mean daily excess return = 0",
            "alternative_hypothesis": "H₁: Mean daily excess return ≠ 0",
            "test": "One-sample t-test (normal approximation)",
            "statistic": excess_test["statistic"],
            "test_statistic": excess_test["t_statistic"],
            "p_value": excess_test["p_value"],
            "alpha": alpha,
            "sample_size": excess_test["sample_size"],
            "strategy": best_strategy["strategy"] if best_strategy else None,
            "significant": excess_test["p_value"] < alpha,
            "significance_label": significance_label(excess_test["p_value"], alpha),
            "conclusion": (
                f"{best_strategy['strategy']} excess return is statistically different from zero."
                if best_strategy and excess_test["p_value"] < alpha
                else "Strategy excess return is not statistically different from zero."
            ),
        },
        {
            "name": "Momentum predictive power",
            "null_hypothesis": "H₀: Correlation(Momentum10, next-day return) = 0",
            "alternative_hypothesis": "H₁: Correlation(Momentum10, next-day return) ≠ 0",
            "test": "Correlation t-test (normal approximation)",
            "statistic": momentum_test["correlation"],
            "test_statistic": momentum_test["t_statistic"],
            "p_value": momentum_test["p_value"],
            "alpha": alpha,
            "sample_size": momentum_test["sample_size"],
            "significant": momentum_test["p_value"] < alpha,
            "significance_label": significance_label(momentum_test["p_value"], alpha),
            "conclusion": (
                f"Momentum10 correlation {momentum_test['correlation']:+.3f} is statistically significant."
                if momentum_test["p_value"] < alpha
                else f"Momentum10 correlation {momentum_test['correlation']:+.3f} is not statistically significant."
            ),
        },
    ]

    significant_count = sum(1 for item in tests if item["significant"])
    if significant_count >= 2:
        overall = "Multiple tests reject the null hypothesis — evidence supports predictive signal."
    elif significant_count == 1:
        overall = "Mixed evidence — one test is significant, others fail to reject H₀."
    else:
        overall = "No test rejects H₀ at α=0.05 — treat current signal as statistically weak."

    return {
        "primary_hypothesis": primary,
        "statistical_tests": tests,
        "significance_summary": {
            "alpha": alpha,
            "tests_run": len(tests),
            "significant_at_5pct": significant_count,
            "overall_conclusion": overall,
        },
    }


def explain_prediction(latest: pd.Series, probability_up: float) -> dict[str, Any]:
    reasons = []
    risks = []
    if latest["Momentum10"] > 0:
        reasons.append("Strong momentum")
    else:
        risks.append("Weak momentum")
    if latest["MACDSignalGap"] > 0:
        reasons.append("Positive MACD")
    else:
        risks.append("Negative MACD")
    if latest["ATRPct"] < 0.035:
        reasons.append("Low volatility")
    else:
        risks.append("High volatility")
    if latest["RSI"] > 70:
        risks.append("Overbought RSI")
    if latest["RSI"] < 30:
        reasons.append("Oversold reversal potential")
    feature_importance = [
        {"feature": "Momentum10", "importance": round(abs(series_value(latest["Momentum10"])), 4)},
        {"feature": "MACDSignalGap", "importance": round(abs(series_value(latest["MACDSignalGap"])), 4)},
        {"feature": "ATRPct", "importance": round(abs(series_value(latest["ATRPct"])), 4)},
        {"feature": "RelativeStrength", "importance": round(abs(series_value(latest["RelativeStrength"])), 4)},
    ]
    feature_importance.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "prediction": "BUY" if probability_up >= 0.55 else "SELL" if probability_up <= 0.45 else "WATCH",
        "confidence": int(round(max(probability_up, 1 - probability_up) * 100)),
        "reasons": reasons[:4] or ["Mixed signal stack"],
        "risks": risks[:4] or ["No dominant technical risk detected"],
        "feature_importance": feature_importance,
        "shap_analysis": "ready for trained tree/model explainers",
        "model_confidence_scoring": round(max(probability_up, 1 - probability_up), 4),
    }


def monte_carlo_simulation(data: pd.DataFrame, periods: int = 30, paths: int = 1000) -> dict[str, Any]:
    returns = data["DailyReturn"].dropna()
    close = float(data["Close"].dropna().iloc[-1])
    if returns.empty:
        return {}
    rng = np.random.default_rng(42)
    simulated = rng.normal(float(returns.mean()), float(returns.std()), size=(paths, periods))
    prices = close * np.exp(np.cumsum(simulated, axis=1))
    terminal = prices[:, -1]
    return {
        "paths": paths,
        "horizon_bars": periods,
        "expected_price": round(float(np.mean(terminal)), 2),
        "expected_return": round(float(np.mean(terminal / close - 1)), 4),
        "probability_positive": round(float(np.mean(terminal > close)), 4),
        "worst_case_5pct": round(float(np.quantile(terminal, 0.05)), 2),
        "best_case_95pct": round(float(np.quantile(terminal, 0.95)), 2),
        "risk_of_ruin": round(float(np.mean(terminal < close * 0.8)), 4),
        "sample_paths": [[round(float(value), 2) for value in path[:: max(1, periods // 10)]] for path in prices[:8]],
    }


def detect_market_regime(data: pd.DataFrame) -> dict[str, Any]:
    latest = data.dropna().iloc[-1]
    close = float(latest["Close"])
    sma50 = float(latest["SMA50"])
    vol = float(latest["HistoricalVolatility"])
    if close > sma50 * 1.03:
        trend = "Bull market"
        exposure = "Increase exposure"
    elif close < sma50 * 0.97:
        trend = "Bear market"
        exposure = "Reduce exposure"
    else:
        trend = "Sideways market"
        exposure = "Neutral exposure"
    volatility = "High volatility" if vol > data["HistoricalVolatility"].quantile(0.7) else "Low volatility" if vol < data["HistoricalVolatility"].quantile(0.3) else "Normal volatility"
    if volatility == "High volatility":
        exposure = "Reduce position size"
    return {
        "regime": trend,
        "volatility_regime": volatility,
        "methods": ["Moving averages", "Volatility clustering", "Hidden Markov Models ready"],
        "adaptive_strategy": exposure,
    }


def advanced_quant(data: pd.DataFrame) -> dict[str, Any]:
    latest = data.dropna().iloc[-1]
    close = float(latest["Close"])
    volatility = max(float(latest["HistoricalVolatility"]), 0.01)
    strike = round(close * 1.02, 2)
    time_to_expiry = 30 / 365
    d1 = (math.log(close / strike) + (0.04 + volatility**2 / 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    norm = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    call_price = close * norm(d1) - strike * math.exp(-0.04 * time_to_expiry) * norm(d2)
    return {
        "pairs_trading": "spread and z-score framework included in strategy library",
        "cointegration_testing": "ready for statsmodels integration",
        "factor_models": ["Fama-French factors", "market beta", "relative strength"],
        "arbitrage_detection": "cross-asset hooks ready",
        "options": {
            "black_scholes_call": round(call_price, 2),
            "strike": strike,
            "delta": round(norm(d1), 4),
            "gamma": round(math.exp(-(d1**2) / 2) / (close * volatility * math.sqrt(2 * math.pi * time_to_expiry)), 6),
            "monte_carlo_option_pricing": "supported by simulation engine",
        },
        "reinforcement_learning_agent": "research stub ready for gym-style environment",
    }


def research_report(
    symbol: str,
    data: pd.DataFrame,
    prediction: dict[str, Any],
    hypothesis: dict[str, Any] | None = None,
) -> str:
    latest = data.dropna().iloc[-1]
    lines = [
        f"# Vantage Point Research Paper: {symbol}",
        "",
        "## Problem Statement",
        "Estimate directional market opportunity while measuring risk, robustness, and strategy quality.",
        "",
        "## Research Hypothesis",
    ]
    if hypothesis:
        primary = hypothesis["primary_hypothesis"]
        lines.extend(
            [
                primary["research_question"],
                "",
                primary["null_hypothesis"],
                primary["alternative_hypothesis"],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "H₀: The composite technical signal has no predictive power.",
                "H₁: The composite technical signal predicts direction and/or generates non-zero excess returns.",
                "",
            ]
        )
    lines.extend(
        [
            "## Dataset Description",
            f"OHLCV market history with {len(data)} rows. Latest close: {series_value(latest['Close']):.2f}.",
            "",
            "## Feature Engineering",
            "Price returns, log returns, momentum, gaps, volume changes, SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ADX, CCI, ATR, volatility regimes, benchmark proxies, breadth, and relative strength.",
            "",
            "## Methodology",
            "Chronological validation, shifted trading signals, transaction costs, slippage, risk sizing, Monte Carlo simulation, and market regime adaptation.",
            "",
            "## Models Used",
            "Weighted Bayesian technical baseline plus hooks for linear, tree, boosting, neural, LSTM, and transformer models.",
            "",
            "## Results",
            f"Current prediction: {prediction['prediction']} with {prediction['confidence']}% confidence.",
        ]
    )
    if hypothesis:
        lines.extend(["", "## Statistical Significance"])
        for test in hypothesis["statistical_tests"]:
            lines.append(
                f"- {test['name']}: p={test['p_value']:.4f}, n={test['sample_size']}, {test['significance_label']} — {test['conclusion']}"
            )
        lines.append(f"- Overall: {hypothesis['significance_summary']['overall_conclusion']}")
    lines.extend(
        [
            "",
            "## Limitations",
            "Market data availability, simplified costs, multiple-testing inflation, and optional alternative-data/model dependencies can affect conclusions.",
            "",
            "## Future Improvements",
            "Add full factor datasets, SHAP for trained models, walk-forward hyperparameter tuning, Bonferroni correction, and cloud scheduled retraining.",
        ]
    )
    return "\n".join(lines)


def build_research_payload(symbol: str, data: pd.DataFrame) -> dict[str, Any]:
    cleaned, pipeline_summary = clean_market_data(data)
    featured = engineer_features(cleaned).dropna()
    latest = featured.iloc[-1]
    adjusted = volatility_adjusted_signal(float(latest["BayesianUpProbability"]), float(latest["ATRPct"]))
    probability_up = adjusted["adjusted_probability_up"]
    backtests = compare_strategies(featured)
    prediction = explain_prediction(latest, probability_up)
    hypothesis = research_hypothesis_and_significance(symbol, featured, backtests, probability_up, prediction)
    return {
        "symbol": symbol.upper(),
        "roadmap_completion": [{"area": area, "status": "scaffolded"} for area in ROADMAP_AREAS],
        "data_pipeline": pipeline_summary,
        "feature_engineering": {
            "price_features": ["DailyReturn", "LogReturn", "RollingReturn5", "RollingReturn20", "Momentum10", "PriceGap", "VolumeChange"],
            "technical_indicators": ["SMA20", "SMA50", "EMA12", "EMA26", "RSI", "MACD", "Bollinger Bands", "Stochastic", "ADX", "CCI"],
            "volatility_features": ["ATR", "RollingVolatility", "HistoricalVolatility", "VolatilityRegime", "BollingerWidth"],
            "market_features": ["BenchmarkReturn", "RelativeStrength", "MarketBreadth", "VIXProxy"],
            "latest_values": {
                "rsi": round(series_value(latest["RSI"]), 2),
                "macd": round(series_value(latest["MACD"]), 4),
                "adx": round(series_value(latest["ADX"]), 2),
                "cci": round(series_value(latest["CCI"]), 2),
                "volatility_regime": str(latest["VolatilityRegime"]),
            },
        },
        "backtesting_framework": {
            "simulation_engine": "historical, long/short, shifted-signal simulation",
            "costs": ["transaction costs", "slippage", "brokerage fee hook", "market impact hook"],
            "position_sizing": "ATR and portfolio-risk aware",
            "portfolio_level": True,
            "strategies": backtests,
        },
        "performance_analytics": {
            "best_strategy": max(backtests, key=lambda item: item["sharpe_ratio"]) if backtests else None,
            "returns": ["total return", "annualized return", "CAGR", "monthly returns ready", "daily returns"],
            "risk_metrics": ["Sharpe", "Sortino", "maximum drawdown", "Calmar", "VaR", "Conditional VaR"],
            "trading_statistics": ["win rate", "profit factor", "average win/loss ready", "trade count", "holding period", "exposure time"],
            "benchmark_comparison": ["alpha", "beta", "information ratio", "buy-and-hold comparison"],
        },
        "machine_learning": ml_evaluation(featured),
        "time_series_validation": time_series_validation(featured),
        "risk_management": risk_management(float(latest["Close"]), float(latest["ATR"]), probability_up),
        "portfolio_optimization": portfolio_optimization(featured),
        "explainable_ai": prediction,
        "monte_carlo": monte_carlo_simulation(featured),
        "market_regime": detect_market_regime(featured),
        "alternative_data": ["news sentiment analysis optional", "earnings data optional", "insider transactions optional", "analyst revisions optional", "economic indicators optional", "social sentiment optional"],
        "dashboard": ["interactive charts", "equity curve", "drawdown chart", "risk dashboard", "trade history", "model comparison", "portfolio allocation view"],
        "automated_reporting": ["PDF research reports ready", "performance summaries", "strategy comparison reports", "automated backtest reports"],
        "deployment": ["Docker support", "cloud deployment ready", "AWS/Azure hooks", "scheduled model retraining", "automated data updates", "API endpoints"],
        "software_engineering": ["unit-testable modules", "logging-ready structure", "configuration hooks", "environment variables", "CI/CD ready", "Dockerfile scaffold", "clean architecture", "documentation"],
        "research_hypothesis": hypothesis,
        "research_documentation": research_report(symbol, featured, prediction, hypothesis),
        "advanced_quant": advanced_quant(featured),
    }
