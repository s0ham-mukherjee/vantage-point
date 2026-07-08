let latestPayload;
let chartType = "candles";

const form = document.querySelector("#predict-form");
const symbolInput = document.querySelector("#symbol");
const periodInput = document.querySelector("#period");
const statusText = document.querySelector("#status");
const canvas = document.querySelector("#price-chart");
const equityCanvas = document.querySelector("#equity-chart");
const drawdownCanvas = document.querySelector("#drawdown-canvas");
const crosshair = document.querySelector("#crosshair");
const crosshairLabel = crosshair.querySelector("span");

const formatMoney = (value) => Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const signedMoney = (value) => `${Number(value) >= 0 ? "+" : "-"}${formatMoney(Math.abs(value))}`;

function setText(id, value) { document.querySelector(`#${id}`).textContent = value; }
function tone(value) { return value > 0 ? "positive" : value < 0 ? "negative" : "neutral"; }

function fitCanvas() {
  return fitSpecificCanvas(canvas);
}

function fitSpecificCanvas(target) {
  const ratio = window.devicePixelRatio || 1;
  const rect = target.getBoundingClientRect();
  target.width = Math.floor(rect.width * ratio);
  target.height = Math.floor(rect.height * ratio);
  const ctx = target.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function chartBounds(candles, payload) {
  const prices = candles.flatMap((candle) => [candle.high, candle.low, candle.sma20, candle.sma50]);
  prices.push(payload.target_peak, payload.stop_loss);
  if (payload.forecast) {
    payload.forecast.points.forEach((point) => prices.push(point.price, point.upper, point.lower));
  }
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const pad = Math.max((max - min) * 0.14, payload.atr * 2);
  return { min: min - pad, max: max + pad };
}

function drawLine(ctx, points, toX, toY, color, width = 1.5, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = toX(index);
    const y = toY(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function renderChart(payload) {
  latestPayload = payload;
  const { ctx, width, height } = fitCanvas();
  const candles = payload.candles;
  const forecast = payload.forecast?.points || [];
  const padding = { top: 24, right: 78, bottom: 34, left: 16 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const bounds = chartBounds(candles, payload);
  const totalSlots = candles.length + forecast.length;
  const step = plotWidth / Math.max(totalSlots - 1, 1);
  const bodyWidth = Math.max(3, Math.min(12, step * 0.58));
  const toX = (index) => padding.left + index * step;
  const toY = (price) => padding.top + ((bounds.max - price) / (bounds.max - bounds.min)) * plotHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#172236";
  ctx.font = "12px Inter, Segoe UI, sans-serif";
  ctx.fillStyle = "#8d99ae";
  for (let i = 0; i <= 6; i += 1) {
    const y = padding.top + (plotHeight / 6) * i;
    const price = bounds.max - ((bounds.max - bounds.min) / 6) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    ctx.fillText(formatMoney(price), width - padding.right + 12, y + 4);
  }
  drawLine(ctx, candles.map((candle) => candle.sma20), toX, toY, "#3b82f6");
  drawLine(ctx, candles.map((candle) => candle.sma50), toX, toY, "#f5b942");
  drawLine(ctx, Array.from({ length: totalSlots }, () => payload.target_peak), toX, toY, "#22ab94", 1, [7, 5]);
  drawLine(ctx, Array.from({ length: totalSlots }, () => payload.stop_loss), toX, toY, "#f23645", 1, [7, 5]);
  if (chartType === "worm") {
    ctx.shadowColor = "rgba(59, 130, 246, 0.45)";
    ctx.shadowBlur = 14;
    drawLine(ctx, candles.map((candle) => candle.close), toX, toY, "#7dd3fc", 3.2);
    ctx.shadowBlur = 0;
  } else {
    candles.forEach((candle, index) => {
      const x = toX(index);
      const color = candle.close >= candle.open ? "#22ab94" : "#f23645";
      const openY = toY(candle.open);
      const closeY = toY(candle.close);
      ctx.strokeStyle = color; ctx.fillStyle = color;
      ctx.beginPath(); ctx.moveTo(x, toY(candle.high)); ctx.lineTo(x, toY(candle.low)); ctx.stroke();
      ctx.fillRect(x - bodyWidth / 2, Math.min(openY, closeY), bodyWidth, Math.max(2, Math.abs(closeY - openY)));
    });
  }
  const last = candles[candles.length - 1];
  if (forecast.length) {
    const startIndex = candles.length - 1;
    const forecastPrices = [last.close, ...forecast.map((point) => point.price)];
    const upper = [last.close, ...forecast.map((point) => point.upper)];
    const lower = [last.close, ...forecast.map((point) => point.lower)];
    ctx.save();
    ctx.beginPath();
    upper.forEach((price, offset) => {
      const x = toX(startIndex + offset);
      const y = toY(price);
      if (offset === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    lower.slice().reverse().forEach((price, reverseIndex) => {
      const offset = lower.length - 1 - reverseIndex;
      ctx.lineTo(toX(startIndex + offset), toY(price));
    });
    ctx.closePath();
    ctx.fillStyle = payload.direction === "Bearish" ? "rgba(242, 54, 69, 0.12)" : "rgba(34, 171, 148, 0.12)";
    ctx.fill();
    ctx.restore();
    drawLine(ctx, forecastPrices, (offset) => toX(startIndex + offset), toY, payload.direction === "Bearish" ? "#f23645" : "#22ab94", 2.6, [8, 5]);
    const finalPoint = forecast[forecast.length - 1];
    ctx.fillStyle = payload.direction === "Bearish" ? "#f23645" : "#22ab94";
    ctx.beginPath();
    ctx.arc(toX(candles.length - 1 + forecast.length), toY(finalPoint.price), 4, 0, Math.PI * 2);
    ctx.fill();
  }

  const lastY = toY(last.close);
  ctx.fillStyle = payload.direction === "Bearish" ? "#f23645" : payload.direction === "Sideways" ? "#f5b942" : "#22ab94";
  ctx.fillRect(width - padding.right + 8, lastY - 11, 66, 22);
  ctx.fillStyle = "#fff";
  ctx.fillText(formatMoney(last.close), width - padding.right + 13, lastY + 4);
  ctx.fillStyle = "#22ab94";
  ctx.fillText(`Target ${formatMoney(payload.target_peak)}`, width - padding.right + 5, toY(payload.target_peak) - 8);
  ctx.fillStyle = "#f23645";
  ctx.fillText(`Stop ${formatMoney(payload.stop_loss)}`, width - padding.right + 5, toY(payload.stop_loss) + 16);
}

function renderStrategies(items) {
  const container = document.querySelector("#strategy-list");
  container.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "strategy-item";
    row.innerHTML = `
      <b>${item.strategy}</b>
      <span>Return <b class="${tone(item.total_return)}">${pct(item.total_return)}</b></span>
      <span>Sharpe <b>${item.sharpe_ratio}</b></span>
      <span>Drawdown <b class="negative">${pct(item.maximum_drawdown)}</b></span>
      <span>Win <b>${pct(item.win_rate)}</b></span>
      <span>Alpha <b class="${tone(item.alpha)}">${pct(item.alpha)}</b></span>
    `;
    container.appendChild(row);
  });
}

function renderSourceStatus(payload) {
  const sourceKind = payload.source_kind === "live" ? "Live" : "Demo";
  const sourceEl = document.querySelector("#source-kind");
  sourceEl.textContent = sourceKind;
  sourceEl.className = payload.source_kind === "live" ? "positive" : "neutral";
  const errors = payload.provider_errors || [];
  const detail = payload.source_kind === "live"
    ? payload.source
    : `Using ${payload.source}. Live providers unavailable${errors.length ? `: ${errors[0]}` : "."}`;
  setText("source-detail", detail);
}

function renderCompactRows(id, rows, emptyText = "No rows yet.") {
  const container = document.querySelector(`#${id}`);
  container.innerHTML = "";
  if (!rows || !rows.length) {
    const row = document.createElement("div");
    row.innerHTML = `<b>${emptyText}</b><span></span><span></span>`;
    container.appendChild(row);
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("div");
    row.innerHTML = item;
    container.appendChild(row);
  });
}

function renderModelPerformance(prediction, research) {
  const classification = research.machine_learning?.classification_metrics || {};
  const probability = research.machine_learning?.probability_metrics || {};
  setText("model-best", prediction.best_model);
  setText("model-accuracy", classification.accuracy !== undefined ? pct(classification.accuracy) : "--");
  setText("model-precision", classification.precision !== undefined ? pct(classification.precision) : "--");
  setText("model-recall", classification.recall !== undefined ? pct(classification.recall) : "--");
  setText("model-f1", classification.f1_score !== undefined ? pct(classification.f1_score) : "--");
  setText("model-brier", probability.brier_score !== undefined ? probability.brier_score : "--");

  const leaderboardRows = prediction.leaderboard?.length
    ? prediction.leaderboard.map((item) => `<b>${item.name}</b><span>Accuracy ${pct(item.accuracy)}</span><span>AUC ${Number(item.roc_auc).toFixed(3)}</span>`)
    : prediction.model_predictions.map((item) => `<b>${item.name}</b><span>Up ${pct(item.probability_up)}</span><span>${prediction.trained_at ? "Trained" : "Baseline"}</span>`);
  renderCompactRows("model-leaderboard", leaderboardRows, "Train models to populate leaderboard.");

  const calibrationRows = (probability.calibration_curve || []).map((item) => `<b>${item.bucket}</b><span>Pred ${pct(item.avg_probability)}</span><span>Actual ${pct(item.actual_rate)}</span>`);
  renderCompactRows("calibration-list", calibrationRows, "Calibration needs more data.");
}

function drawSeriesChart(target, points, valueKey, options = {}) {
  if (!target || !points || !points.length) return;
  const { ctx, width, height } = fitSpecificCanvas(target);
  const padding = { top: 18, right: 58, bottom: 24, left: 16 };
  const values = points.map((point) => Number(point[valueKey]));
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 0.01;
    max += 0.01;
  }
  const pad = (max - min) * 0.12;
  min -= pad;
  max += pad;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const toX = (index) => padding.left + (plotWidth * index) / Math.max(points.length - 1, 1);
  const toY = (value) => padding.top + ((max - value) / (max - min)) * plotHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0a0f18";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#172236";
  ctx.fillStyle = "#8d99ae";
  ctx.font = "12px Inter, Segoe UI, sans-serif";
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight / 4) * i;
    const value = max - ((max - min) / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    ctx.fillText(options.percent ? pct(value) : value.toFixed(2), width - padding.right + 10, y + 4);
  }

  ctx.save();
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = toX(index);
    const y = toY(Number(point[valueKey]));
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(toX(points.length - 1), height - padding.bottom);
  ctx.lineTo(toX(0), height - padding.bottom);
  ctx.closePath();
  ctx.fillStyle = options.fill || "rgba(34, 171, 148, 0.1)";
  ctx.fill();
  ctx.restore();

  drawLine(ctx, values, toX, toY, options.color || "#22ab94", 2.4);
}

function renderPerformanceCharts(bestStrategy) {
  if (!bestStrategy) return;
  setText("equity-title", `${bestStrategy.strategy} Equity Curve`);
  setText("equity-summary", `Return ${pct(bestStrategy.total_return)} | Sharpe ${bestStrategy.sharpe_ratio}`);
  setText("drawdown-title", `${bestStrategy.strategy} Drawdown`);
  setText("drawdown-summary", `Max ${pct(bestStrategy.maximum_drawdown)} | Calmar ${bestStrategy.calmar_ratio}`);
  drawSeriesChart(equityCanvas, bestStrategy.equity_curve, "equity", { color: "#22ab94", fill: "rgba(34, 171, 148, 0.12)" });
  drawSeriesChart(drawdownCanvas, bestStrategy.drawdown_curve, "drawdown", { color: "#f23645", fill: "rgba(242, 54, 69, 0.12)", percent: true });
}

function renderTags(id, values) {
  const container = document.querySelector(`#${id}`);
  container.innerHTML = "";
  values.forEach((value) => {
    const tag = document.createElement("span");
    tag.textContent = value;
    container.appendChild(tag);
  });
}

function renderRoadmap(items) {
  const container = document.querySelector("#roadmap-list");
  container.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("span");
    row.textContent = `${item.area}: ${item.status}`;
    container.appendChild(row);
  });
}

function renderSignificanceTests(tests) {
  const container = document.querySelector("#significance-tests");
  container.innerHTML = "";
  if (!tests?.length) {
    container.textContent = "No statistical tests available.";
    return;
  }
  tests.forEach((test) => {
    const row = document.createElement("div");
    row.className = "compact-row";
    const label = test.significant ? "positive" : "neutral";
    row.innerHTML = `<span>${test.name}</span><strong class="${label}">p=${test.p_value.toFixed(4)} · ${test.significance_label}</strong>`;
    container.appendChild(row);
    const detail = document.createElement("p");
    detail.className = "significance-detail";
    detail.textContent = test.conclusion;
    container.appendChild(detail);
  });
}

function renderResearch(payload) {
  setText("regime-label", payload.market_regime.regime);
  setText("regime-action", `${payload.market_regime.volatility_regime}. ${payload.market_regime.adaptive_strategy}.`);
  setText("xai-prediction", payload.explainable_ai.prediction);
  setText("xai-confidence", `${payload.explainable_ai.confidence}% confidence`);
  renderTags("xai-reasons", payload.explainable_ai.reasons);
  renderTags("xai-risks", payload.explainable_ai.risks);
  setText("mc-expected", formatMoney(payload.monte_carlo.expected_price));
  setText("mc-positive", pct(payload.monte_carlo.probability_positive));
  setText("mc-worst", formatMoney(payload.monte_carlo.worst_case_5pct));
  setText("portfolio-weight", pct(payload.portfolio_optimization.mean_variance_optimization.max_sharpe_weight));
  setText("risk-parity", pct(payload.portfolio_optimization.risk_parity.weight));
  setText("min-vol", pct(payload.portfolio_optimization.minimum_volatility_portfolio.weight));
  setText("walk-forward", payload.time_series_validation.walk_forward_validation.windows);
  setText("oos-test", payload.time_series_validation.out_of_sample_testing ? "Enabled" : "Needs more rows");
  setText("leakage-guard", "Active");
  if (payload.research_hypothesis) {
    const hypothesis = payload.research_hypothesis;
    setText("hypothesis-question", hypothesis.primary_hypothesis.research_question);
    setText("hypothesis-null", hypothesis.primary_hypothesis.null_hypothesis);
    setText("hypothesis-alt", hypothesis.primary_hypothesis.alternative_hypothesis);
    setText("hypothesis-signal", `${hypothesis.primary_hypothesis.current_prediction} (${pct(hypothesis.primary_hypothesis.probability_up)} up)`);
    renderSignificanceTests(hypothesis.statistical_tests);
    setText("significance-conclusion", hypothesis.significance_summary.overall_conclusion);
  }
  renderRoadmap(payload.roadmap_completion);
  setText("research-paper", payload.research_documentation);
  if (latestPayload) renderModelPerformance(latestPayload.ml_prediction, payload);
}

async function loadResearch() {
  const params = new URLSearchParams({ symbol: symbolInput.value.trim() || "AAPL", period: periodInput.value });
  const response = await fetch(`/api/research?${params.toString()}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Research layer failed");
  renderResearch(payload);
}

function renderMonitor(monitor, risk) {
  if (!monitor) return;
  const pnl = document.querySelector("#monitor-pnl");
  pnl.textContent = signedMoney(monitor.pnl);
  pnl.className = tone(monitor.pnl);
  setText("monitor-status", monitor.status);
  setText("monitor-side", risk.side);
  setText("risk-qty", risk.suggested_quantity);
  setText("monitor-entry", formatMoney(monitor.entry_price));
  setText("monitor-percent", `${monitor.pnl_percent >= 0 ? "+" : ""}${monitor.pnl_percent}%`);
  setText("monitor-updated", `${monitor.symbol} entry from ${monitor.entry_time}. Updated ${monitor.updated_at}.`);
}

async function loadPrediction(recordEntry = true) {
  const params = new URLSearchParams({ symbol: symbolInput.value.trim() || "AAPL", period: periodInput.value, record: recordEntry ? "1" : "0" });
  statusText.textContent = "Running Vantage Point Research Engine...";
  try {
    const response = await fetch(`/api/predict?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Prediction failed");
    const direction = document.querySelector("#direction");
    direction.textContent = payload.direction;
    direction.className = payload.direction === "Bullish" ? "positive" : payload.direction === "Bearish" ? "negative" : "neutral";
    setText("symbol-label", payload.symbol);
    setText("period-label", payload.period.toUpperCase());
    setText("confidence", payload.confidence);
    setText("last-price", formatMoney(payload.last_price));
    setText("target-peak", formatMoney(payload.target_peak));
    setText("stop-loss", formatMoney(payload.stop_loss));
    setText("risk-reward", payload.risk_reward);
    setText("ml-probability", pct(payload.ml_prediction.probability_up));
    setText("bayes-probability", pct(payload.bayesian_probability_up));
    setText("weighted-score", payload.weighted_score);
    setText("atr", payload.atr);
    setText("forecast-trend", `${payload.forecast.trend} trend`);
    setText("forecast-price", `Projected ${formatMoney(payload.forecast.expected_price)} in ${payload.forecast.horizon_bars} bars`);
    renderSourceStatus(payload);
    const bar = document.querySelector("#confidence-bar");
    bar.style.width = `${payload.confidence}%`;
    bar.style.background = payload.direction === "Bearish" ? "#f23645" : payload.direction === "Sideways" ? "#f5b942" : "#22ab94";
    renderChart(payload);
    renderStrategies(payload.backtests);
    renderPerformanceCharts(payload.best_strategy);
    renderMonitor(payload.monitor, payload.risk_management);
    await loadResearch();
    statusText.textContent = `${payload.source_kind === "live" ? "Live data active" : "Demo data active"}: ${payload.source}. Best strategy: ${payload.best_strategy.strategy}. Best model: ${payload.ml_prediction.best_model}.`;
  } catch (error) {
    statusText.textContent = error.message;
  }
}

async function refreshMonitor() {
  try {
    const response = await fetch("/api/monitor");
    const monitor = await response.json();
    if (response.ok && latestPayload) renderMonitor(monitor, latestPayload.risk_management);
  } catch (error) {}
}

form.addEventListener("submit", (event) => { event.preventDefault(); loadPrediction(); });
periodInput.addEventListener("change", () => loadPrediction());
document.querySelectorAll("[data-chart-type]").forEach((button) => {
  button.addEventListener("click", () => {
    chartType = button.dataset.chartType;
    document.querySelectorAll("[data-chart-type]").forEach((item) => item.classList.toggle("active", item === button));
    if (latestPayload) renderChart(latestPayload);
  });
});
document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-tab]").forEach((item) => item.classList.toggle("active", item === button));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === button.dataset.tab));
    if (latestPayload) renderPerformanceCharts(latestPayload.best_strategy);
  });
});
canvas.addEventListener("mousemove", (event) => {
  if (!latestPayload) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const index = Math.max(0, Math.min(latestPayload.candles.length - 1, Math.round((x / rect.width) * (latestPayload.candles.length - 1))));
  crosshair.hidden = false;
  crosshair.style.setProperty("--x", `${x}px`);
  crosshair.style.setProperty("--y", `${y}px`);
  crosshairLabel.textContent = formatMoney(latestPayload.candles[index].close);
});
canvas.addEventListener("mouseleave", () => { crosshair.hidden = true; });
window.addEventListener("resize", () => { if (latestPayload) { renderChart(latestPayload); renderPerformanceCharts(latestPayload.best_strategy); } });
window.addEventListener("load", () => { loadPrediction(); setInterval(refreshMonitor, 15000); });
