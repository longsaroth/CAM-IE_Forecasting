"""Forecast job: TimesFM 2.5 with automatic Holt-Winters ETS fallback,
then daily -> monthly -> yearly aggregation."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .models import BACKUP, PRIMARY

log = logging.getLogger("camie_forecast")
ERROR_CORR = 0.3  # assumed correlation of daily errors when summing intervals


# ---------------------------------------------------------------- metrics
def wape(y, yhat) -> float:
    return float(np.sum(np.abs(y - yhat)) / max(np.sum(np.abs(y)), 1e-9))


def coverage(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------------------------------------------------------------- forecast with fallback
def validate(fc: pd.DataFrame, history: pd.DataFrame, horizon: int) -> None:
    """Reject forecasts that are broken or implausible, so the backup takes over."""
    if len(fc) != horizon:
        raise ValueError(f"expected {horizon} rows, got {len(fc)}")
    vals = fc[["yhat", "yhat_lower", "yhat_upper"]].to_numpy()
    if not np.isfinite(vals).all():
        raise ValueError("forecast contains NaN/inf")
    recent = history["y"].tail(90).mean()
    if recent > 0 and fc["yhat"].head(90).mean() < 0.05 * recent:
        raise ValueError("forecast collapsed to ~0 vs recent history")
    if recent > 0 and fc["yhat"].head(90).mean() > 20 * recent:
        raise ValueError("forecast exploded vs recent history")


def forecast_with_fallback(history: pd.DataFrame, horizon: int):
    """Returns (forecast_df, model_name, fallback_reason or None)."""
    try:
        fc = PRIMARY().fit(history).predict(horizon)
        validate(fc, history, horizon)
        return fc, PRIMARY.name, None
    except Exception as exc:  # model download/load/inference/validation failure
        reason = f"{type(exc).__name__}: {exc}"
        log.warning("TimesFM failed (%s) - falling back to ETS", reason)
        fc = BACKUP().fit(history).predict(horizon)
        return fc, BACKUP.name, reason


def backtest(df: pd.DataFrame, folds: int = 3, horizon: int = 30) -> dict:
    """Rolling-origin accuracy check of the model actually used in production."""
    rows = []
    for k in range(folds, 0, -1):
        cut = len(df) - k * horizon
        train, test = df.iloc[:cut], df.iloc[cut:cut + horizon]
        fc, used, _ = forecast_with_fallback(train, horizon)
        rows.append(dict(model=used, wape=wape(test.y.values, fc.yhat.values),
                         mae=float(np.mean(np.abs(test.y.values - fc.yhat.values))),
                         coverage80=coverage(test.y.values, fc.yhat_lower.values, fc.yhat_upper.values)))
    out = pd.DataFrame(rows)
    return dict(folds=folds, horizon_days=horizon,
                **out[["wape", "mae", "coverage80"]].mean().round(4).to_dict())


# ---------------------------------------------------------------- aggregation
def aggregate(daily_fc: pd.DataFrame, actual: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Sum daily values to month ('MS') or year ('YS'). Intervals are combined
    assuming correlated daily errors (rho = ERROR_CORR)."""
    fc = daily_fc.copy()
    fc["s_lo"] = fc["yhat"] - fc["yhat_lower"]
    fc["s_hi"] = fc["yhat_upper"] - fc["yhat"]

    def _agg_hw(s: pd.Series) -> float:
        return float(np.sqrt((1 - ERROR_CORR) * np.sum(s ** 2) + ERROR_CORR * np.sum(s) ** 2))

    g = fc.groupby(pd.Grouper(key="ds", freq=freq))
    f = pd.DataFrame({"forecast": g["yhat"].sum(), "hw_lo": g["s_lo"].apply(_agg_hw),
                      "hw_hi": g["s_hi"].apply(_agg_hw)})
    a = actual.groupby(pd.Grouper(key="ds", freq=freq))["y"].sum().rename("actual")
    out = pd.concat([a, f], axis=1).fillna({"actual": 0, "forecast": 0, "hw_lo": 0, "hw_hi": 0})
    out["total"] = out["actual"] + out["forecast"]          # partial period = actual + remainder
    out["lower"] = (out["total"] - out["hw_lo"]).clip(lower=out["actual"])
    out["upper"] = out["total"] + out["hw_hi"]
    out["type"] = np.select([out["forecast"] == 0, out["actual"] == 0], ["actual", "forecast"], "partial")
    return out.reset_index(names="ds")


def _records(df: pd.DataFrame, fmt: str) -> list[dict]:
    return [dict(date=r.ds.strftime(fmt), type=r.type,
                 actual=None if r.type == "forecast" else round(r.actual),
                 forecast=None if r.type == "actual" else round(r.total),
                 lower=None if r.type == "actual" else round(r.lower),
                 upper=None if r.type == "actual" else round(r.upper)) for r in df.itertuples()]


# ---------------------------------------------------------------- job
def run_forecast(df: pd.DataFrame, years_ahead: int = 2, do_backtest: bool = True) -> dict:
    last = df["ds"].max()
    horizon = (pd.Timestamp(year=last.year + years_ahead, month=12, day=31) - last).days

    fc, used, reason = forecast_with_fallback(df, horizon)
    hist_daily = df[df["ds"] > last - pd.Timedelta(days=365)]
    daily = pd.concat([
        hist_daily.assign(type="actual", actual=hist_daily.y, total=np.nan, lower=np.nan, upper=np.nan),
        fc.assign(type="forecast", actual=np.nan, total=fc.yhat, lower=fc.yhat_lower, upper=fc.yhat_upper),
    ])
    daily = daily[daily["ds"] <= last + pd.Timedelta(days=180)]

    return dict(
        forecast_run_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        last_actual_date=str(last.date()),
        metric="registrations",
        interval="80%",
        model=used,                      # "timesfm" normally, "ets" if the backup was used
        primary_model=PRIMARY.name,
        fallback_used=reason is not None,
        fallback_reason=reason,
        backtest=backtest(df) if do_backtest else None,
        series=dict(
            daily=_records(daily, "%Y-%m-%d"),
            monthly=_records(aggregate(fc, df[df.ds >= pd.Timestamp(last.year - 2, 1, 1)], "MS"), "%Y-%m"),
            yearly=_records(aggregate(fc, df, "YS"), "%Y"),
        ),
    )


def save(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, separators=(",", ":")))
