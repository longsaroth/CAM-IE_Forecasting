"""CAM-IE forecasting models: TimesFM 2.5 (primary) and Holt-Winters ETS (automatic backup).

Both share one interface.

Every model implements:
    fit(history: DataFrame[ds, y]) -> self
    predict(horizon: int) -> DataFrame[ds, yhat, yhat_lower, yhat_upper]

Intervals are 80% prediction intervals (10th-90th percentile).
"""
from __future__ import annotations

import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=ConvergenceWarning)

Z80 = 1.2816


def _future_index(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")


def _clip(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("yhat", "yhat_lower", "yhat_upper"):
        df[c] = df[c].clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Backup: Holt-Winters ETS (damped trend + weekly seasonality, yearly via Fourier
#    de-seasonalisation) — fast, robust statistical baseline
# ---------------------------------------------------------------------------
class ETSForecaster:
    name = "ets"

    def fit(self, history: pd.DataFrame):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self.history = history.reset_index(drop=True)
        y = self.history["y"].to_numpy()
        doy = pd.DatetimeIndex(self.history["ds"]).dayofyear.to_numpy()
        # yearly seasonal index estimated with 2 Fourier terms on log scale
        self.F = lambda d: np.column_stack([np.ones_like(d, dtype=float)] + [
            f(2 * np.pi * k * d / 365.25) for k in (1, 2) for f in (np.sin, np.cos)])
        logy = np.log1p(y)
        t = np.arange(len(y))
        A = np.column_stack([self.F(doy), t])
        coef, *_ = np.linalg.lstsq(A, logy, rcond=None)
        self.ycoef = coef[1:5]
        yearly = np.exp(self.F(doy)[:, 1:] @ self.ycoef)
        adj = y / yearly
        self.m = ExponentialSmoothing(adj + 1, trend="add", damped_trend=False,
                                      seasonal="mul", seasonal_periods=7,
                                      initialization_method="estimated").fit()
        self.resid_sd = np.std(self.m.resid[-180:])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        future = _future_index(self.history, horizon)
        yearly = np.exp(self.F(future.dayofyear.to_numpy())[:, 1:] @ self.ycoef)
        base = np.asarray(self.m.forecast(horizon)) - 1
        mid = base * yearly
        sd = self.resid_sd * yearly * np.sqrt(1 + np.arange(horizon) / 60)
        out = pd.DataFrame({"ds": future, "yhat": mid,
                            "yhat_lower": mid - Z80 * sd, "yhat_upper": mid + Z80 * sd})
        return _clip(out)




# ---------------------------------------------------------------------------
# Primary: TimesFM 2.5 (Google, 200M-param foundation model, Apache-2.0 weights)
#    Zero-shot: no training, "fit" only stores history. Model is loaded once
#    per process and reused across backtest folds / forecast runs.
# ---------------------------------------------------------------------------
class TimesFMForecaster:
    name = "timesfm"
    CHECKPOINT = "google/timesfm-2.5-200m-pytorch"
    MAX_CONTEXT = 2048   # ~5.6 years of daily data (multiple of 32-step patches)
    MAX_HORIZON = 1024   # continuous quantile head supports up to ~1k steps
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            import timesfm
            import torch

            torch.set_float32_matmul_precision("high")
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cls.CHECKPOINT)
            m.compile(timesfm.ForecastConfig(
                max_context=cls.MAX_CONTEXT, max_horizon=cls.MAX_HORIZON,
                normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True,  # counts >= 0
                fix_quantile_crossing=True))
            cls._model = m
        return cls._model

    def fit(self, history: pd.DataFrame):
        self.history = history.reset_index(drop=True)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        model = self._load()
        ctx = self.history["y"].to_numpy(dtype=np.float32)
        mids, los, his = [], [], []
        # Horizons beyond MAX_HORIZON are produced in chunks, feeding the
        # median back as context (only needed when run early in a year).
        while sum(len(m) for m in mids) < horizon:
            step = min(self.MAX_HORIZON, horizon - sum(len(m) for m in mids))
            point, quant = model.forecast(horizon=step, inputs=[ctx[-self.MAX_CONTEXT:]])
            mids.append(point[0]); los.append(quant[0][:, 1]); his.append(quant[0][:, 9])
            ctx = np.concatenate([ctx, point[0].astype(np.float32)])
        mid, lo, hi = (np.concatenate(a) for a in (mids, los, his))
        out = pd.DataFrame({"ds": _future_index(self.history, horizon), "yhat": mid,
                            "yhat_lower": np.minimum(lo, mid), "yhat_upper": np.maximum(hi, mid)})
        return _clip(out)

PRIMARY, BACKUP = TimesFMForecaster, ETSForecaster
