import pytest
from fastapi.testclient import TestClient

from camie_forecast import pipeline
from camie_forecast.api import app
import pandas as pd
import numpy as np

def generate_demo_series():
    ds = pd.date_range("2025-01-01", periods=90, freq="D")
    return pd.DataFrame({"ds": ds, "y": np.random.uniform(50, 150, 90)})

c = TestClient(app)


@pytest.mark.parametrize("g", ["daily", "monthly", "yearly"])
def test_forecast_endpoint(g):
    r = c.get("/api/registrations/forecast", params={"granularity": g, "horizon": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] in ("timesfm", "ets")
    fut = [x for x in body["data"] if x["type"] == "forecast"]
    assert 0 < len(fut) <= 5
    assert all(0 <= x["lower"] <= x["forecast"] <= x["upper"] for x in fut)


def test_health():
    r = c.get("/api/registrations/forecast/health")
    assert r.status_code == 200 and "fallback_used" in r.json()


def test_fallback_to_ets_when_timesfm_fails(monkeypatch):
    class Broken:
        name = "timesfm"
        def fit(self, h): raise RuntimeError("model download failed")

    monkeypatch.setattr(pipeline, "PRIMARY", Broken)
    df = generate_demo_series()
    fc, used, reason = pipeline.forecast_with_fallback(df, 60)
    assert used == "ets" and "model download failed" in reason
    assert len(fc) == 60 and (fc.yhat >= 0).all()


def test_fallback_on_implausible_forecast(monkeypatch):
    import pandas as pd

    class Zero:
        name = "timesfm"
        def fit(self, h): self.h = h; return self
        def predict(self, n):
            ds = pd.date_range(self.h.ds.max() + pd.Timedelta(days=1), periods=n)
            return pd.DataFrame({"ds": ds, "yhat": 0.0, "yhat_lower": 0.0, "yhat_upper": 0.0})

    monkeypatch.setattr(pipeline, "PRIMARY", Zero)
    _, used, reason = pipeline.forecast_with_fallback(generate_demo_series(), 30)
    assert used == "ets" and "collapsed" in reason
    