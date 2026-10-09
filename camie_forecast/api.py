"""FastAPI endpoints for the CAM-IE Registration Forecast feature.

GET  /api/registrations/forecast?granularity=daily|monthly|yearly&horizon=12
POST /api/registrations/forecast/run      (admin / nightly cron: regenerate)
GET  /api/registrations/forecast/health   (which model served the last run)
"""
# pyrefly: ignore [invalid-syntax]
from camie_forecast.data import load_daily_registrations
# pyrefly: ignore [invalid-syntax]
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .pipeline import run_forecast, save

CACHE = Path(os.getenv("CAMIE_FORECAST_CACHE", str(Path(__file__).resolve().parent.parent / "data" / "forecast.json")))
DB_URL = os.getenv("CAMIE_DB_URL")  # e.g. postgresql+psycopg://user:pass@host/camie

META = ("forecast_run_id", "generated_at", "last_actual_date", "interval", "model",
        "primary_model", "fallback_used", "fallback_reason", "backtest")

app = FastAPI(title="CA M-IE Registration Forecast")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"])


def _load_history():
    return load_daily_registrations()


def _retrain() -> None:
    save(run_forecast(_load_history()), CACHE)


def _run() -> dict:
    if not CACHE.exists():
        raise HTTPException(503, "Forecast not generated yet. POST /api/registrations/forecast/run")
    return json.loads(CACHE.read_text())


@app.get("/api/registrations/forecast")
def get_forecast(
    granularity: Literal["daily", "monthly", "yearly"] = "monthly",
    horizon: int | None = Query(None, ge=1, le=1095, description="number of future periods"),
):
    run = _run()
    rows = run["series"][granularity]
    if horizon:
        rows = [r for r in rows if r["type"] != "forecast"] + [r for r in rows if r["type"] == "forecast"][:horizon]
    return {k: run[k] for k in META} | {"granularity": granularity, "data": rows}


@app.get("/api/registrations/forecast/health")
def health():
    run = _run()
    return {k: run[k] for k in ("generated_at", "model", "fallback_used", "fallback_reason")}


@app.post("/api/registrations/forecast/run", status_code=202)
def trigger_run(tasks: BackgroundTasks):
    tasks.add_task(_retrain)
    return {"status": "accepted"}