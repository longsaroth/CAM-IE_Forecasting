"""Nightly job: python run_job.py  (or schedule via cron / Airflow / K8s CronJob)."""
import logging
import shutil
import time
from pathlib import Path

from camie_forecast.api import CACHE, _load_history
from camie_forecast.pipeline import run_forecast, save

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    t0 = time.time()
    result = run_forecast(_load_history())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    save(result, CACHE)
    web_dest = Path(__file__).resolve().parent / "web" / "forecast.json"
    if web_dest.parent.exists():
        shutil.copy(CACHE, web_dest)
    print("model used:", result["model"], "| fallback:", result["fallback_reason"] or "no")
    print("backtest:", result["backtest"])
    print(f"done in {time.time() - t0:.1f}s -> {CACHE}")
