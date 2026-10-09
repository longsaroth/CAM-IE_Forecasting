# CAM-IE Registration Forecast

Forecasts future registrations and serves them at daily, monthly and yearly granularity for the CAM-IE graph.

**Model:** TimesFM 2.5 (Google, 200M-parameter time-series foundation model, zero-shot).
**Automatic backup:** Holt-Winters ETS — used only if TimesFM fails.

## How it works
1. **Aggregate** raw registrations to a daily series (missing days filled with 0) – `data.py`.
2. **Forecast** daily to the end of year +2 with TimesFM 2.5 – `pipeline.forecast_with_fallback`.
3. **Fallback:** if TimesFM can't download/load, raises, or returns an implausible forecast
   (wrong length, NaN/inf, mean < 5% or > 20× the last 90 days), the job logs a warning and uses ETS.
   The run records `model`, `fallback_used` and `fallback_reason`; the dashboard shows a badge and banner.
4. **Accuracy check:** 3 rolling folds × 30 days with the model actually used (WAPE, MAE, 80% coverage).
5. **Aggregate** daily forecasts to months/years (bottom-up, so all views add up). Intervals combined assuming
   correlated daily errors (rho = 0.3). The current month/year is `partial` = actual-to-date + forecast remainder.

## Run
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export CAMIE_DB_URL=postgresql+psycopg://user:pass@host/camie   # omit to use synthetic demo data
python run_job.py                      # nightly job -> data/forecast.json
uvicorn camie_forecast.api:app --port 8000
curl "localhost:8000/api/registrations/forecast?granularity=monthly&horizon=12"
curl "localhost:8000/api/registrations/forecast/health"     # which model served the last run
pytest -q                              # includes forced-fallback tests
```
Adjust `DAILY_SQL` in `data.py` to the CAM-IE `registrations` table.

## TimesFM notes
- Checkpoint `google/timesfm-2.5-200m-pytorch` (Apache-2.0, OK for self-hosted production).
  TimesFM 3.0 weights are non-commercial; use 3.0 only via BigQuery ML for production.
- Loaded once per process (~60 s cold on CPU, ~1 GB RAM), then ~0.5 s per forecast.
  Bake the Hugging Face cache (`HF_HOME`) into the container image so a download outage doesn't trigger the fallback.
- Context: last 2,048 days. Horizon: up to 1,024 days per call; longer horizons are chunked automatically.
- Alert on `fallback_used=true` from `/health` — the graph keeps working, but someone should check why.

## Response shape
```json
{"granularity":"monthly","model":"timesfm","fallback_used":false,"fallback_reason":null,
 "last_actual_date":"2026-10-04","interval":"80%",
 "data":[{"date":"2026-09","type":"actual","actual":4736,"forecast":null,"lower":null,"upper":null},
         {"date":"2026-10","type":"partial","actual":524,"forecast":4529,"lower":4079,"upper":4991},
         {"date":"2026-11","type":"forecast","actual":null,"forecast":4788,"lower":4254,"upper":5333}]}
```
