import os
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
from dotenv import load_dotenv

from .parsers import parse_khmer_date

ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=ENV_PATH)
logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent.parent / 'data' / 'cache_daily_registrations.json'
BASE_URL = os.getenv('CAMIE_DB_URL', 'https://ie-api-report.digitaleconomy.gov.kh').rstrip('/')
API_TOKEN = os.getenv('CAMIE_API_TOKEN', '')
START_MONTH = os.getenv('START_MONTH', '2024-01')


def get_month_range(start_month: str, end_month: Optional[str] = None) -> List[str]:
    """Generates a list of 'YYYY-MM' strings between start and end month."""
    start_dt = datetime.strptime(start_month, '%Y-%m')
    if not end_month:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_month, '%Y-%m')

    months = []
    current = start_dt
    while (current.year, current.month) <= (end_dt.year, end_dt.month):
        months.append(current.strftime('%Y-%m'))
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = datetime(year, month, 1)
    return months


def fetch_month_data(month_str: str) -> List[Dict[str, Any]]:
    """Fetches daily registration data for a specific month from the API."""
    url = f'{BASE_URL}/api/v1/internal/report-dashboard/get-dashboard-daily-registration'
    params = {'month': month_str}
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Accept': 'application/json'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=25)
        response.raise_for_status()
        res_json = response.json()
        return res_json.get('data', {}).get('days', [])
    except Exception as e:
        logger.warning(f'Failed to fetch data for month {month_str}: {e}')
        return []


def load_cached_data() -> Dict[str, int]:
    """Loads previously saved {date_str: total} from local cache."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f'Could not read cache: {e}')
    return {}


def save_cached_data(records: Dict[str, int]) -> None:
    """Saves {date_str: total} map to cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, sort_keys=True)


def load_daily_registrations(*args, force_refresh_all: bool = False, **kwargs) -> pd.DataFrame:
    """
    Ingests live data from 2024 to present:
    - Loads cached past months.
    - Fetches missing months and the ongoing current month.
    - Parses Khmer dates and extracts 'total.total'.
    - Returns a continuous DataFrame with columns ['ds', 'y'].
    """
    current_month_str = datetime.now().strftime('%Y-%m')
    cached_records = {} if force_refresh_all else load_cached_data()
    all_months = get_month_range(START_MONTH, current_month_str)

    new_data_fetched = False

    for m in all_months:
        is_current_or_prev_month = (m >= current_month_str)
        month_already_cached = any(k.startswith(m) for k in cached_records)

        if month_already_cached and not is_current_or_prev_month and not force_refresh_all:
            continue

        logger.info(f'Fetching live data for month {m}...')
        days = fetch_month_data(m)
        if not days:
            continue

        for item in days:
            raw_date = item.get('date', '')
            iso_date = parse_khmer_date(raw_date)
            total_val = item.get('total', {}).get('total', 0)
            if iso_date:
                cached_records[iso_date] = int(total_val)
                new_data_fetched = True

    if new_data_fetched or not CACHE_FILE.exists():
        save_cached_data(cached_records)

    if not cached_records:
        raise ValueError('No registration records found from API or cache.')

    df = pd.DataFrame(
        [{'ds': pd.to_datetime(d), 'y': float(val)} for d, val in cached_records.items()]
    )
    df = df.sort_values('ds').drop_duplicates(subset=['ds'])

    # Only keep data up to today to prevent future placeholder 0s from distorting the forecast
    today_ts = pd.Timestamp(datetime.now().date())
    df = df[df['ds'] <= today_ts]

    full_date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
    df = df.set_index('ds').reindex(full_date_range, fill_value=0.0).reset_index()
    df.rename(columns={'index': 'ds'}, inplace=True)
    df['y'] = df['y'].astype(float)

    return df
