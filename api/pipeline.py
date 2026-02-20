"""
WiscoNet disease-risk pipeline.

This module contains the top-level orchestration logic:
  1. Load the station list (cached).
  2. Fetch hourly measurement data for each station (batched, cached).
  3. Optionally fetch biomass data for each station (batched).
  4. Merge station metadata.
  5. Compute disease-risk scores in parallel worker processes.
  6. Return a tidy DataFrame.

Public surface
--------------
``retrieve(...)`` – synchronous entry point used by callers that do not
                    manage an event loop themselves.
"""

import asyncio
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd

from api.config.constants import BIOMASS_COLS, FINAL_COLUMNS, STATION_META_COLS
from api.services.http_client import create_session
from api.services.risk_processor import chunk_dataframe, compute_risks
from api.services.wisconet_service import (
    fetch_biomass_in_batches,
    fetch_measurements_in_batches,
    load_stations,
)

logger = logging.getLogger(__name__)


# ── Core async pipeline ────────────────────────────────────────────────────────

async def _run_pipeline(
    input_date: str,
    input_station_id: str | None = None,
    days: int = 1,
    planting_date: str | None = None,
    termination_date: str | None = None,
) -> pd.DataFrame | None:
    """
    Async implementation of the WiscoNet risk pipeline.

    Args:
        input_date:        Reference date for rolling averages (YYYY-MM-DD).
        input_station_id:  Restrict output to a single station (optional).
        days:              How many recent days of risk values to return.
        planting_date:     Cover-crop planting date for biomass (optional).
        termination_date:  Cover-crop termination date for biomass (optional).

    Returns:
        Tidy DataFrame or ``None`` on failure.
    """
    try:
        async with create_session() as session:

            # 1) Station list ─────────────────────────────────────────────────
            stations = await load_stations(session, input_date)
            if stations is None or stations.empty:
                logger.warning("No stations available.")
                return None

            if input_station_id:
                stations = stations[stations["station_id"] == input_station_id]
                if stations.empty:
                    logger.warning("Station %s not found.", input_station_id)
                    return None

            station_ids = stations["station_id"].tolist()

            # 2) Measurements + optional biomass (concurrent) ─────────────────
            run_biomass = bool(planting_date and termination_date)

            if run_biomass:
                measurements_list, biomass_df = await asyncio.gather(
                    fetch_measurements_in_batches(session, station_ids, input_date, days),
                    fetch_biomass_in_batches(session, station_ids, planting_date, termination_date),
                )
            else:
                measurements_list = await fetch_measurements_in_batches(session, station_ids, input_date, days)
                biomass_df = None

            if not measurements_list:
                logger.warning("No measurement data returned.")
                return None

            meas_df = pd.concat(measurements_list, ignore_index=True)

            # 3) Merge station metadata ────────────────────────────────────────
            meta   = stations[STATION_META_COLS]
            merged = pd.merge(meta, meas_df, on="station_id", how="inner")
            if merged.empty:
                logger.warning("Metadata merge produced an empty DataFrame.")
                return None

            # 4) Merge biomass ─────────────────────────────────────────────────
            if biomass_df is not None and not biomass_df.empty:
                merged = pd.merge(merged, biomass_df, on="station_id", how="left")
                logger.info("Biomass merged for %d station(s).", biomass_df["station_id"].nunique())
            else:
                for col in BIOMASS_COLS:
                    merged[col] = np.nan

            # 5) Parallel risk computation ──────────────────────────────────
            chunks = chunk_dataframe(merged, os.cpu_count() or 1)
            with ProcessPoolExecutor() as executor:
                futures   = [executor.submit(compute_risks, c) for c in chunks]
                processed = [f.result() for f in as_completed(futures)]
            final = pd.concat(processed, ignore_index=True)

            # 6) Finalise dates ────────────────────────────────────────────────
            final["date"]             = pd.to_datetime(final["date"])
            final["forecasting_date"] = (final["date"] + timedelta(days=1)).dt.strftime("%Y-%m-%d")

            available = [c for c in FINAL_COLUMNS if c in final.columns]
            return final[available]

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        traceback.print_exc()
        return None


# ── Public synchronous entry point ─────────────────────────────────────────────

def retrieve(
    input_date: str,
    input_station_id: str | None = None,
    days: int = 1,
    planting_date: str | None = None,
    termination_date: str | None = None,
) -> pd.DataFrame | None:
    """
    Synchronous wrapper around the WiscoNet async pipeline.

    Args:
        input_date:        Reference date for disease-risk rolling averages
                           (YYYY-MM-DD).
        input_station_id:  Limit results to a single station (optional).
        days:              Number of recent risk-score rows to return.
        planting_date:     Cover-crop planting date for biomass estimation
                           (YYYY-MM-DD, optional).
        termination_date:  Cover-crop termination date for biomass estimation
                           (YYYY-MM-DD, optional).

    Returns:
        pd.DataFrame with disease-risk columns and, when crop dates are
        supplied, biomass columns (cgdd_60d_ap, rain_60d_ap, cgdd_60d_bt,
        biomass_lb_acre, biomass_color, biomass_message).
        Returns ``None`` on failure.

    Example::

        df = retrieve("2024-08-01", days=7)
        df = retrieve("2024-08-01", planting_date="2024-04-15", termination_date="2024-07-01")
    """
    return asyncio.run(
        _run_pipeline(input_date, input_station_id, days, planting_date, termination_date)
    )