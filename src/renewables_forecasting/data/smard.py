import requests
import json
import string
import time
import pandas as pd
import xarray as xr

from datetime import datetime, timedelta
from pathlib import Path


def _check_template(template: str, required_keys: list):
    keys = [key for _, key, _, _ in string.Formatter().parse(template) if key is not None]
    assert all(k in keys for k in required_keys), f"Template missing required keys: {required_keys}"


def _resolve_smard_timestamps(url: str, start: datetime, end: datetime):

    response = requests.get(url=url)
    response.raise_for_status()
    timestamps_dict = response.json()
    timestamps_all = timestamps_dict["timestamps"]

    # Timestamps mark the start of weekly chunks.
    # Start filtering one week prior incorporate all timestamps for days of interest
    start_filter = start - timedelta(weeks=1)
    start_stamp = start_filter.timestamp() * 1000  # In millisecs
    end_stamp = end.timestamp() * 1000
    stamps = [ts for ts in timestamps_all if start_stamp <= ts <= end_stamp]
    return stamps


def download_smard_generation(
        gen_url_template: str,
        timestamps_url: str,
        out_path: Path,
        start: datetime,
        end: datetime,
        read_timeout: int,
        connect_timeout: int,
        time_col: str = "time",
        timestamp_ms_col: str = "timestamp_ms",
        value_col: str = "generation_mwh",
) -> None:
    """
    Downloads SMARD generation data for a requested period and saves it as a
    clean CSV with millisecond timestamps, converted datetimes, and generation
    values.

    SMARD serves data in weekly chunks, each identified by a Unix millisecond
    timestamp marking the start of the chunk. To ensure the first chunk
    covering the requested start date is included, the timestamp index is
    fetched starting one week before start. All entries outside the exact
    requested period [start, end) are trimmed before saving, so the output
    contains only the requested data.

    Timestamps are stored both as the original Unix milliseconds (for
    reproducibility and auditability) and as timezone-naive German local time
    (CET/CEST, Europe/Berlin) for direct alignment with ERA5 and capacity
    grid data in the pipeline. The timezone is stripped for consistency with
    ERA5 data stored timezone-naive after convert_era5_utc_to_german_time(),
    but CET/CEST is the implied timezone throughout.

    Parameters
    ----------
    gen_url_template:
        URL template for fetching generation data for a single chunk. Must
        contain a {timestamp} placeholder, e.g.
        'https://www.smard.de/app/chart_data/4068/DE/4068_DE_hour_{timestamp}.json'
    timestamps_url:
        URL to fetch the index of available chunk timestamps for the requested
        generation series, e.g.
        'https://www.smard.de/app/chart_data/4068/DE/index_hour.json'
    out_path:
        Path to write the output CSV file.
    start:
        Start of the requested period, inclusive. Must be a timezone-naive
        datetime in German local time, e.g. datetime(2015, 1, 1).
    end:
        End of the requested period, exclusive. Set to the first moment after
        the last desired entry, e.g. datetime(2026, 1, 1) to include all of
        2025-12-31.
    read_timeout:
        Read timeout in seconds for each API request.
    connect_timeout:
        Connection timeout in seconds for each API request.
    time_col:
        Name for the converted datetime column. Default is 'time'.
    timestamp_ms_col:
        Name for the raw Unix millisecond timestamp column. Default is
        'timestamp_ms'. Retained for reproducibility so the timezone
        conversion can always be verified or re-derived.
    value_col:
        Name for the generation column. Default is 'generation_mwh'.
    """

    _check_template(gen_url_template, ["timestamp"])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch chunk timestamps, going one week back from start to ensure the
    # first chunk covering the requested start date is included
    timestamps = _resolve_smard_timestamps(timestamps_url, start, end)

    series = []

    for timestamp in timestamps:
        url = gen_url_template.format(timestamp=timestamp)
        response = requests.get(url, timeout=(connect_timeout, read_timeout))
        response.raise_for_status()
        series.extend(response.json()["series"])
        time.sleep(0.5)  # At least half a second between calls

    # Trim to exact requested period [start, end) — necessary because the
    # one-week lookback causes the first chunk to contain entries before start,
    # and end is typically set one day past the last desired date to ensure
    # the final entries are included regardless of chunk boundaries
    start_ms = start.timestamp() * 1000
    end_ms = end.timestamp() * 1000
    series = [entry for entry in series if start_ms <= entry[0] < end_ms]

    # Build DataFrame with raw millisecond timestamps and converted datetimes
    df = pd.DataFrame(series, columns=[timestamp_ms_col, value_col])

    # Convert Unix milliseconds to timezone-naive German local time.
    # SMARD timestamps are in CET/CEST — convert via UTC then strip timezone
    # for consistency with ERA5 data in the pipeline.
    df[time_col] = (
        pd.to_datetime(df[timestamp_ms_col], unit="ms", utc=True)
        .dt.tz_convert("Europe/Berlin")
        .dt.tz_localize(None)
    )

    # To be safe for potential in-chunk non-chronological order:
    # Reorder columns and sort chronologically, to be safe
    df = df[[timestamp_ms_col, time_col, value_col]]
    df = df.sort_values(time_col).reset_index(drop=True)

    print(f"  Downloaded {len(df):,} hourly entries")
    print(f"  First: {df[time_col].iloc[0]}")
    print(f"  Last:  {df[time_col].iloc[-1]}")
    print(f"  Total generation: {df[value_col].sum():,.1f} MWh")

    df.to_csv(out_path, index=False)
    print(f"  Saved to: {out_path}")


def apply_daylight_mask_to_generation(
        mask_path: Path,
        generation_csv_path: Path,
        out_csv_path: Path,
        time_col: str = "time",
) -> None:
    """
    Applies a precomputed daylight mask to SMARD generation data stored as a
    CSV, keeping only rows within the daylight window and discarding night hours.

    Both the generation CSV and the daylight mask must be in German local time
    (CET/CEST, Europe/Berlin) for the mask to align correctly. The generation
    CSV time column should have been produced by parse_smard_gen_json_to_csv()
    which converts SMARD timestamps to timezone-naive German local time. The
    daylight mask is built from ERA5 ssrd data also converted to German local
    time by convert_era5_utc_to_german_time().

    Note: this function is intended for solar generation only. Wind generation
    operates around the clock and should not have night hours masked out.

    Parameters
    ----------
    mask_path:
        Path to the daylight mask NetCDF file produced by build_daylight_mask().
    generation_csv_path:
        Path to the SMARD generation CSV with a datetime time column, produced
        by parse_smard_gen_json_to_csv().
    out_csv_path:
        Path to write the filtered generation CSV.
    time_col:
        Name of the datetime column in the generation CSV. Default is 'time'.
    """

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading daylight mask from {mask_path} ...")
    mask = xr.open_dataset(mask_path)["daylight_mask"]
    day_times = set(pd.DatetimeIndex(mask.time.values[mask.values]))

    print(f"Loading generation data from {generation_csv_path} ...")
    df = pd.read_csv(generation_csv_path, parse_dates=[time_col])

    n_before = len(df)
    df_filtered = df[df[time_col].isin(day_times)].copy()
    n_after = len(df_filtered)

    print(f"  Rows before masking: {n_before:,}")
    print(f"  Rows after masking:  {n_after:,}  ({n_after / n_before * 100:.1f}%)")

    df_filtered.to_csv(out_csv_path, index=False)
    print(f"  Saved to: {out_csv_path}")
