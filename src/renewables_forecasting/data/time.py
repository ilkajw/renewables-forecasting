import pandas as pd
import numpy as np
from pathlib import Path


def build_cyclical_time_features(
        start: str,
        end: str,
        out_path: Path,
) -> None:
    """
    Builds a CSV of cyclical time features (sin/cos encodings) for every hour
    in the requested period and saves it to disk.

    Each timestamp is encoded as four features:
      - doy_sin, doy_cos: position in the year (day-of-year cycle)
      - hod_sin, hod_cos: position in the day (hour-of-day cycle)

    Together each sin/cos pair uniquely encodes every timestamp as a point
    on the unit circle, with smooth wraparound at year-end and midnight.
    Timestamps are in UTC, consistent with the rest of the pipeline.

    Parameters
    ----------
    start:
        Start of the period, inclusive. Format: 'YYYY-MM-DD HH:MM'.
    end:
        End of the period, inclusive. Format: 'YYYY-MM-DD HH:MM'.
    out_path:
        Path to write the output CSV file.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    times = pd.date_range(start=start, end=end, freq="h")

    day_of_year = times.day_of_year
    days_in_year = times.is_leap_year.map({True: 366, False: 365})
    hour_of_day = times.hour

    df = pd.DataFrame({
        "time": times,
        "doy_sin": np.sin(2 * np.pi * day_of_year / days_in_year),
        "doy_cos": np.cos(2 * np.pi * day_of_year / days_in_year),
        "hod_sin": np.sin(2 * np.pi * hour_of_day / 24),
        "hod_cos": np.cos(2 * np.pi * hour_of_day / 24),
    })

    df.to_csv(out_path, index=False)

    print(f"  Cyclical time features saved to: {out_path}")
    print(f"  Period: {times[0]} to {times[-1]}")
    print(f"  Total hours: {len(df):,}")
