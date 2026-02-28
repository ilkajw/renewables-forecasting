import requests
import json
import string
import time, sys

from datetime import datetime
from pathlib import Path

from renewables_forecasting.config.data_sources import SMARD_SOLAR_GEN_TIMESTAMPS_URL


def check_template(template: str, required_keys: list):
    keys = [key for _, key, _, _ in string.Formatter().parse(template) if key is not None]
    assert all(k in keys for k in required_keys), f"Template missing required keys: {required_keys}"


def resolve_smard_timestamps(start: datetime, end: datetime):

    response = requests.get(url=SMARD_SOLAR_GEN_TIMESTAMPS_URL)
    response.raise_for_status()
    timestamps_dict = response.json()
    timestamps_all = timestamps_dict["timestamps"]

    start_stamp = start.timestamp() * 1000  # In millisecs
    end_stamp = end.timestamp() * 1000
    stamps = [ts for ts in timestamps_all if start_stamp <= ts <= end_stamp]
    # print(stamps)
    return stamps


def download_smard_solar_gen(
        url_template: str,
        out_path_template: Path,
        start: datetime,
        end: datetime,
        read_timeout: int,
        connect_timeout: int
):

    check_template(url_template, ["timestamp"])  # Check for expected template string key word
    check_template(str(out_path_template), ["start", "end"])  # out_path_template is Path from joining with pathin paths.py

    out_path = Path(str(out_path_template).format(start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = resolve_smard_timestamps(start, end)  # Get list of timestamps relevant to our time interval

    data = {"series": []}

    for timestamp in timestamps:

        url = url_template.format(timestamp=timestamp)

        response = requests.get(url, timeout=(connect_timeout, read_timeout))  # Call API for current timestamp
        response.raise_for_status()
        data["series"].extend(response.json()["series"])  # Concatenate (timestamp, generation volume) series
        time.sleep(0.5)  # At least half a sec between calls

    with open(out_path, "w") as out:
        json.dump(data, out)  # Save


