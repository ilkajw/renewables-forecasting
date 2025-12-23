import requests
from pathlib import Path
from datetime import date
from typing import Dict

from renewables_forecasting.config.technologies import VariableConfig


def _download_file(url, target):
    response = requests.get(url, stream=True, timeout=1000)
    # Fail loudly
    response.raise_for_status()
    with open(target, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _goto_next_month(current: date):
    year = current.year
    month = current.month
    month += 1
    if month == 13:
        month = 1
        year += 1
    return date(year, month, 1)


def download_cosmo_rea6(
    variables: Dict[str, VariableConfig],
    start: date,
    end: date,
    output_dir: Path,
):
    # Working with months only.
    # No unambiguity by enforcing use of first of month
    assert start.day == 1
    assert end.day == 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Directory per var
    for var in variables.values():
        d = start
        while d <= end:
            year = d.year
            month = d.month

            # Construct server file name
            filename = var.filename_pattern.format(var=var.name, year=year, month=month)
            url = f"{var.base_url}/{filename}"

            # Define location to store to
            target = output_dir / var.name / filename
            target.parent.mkdir(parents=True, exist_ok=True)

            # Check for existing file. If so, skip
            if target.exists():
                d = _goto_next_month(d)
                continue

            _download_file(url, target)

            # Increment
            d = _goto_next_month(d)


