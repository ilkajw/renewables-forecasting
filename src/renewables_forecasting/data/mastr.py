import requests
import zipfile
import csv
import pandas as pd
import sqlite3

import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from datetime import date
from pathlib import Path
from typing import List


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name


def download_mastr_gesamtdatenuebersicht(
        url, target_dir: Path, overwrite=False, connect_timeout=30, read_timeout=600
) -> Path:

    target_dir.mkdir(parents=True, exist_ok=True)
    fname = filename_from_url(url)
    out_path = target_dir / fname

    # Overwrite behaviour
    if out_path.exists() and not overwrite:
        return out_path

    # tmp file for partial downloads avoids corrupted zip at out_path
    tmp_path = out_path.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as response:
            # Fail loudly
            response.raise_for_status()

            with open(tmp_path, "wb") as tmp:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        tmp.write(chunk)

            # Swap label to out_path
            tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)  # Remove tmp on error
        raise

    return out_path


def filter_solar_xml_from_gesamtdatenuebersicht_to_csv(
        zip_path: Path,
        inbetriebnahme_start: date,
        inbetriebnahme_end: date,
        variables: List[str],
        out_csv: Path = Path("einheiten_solar.csv")
) -> None:

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Open zip to read solar xml from. Open solar xml and output file
    with zipfile.ZipFile(zip_path) as zf, open(out_csv, "w", newline="", encoding="utf-8") as out:

        # Prep out_csv file
        w = csv.writer(out)
        w.writerow(variables)

        # Collect all files with naming 'EinheitenSolar_x.xml', x=[1, 49] from zip
        solar_xmls = [n for n in zf.namelist() if "EinheitenSolar" in n]

        # iterparse stopping after every full element, loop through xml tree elements with index (event, elem)
        for f in solar_xmls:
            with zf.open(f) as curr_xml:
                for _, elem in ET.iterparse(curr_xml, events=("end",)):

                    # Filter for EinheitSolar elements, leaving out any meta data
                    if elem.tag.endswith("EinheitSolar"):

                        # Get Inbetriebnahmedatum to filter for plants of interest
                        d = elem.findtext("Inbetriebnahmedatum")
                        d = date.fromisoformat(d) if d else None  # Get python date type

                        # Extract variables for solar units of interest
                        if d is None or (inbetriebnahme_start <= d <= inbetriebnahme_end):
                            fields = [elem.findtext(var) or "" for var in variables]

                            # Write unit data to csv file
                            w.writerow(fields)

                        elem.clear()  # Free memory


def csv_to_sql(csv_path: Path, sql_path: Path, overwrite: bool = True):

    if overwrite:
        sql_path.unlink(missing_ok=True)  # Delete db

    with sqlite3.connect(sql_path) as conn:
        for chunk in pd.read_csv(csv_path, chunksize=100_000):
            chunk.to_sql("einheiten_solar", conn, if_exists='append', index=False)

