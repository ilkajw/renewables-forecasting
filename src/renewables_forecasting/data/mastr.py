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
        url: str, out_path: Path, overwrite: bool = False, connect_timeout: int = 30, read_timeout: int = 600
) -> Path:

    out_path.parent.mkdir(parents=True, exist_ok=True)

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


def filter_xmls_from_gesamtdatenuebersicht_to_csv(
        zip_path: Path,
        naming_files: str,
        naming_units: str,
        start: date,
        end: date,
        variables: List[str],
        exclude_filters: dict[str, str] | None = None,
        out_csv: Path = Path("einheiten_solar.csv")
) -> None:

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Open zip to read solar xml from. Open solar xml and output file
    with zipfile.ZipFile(zip_path) as zf, open(out_csv, "w", newline="", encoding="utf-8") as out:

        # Prep out_csv file
        w = csv.writer(out)
        w.writerow(variables)

        # Collect all xmls with naming 'EinheitenSolar_x.xml' or 'EinheitenWind_x.xml', x=[1, 49] from zip
        xmls = [n for n in zf.namelist() if naming_files in n]

        for f in xmls:
            with zf.open(f) as curr_xml:

                # iterparse stopping after every full element, loop through xml tree elements with index (event, elem)
                for _, elem in ET.iterparse(curr_xml, events=("end",)):

                    # Filter for EinheitSolar or EinheitWind elements, leaving out any meta data
                    if elem.tag.endswith(naming_units):

                        # Get dates to derive periods online and offline
                        inbetriebnahme_d = elem.findtext("Inbetriebnahmedatum")
                        endg_stilllegung_d = elem.findtext("DatumEndgueltigeStilllegung")
                        vorr_stilllegung_d = elem.findtext("DatumBeginnVoruebergehendeStilllegung")
                        wiederaufnahme_d = elem.findtext("DatumWiederaufnahmeBetrieb")

                        # Get python date types
                        inbetriebnahme_d = date.fromisoformat(inbetriebnahme_d) if inbetriebnahme_d else None
                        endg_stilllegung_d = date.fromisoformat(endg_stilllegung_d) if endg_stilllegung_d else None
                        vorr_stilllegung_d = date.fromisoformat(vorr_stilllegung_d) if vorr_stilllegung_d else None
                        wiederaufnahme_d = date.fromisoformat(wiederaufnahme_d) if wiederaufnahme_d else None

                        not_decomm_before_start = endg_stilllegung_d is None or start <= endg_stilllegung_d
                        comm_before_end = inbetriebnahme_d is None or inbetriebnahme_d <= end
                        only_temp_off = (
                            vorr_stilllegung_d is None  # never temporarily off
                            or start < vorr_stilllegung_d  # went off during period
                            or (wiederaufnahme_d is not None and wiederaufnahme_d <= end)  # came back on before end
                        )

                        # Extract variables for solar units online during period:
                        if not_decomm_before_start and comm_before_end and only_temp_off:

                            # Skip unit if any exclude-filter applies
                            if exclude_filters and any(elem.findtext(k) == v for k, v in exclude_filters.items()):
                                elem.clear()  # Free memory
                                continue

                            fields = [elem.findtext(var) or "" for var in variables]

                            # Write unit data to csv file
                            w.writerow(fields)

                        elem.clear()  # Free memory


def filter_wind_xml_from_gesamtdatenuebersicht_to_csv(
        zip_path: Path,
        inbetriebnahme_start: date,
        inbetriebnahme_end: date,
        variables: List[str],
        out_csv: Path = Path("einheiten_wind.csv")
) -> None:

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Open zip to read solar xml from. Open solar xml and output file
    with zipfile.ZipFile(zip_path) as zf, open(out_csv, "w", newline="", encoding="utf-8") as out:

        # Prep out_csv file
        w = csv.writer(out)
        w.writerow(variables)

        # Collect all files with naming 'EinheitenWind_x.xml', x=[1, 49] from zip
        wind_xmls = [n for n in zf.namelist() if "EinheitenWind" in n]

        # iterparse stopping after every full element, loop through xml tree elements with index (event, elem)
        for f in wind_xmls:
            with zf.open(f) as curr_xml:
                for _, elem in ET.iterparse(curr_xml, events=("end",)):

                    # Filter for EinheitSolar elements, leaving out any meta data
                    if elem.tag.endswith("EinheitWind"):

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

