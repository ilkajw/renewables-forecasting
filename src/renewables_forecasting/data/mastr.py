import requests
import zipfile
import csv
import pandas as pd
import sqlite3

import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from datetime import date
from pathlib import Path
from typing import List, Dict, Tuple

from renewables_forecasting.config.data_constants import MISSING_PLZS_TO_COORDS_DICT


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

                        # Leave out foreign plants with postal codes of length 4 and 6
                        plz = elem.findtext("Postleitzahl")

                        # Take in units with missing plz. For wind only, as there is no solar plant without plz
                        if plz is not None and len(plz) != 5:
                            continue

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


def csv_to_sql(csv_path: Path, sql_path: Path, name_table: str, overwrite: bool = True):

    if overwrite:
        sql_path.unlink(missing_ok=True)  # Delete db

    with sqlite3.connect(sql_path) as conn:
        for chunk in pd.read_csv(csv_path, chunksize=100_000, dtype={'Postleitzahl': str}):
            chunk.to_sql(name_table, conn, if_exists='append', index=False)


def resolve_plant_commissioning_dates(
        plants_csv_path: Path,
        out_csv_path: Path,
        rejected_csv_path: Path,
) -> None:
    """
    Preprocessing step between raw MaStR CSV ingestion and coord assignment.

    Reads the filtered solar plant CSV, handles the
    InbetriebnahmedatumAmAktuellenStandort field, and writes two outputs:

      out_csv_path:
          Cleaned plant data with an added InbetriebnahmedatumEffektiv column.
          This is the date the grid building code should use instead of
          Inbetriebnahmedatum directly. For relocated plants with a valid
          (non-negative) gap it equals InbetriebnahmedatumAmAktuellenStandort;
          for all other plants it equals Inbetriebnahmedatum.

      rejected_csv_path:
          Plants with a negative gap (InbetriebnahmedatumAmAktuellenStandort <
          Inbetriebnahmedatum) — these have internally inconsistent dates and
          are excluded from the cleaned output. Saved separately so statistics
          can be recovered later.

    The original Inbetriebnahmedatum column is preserved unchanged in both
    outputs so the substitution is fully auditable.

    Parameters
    ----------
    plants_csv_path:
        Path to the filtered solar plant CSV produced by
        filter_xmls_from_gesamtdatenuebersicht_to_csv().
    out_csv_path:
        Path to write the cleaned plant CSV.
    rejected_csv_path:
        Path to write the rejected (bad-data) plant CSV.
    """

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────

    print("Loading raw plant data from CSV ...")

    df = pd.read_csv(plants_csv_path, dtype={"Postleitzahl": str})

    print(f"  Total plants loaded: {len(df):,}")

    # Explicitly convert date columns — read_csv can silently produce object
    # dtype when values are mixed or empty, which causes date arithmetic to fail
    for col in ["Inbetriebnahmedatum", "InbetriebnahmedatumAmAktuellenStandort"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Nettonennleistung"] = pd.to_numeric(df["Nettonennleistung"], errors="coerce")

    # ── Identify plant categories ─────────────────────────────────────────────

    has_reloc = df["InbetriebnahmedatumAmAktuellenStandort"].notna()

    # Compute gap only where relocation date is present
    df.loc[has_reloc, "gap_days"] = (
            df.loc[has_reloc, "InbetriebnahmedatumAmAktuellenStandort"]
            - df.loc[has_reloc, "Inbetriebnahmedatum"]
    ).dt.days

    # Three categories:
    #   1. No relocation date — standard plants, use Inbetriebnahmedatum as-is
    #   2. Valid relocation (gap >= 0) — use InbetriebnahmedatumAmAktuellenStandort
    #   3. Invalid relocation (gap < 0) — bad data, exclude entirely
    no_reloc = ~has_reloc
    valid_reloc = has_reloc & (df["gap_days"] >= 0)
    invalid_reloc = has_reloc & (df["gap_days"] < 0)

    print(f"\n  Plant categories:")
    print(f"    No relocation date:       {no_reloc.sum():>10,}")
    print(f"    Valid relocation (>=0d):  {valid_reloc.sum():>10,}")
    print(f"    Invalid relocation (<0d): {invalid_reloc.sum():>10,}  -> rejected")

    # ── Reject bad-data plants ────────────────────────────────────────────────

    df_rejected = df[invalid_reloc].copy()

    print(f"\n  Rejected plants:")
    print(f"    Count:                   {len(df_rejected):,}")
    print(f"    Total Nettonennleistung: {df_rejected['Nettonennleistung'].sum() / 1e3:.2f} MW")

    df_rejected.drop(columns=["gap_days"]).to_csv(rejected_csv_path, index=False)
    print(f"  Rejected plants saved to: {rejected_csv_path}")

    # ── Build InbetriebnahmedatumEffektiv ─────────────────────────────────────
    # Default to Inbetriebnahmedatum for all plants, then overwrite for valid
    # relocated plants with InbetriebnahmedatumAmAktuellenStandort.
    # The original Inbetriebnahmedatum column is never modified.

    df_clean = df[~invalid_reloc].copy()

    # Start with Inbetriebnahmedatum as the default for all plants
    df_clean["InbetriebnahmedatumEffektiv"] = df_clean["Inbetriebnahmedatum"]

    # Overwrite for valid relocated plants using their current-location date.
    # Use EinheitMastrNummer as the alignment key to avoid any index
    # misalignment after the invalid_reloc filter above.
    reloc_mask = df_clean["gap_days"] >= 0
    df_clean.loc[reloc_mask, "InbetriebnahmedatumEffektiv"] = (
        df_clean.loc[reloc_mask, "InbetriebnahmedatumAmAktuellenStandort"]
    )

    # ── Sanity check ──────────────────────────────────────────────────────────

    n_missing_effektiv = (
            df_clean["InbetriebnahmedatumEffektiv"].isna()
            & df_clean["Inbetriebnahmedatum"].notna()
    ).sum()
    if n_missing_effektiv > 0:
        print(f"\n  WARNING: {n_missing_effektiv} plants have NaT "
              f"InbetriebnahmedatumEffektiv despite valid Inbetriebnahmedatum "
              f"— check preprocessing logic.")

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n  InbetriebnahmedatumEffektiv substitutions:")
    print(f"    Plants using Inbetriebnahmedatum (unchanged):        "
          f"{no_reloc.sum():>10,}")
    print(f"    Plants using InbetriebnahmedatumAmAktuellenStandort: "
          f"{valid_reloc.sum():>10,}")

    # ── Write cleaned output ──────────────────────────────────────────────────

    # Drop the temporary gap_days column — only needed for categorisation
    df_clean = df_clean.drop(columns=["gap_days"])

    df_clean.to_csv(out_csv_path, index=False)
    print(f"\n  Cleaned plants saved to: {out_csv_path}")
    print(f"  Total plants in cleaned output: {len(df_clean):,}")


def download_geonames_postal_code_data(
        url: str,
        out_path: Path,
        connect_timeout: int = 30,
        read_timeout: int = 600,
) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # tmp file for partial downloads avoids corrupted zip at out_path
    tmp_path = out_path.with_suffix(".tmp")

    try:
        with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as response:
            response.raise_for_status()

            with open(tmp_path, "wb") as tmp:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        tmp.write(chunk)

            tmp_path.replace(out_path)

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _get_plz_to_lat_lon_mapping(
        plz_zip_path: Path,
        file_name: str = "DE.txt"
) -> Dict[str, Tuple[float, float]]:

    # Open geonames zip
    with zipfile.ZipFile(plz_zip_path) as zf:
        with zf.open(file_name) as f:
            # get data frame from txt file
            plz_df = pd.read_csv(
                f,
                sep='\t',
                header=None,
                names=['country', 'plz', 'city', 'state', 'state_code',
                       'district', 'dist_code', 'county', 'county_code',
                       'lat', 'lon', 'accuracy'],
                dtype={'plz': str}
            )

    # Filter out company and institution entries
    plz_df = plz_df[plz_df['state_code'].str.isalpha().fillna(False)]

    # Get plz: (lat, lon)) dict
    plz_to_coords = dict(zip(plz_df['plz'].astype(str),
                             zip(plz_df['lat'], plz_df['lon'])))

    # Add missing PLZs that were looked up manually
    plz_to_coords = MISSING_PLZS_TO_COORDS_DICT | plz_to_coords

    return plz_to_coords


def assign_coords_to_plants(
        plz_data_path: Path,
        plants_csv_path: Path,
        out_path: Path,
        keep_existing_coords: bool = False,
):
    plz_to_coords_dict = _get_plz_to_lat_lon_mapping(plz_data_path)

    plants_df = pd.read_csv(plants_csv_path, dtype={'Postleitzahl': str})

    # Warn about inconsistent coords
    for idx, plant in plants_df.iterrows():
        if pd.isna(plant["Breitengrad"]) and not pd.isna(plant["Laengengrad"]):
            print(f"Got Laengengrad but not Breitengrad for {plant['EinheitMastrNummer']}, overwriting both.")
        elif pd.isna(plant["Laengengrad"]) and not pd.isna(plant["Breitengrad"]):
            print(f"Got Breitengrad but not Laengengrad for {plant['EinheitMastrNummer']}, overwriting both.")

    # Vectorized coord assignment for missing values
    if keep_existing_coords:
        missing = pd.isna(plants_df["Breitengrad"]) | pd.isna(plants_df["Laengengrad"])
        coords = plants_df.loc[missing, "Postleitzahl"].astype(str).map(plz_to_coords_dict)
        plants_df.loc[missing, "Breitengrad"] = coords.map(lambda x: x[0] if pd.notna(x) else None)
        plants_df.loc[missing, "Laengengrad"] = coords.map(lambda x: x[1] if pd.notna(x) else None)

    else:
        # Extract coords with plant indices
        coords = plants_df["Postleitzahl"].astype(str).map(plz_to_coords_dict)
        plants_df["Breitengrad"] = coords.map(lambda x: x[0] if pd.notna(x) else None)
        plants_df["Laengengrad"] = coords.map(lambda x: x[1] if pd.notna(x) else None)

    plants_df.to_csv(out_path, index=False)
