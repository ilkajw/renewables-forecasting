from pathlib import Path
from renewables_forecasting.config.data_sources import MASTR_GESAMTDATENUEBERSICHT_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

# ── ERA5 weather data ───────────────────────────────────────────────────────────────
ERA5_RAW_DATA_DIR = RAW_DATA_DIR / "era5"
ERA5_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / "era5"


# ── Solar ──────────────
ERA5_RAW_SOLAR_DATA_DIR = ERA5_RAW_DATA_DIR / "solar"
ERA5_PROCESSED_SOLAR_DATA_DIR = ERA5_PROCESSED_DATA_DIR / "solar"
ERA5_CET_SOLAR_DATA_DIR = ERA5_PROCESSED_SOLAR_DATA_DIR / "cet"
ERA5_MASKED_SOLAR_DATA_DIR = ERA5_PROCESSED_SOLAR_DATA_DIR / "daylight_masked"

# ── Wind ──────────────
ERA5_RAW_WIND_DATA_DIR = ERA5_RAW_DATA_DIR / " wind"
ERA5_PROCESSED_WIND_DATA_DIR = ERA5_PROCESSED_DATA_DIR / "wind"

# For viz
ERA5_EUROPE_DIR = ERA5_RAW_DATA_DIR / "europe"
ERA5_EUROPE_SOLAR_DATA_DIR = ERA5_EUROPE_DIR / "solar"
ERA5_EUROPE_WIND_DATA_DIR = ERA5_EUROPE_DIR / "wind"


# ── MaStR plant data ───────────────────────────────────────────────────────────────

MASTR_DATA_DIR_RAW = RAW_DATA_DIR / "mastr"
MASTR_DATA_DIR_PROCESSED = PROCESSED_DATA_DIR / "mastr"

MASTR_GESAMTDATENUEBERSICHT_PATH = MASTR_DATA_DIR_RAW / f"Gesamtdatenexport_{MASTR_GESAMTDATENUEBERSICHT_VERSION}.zip"

# ── Solar plants data ─────────

MASTR_SOLAR_PLANTS_FILTERED_CSV = MASTR_DATA_DIR_PROCESSED / \
                                  f"solar/einheiten_solar_{MASTR_GESAMTDATENUEBERSICHT_VERSION}.csv"
MASTR_SOLAR_PLANTS_FILTERED_SQLITE = MASTR_DATA_DIR_PROCESSED / \
                                     f"solar/einheiten_solar_{MASTR_GESAMTDATENUEBERSICHT_VERSION}.db"

# Solar plants with explicit effective start dates
MASTR_SOLAR_PLANTS_EFFECTIVE_START_DATE_CSV = MASTR_DATA_DIR_PROCESSED / \
                                f"solar/einheiten_solar_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_comm_date_resolved.csv"
MASTR_SOLAR_PLANTS_REJECTED_CSV = MASTR_DATA_DIR_PROCESSED / \
                                  f"solar/einheiten_solar_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_comm_date_rejected.csv"

# Filtered solar plants data with added coords
MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV = MASTR_DATA_DIR_PROCESSED / \
                                         f"solar/einheiten_solar_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_with_coords.csv"

MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_SQLITE = MASTR_DATA_DIR_PROCESSED / \
                                        f"solar/einheiten_solar_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_with_coords.db"

# ── Wind plants data ─────────

MASTR_WIND_PLANTS_FILTERED_CSV = MASTR_DATA_DIR_PROCESSED / \
                                 f"wind/einheiten_wind_{MASTR_GESAMTDATENUEBERSICHT_VERSION}.csv"
MASTR_WIND_PLANTS_FILTERED_SQLITE = MASTR_DATA_DIR_PROCESSED / \
                                    f"wind/einheiten_wind_{MASTR_GESAMTDATENUEBERSICHT_VERSION}.db"

# Wind plants with explicit effective start dates
MASTR_WIND_PLANTS_EFFECTIVE_START_DATE_CSV = MASTR_DATA_DIR_PROCESSED / \
                                 f"wind/einheiten_wind_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_comm_date_resolved.csv"
MASTR_WIND_PLANTS_REJECTED_CSV = MASTR_DATA_DIR_PROCESSED / \
                                 f"wind/einheiten_wind_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_comm_date_rejected.csv"

# Filtered wind plants data with added coords
MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV = MASTR_DATA_DIR_PROCESSED / \
                        f"wind/einheiten_wind_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_comm_date_resolved_with_coords.csv"
MASTR_SOLAR_PLANTS_EFFECTIVE_START_WITH_COORDS_SQLITE = MASTR_DATA_DIR_PROCESSED / \
                        f"wind/einheiten_wind_{MASTR_GESAMTDATENUEBERSICHT_VERSION}_comm_date_resolved_with_coords.db"


# ── SMARD generation data ───────────────────────────────────────────────────────────────

SMARD_RAW_DATA_DIR = RAW_DATA_DIR / "smard"

# ── Solar ─────────

SMARD_SOLAR_DATA_DIR_RAW = SMARD_RAW_DATA_DIR / "solar"
SMARD_SOLAR_GENERATION_SERIES_JSON = SMARD_SOLAR_DATA_DIR_RAW / "smard_solar_generation.json"
SMARD_SOLAR_GENERATION_SERIES_CSV = SMARD_SOLAR_DATA_DIR_RAW / "smard_solar_generation.json"
SMARD_SOLAR_GENERATION_SERIES_MASKED_CSV = SMARD_SOLAR_DATA_DIR_RAW / "smard_solar_generation_masked.json"


# ── Geonames postal code data ───────────────────────────────────────────────────────────────
GEONAMES_POSTAL_CODE_DATA = RAW_DATA_DIR / "geonames/geonames_postal_code_data_DE.zip"


# ── Capacity grids ───────────────────────────────────────────────────────────────
CAPACITY_GRIDS_DIR = PROCESSED_DATA_DIR / "capacity_grids"

SOLAR_CAPACITY_GRIDS_ZARR_STORE = CAPACITY_GRIDS_DIR / "solar"
WIND_CAPACITY_GRIDS_ZARR_STORE = CAPACITY_GRIDS_DIR / "wind"

GRID_REFERENCE_DS_STORE = ERA5_RAW_SOLAR_DATA_DIR / "ssrd/ssrd_2015-01.nc"

# ── Daylight boolean mask ───────────────────────────────────────────────────────────────
DAYLIGHT_MASK_NETCDF_PATH = PROCESSED_DATA_DIR / "daylight_mask"

# ── DWD COSMO REA6 weather data ───────────────────────────────────────────────────────────────
DWD_DATA_DIR_RAW = RAW_DATA_DIR / "dwd"
DWD_DATA_DIR_PROCESSED = PROCESSED_DATA_DIR / "dwd"
DWD_DATA_DIR_FEATURES = FEATURES_DATA_DIR / "dwd"

# Solar
DWD_SOLAR_DATA_DIR_RAW = DWD_DATA_DIR_RAW / "solar"
DWD_SOLAR_DATA_DIR_PROCESSED = DWD_DATA_DIR_PROCESSED / "solar"
DWD_SOLAR_DATA_DIR_FEATURES = DWD_DATA_DIR_FEATURES / "solar"

# Wind
DWD_WIND_DATA_DIR_RAW = DWD_DATA_DIR_RAW / "wind"
DWD_WIND_DATA_DIR_FEATURES = DWD_DATA_DIR_FEATURES / "wind"

# Regrid
TARGET_GRID_ZARR_STORE = PROCESSED_DATA_DIR / "target_grid"  # <-- obsolete with era5 data