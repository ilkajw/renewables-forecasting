from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

DWD_RAW_DATA_DIR = RAW_DATA_DIR / "dwd"
MASTR_RAW_DATA_DIR = RAW_DATA_DIR / "mastr"
SMARD_RAW_DATA_DIR = RAW_DATA_DIR / "smard"

MASTR_GESAMTDATENUEBERSICHT_PATH = MASTR_RAW_DATA_DIR / "mastr_gesamtdatenuebersicht.zip"
MASTR_SOLAR_PLANTS_CSV_PATH = MASTR_RAW_DATA_DIR / "solar/einheiten_solar.csv"
MASTR_SOLAR_PLANTS_SQLITE_PATH = MASTR_RAW_DATA_DIR / "solar/einheiten_solar.db"

SMARD_SOLAR_GENERATION_SERIES_JSON_TEMPLATE = SMARD_RAW_DATA_DIR / "solar/solar_series_{start}_{end}.json"
